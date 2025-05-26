# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from typing import Dict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList
from mmaction.models import BaseHead

from .models_transformer import TransformerDecoder, TransformerDecoderLayer


@MODELS.register_module()
class AnticipationHead(BaseHead):
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 2048,
        loss_cls: ConfigType = dict(type="BCELossWithLogits"),
        pos_weight: float = 1,
        dropout: float = 0.4,
        init_std: float = 0.01,
        num_clips: int = 50,
        with_rnn: bool = False,
        rnn_num_layers: int = 1,
        rnn_bidirectional: bool = False,
        rnn_dropout: float = 0.5,
        with_decoder: bool = False,
        num_heads: int = 8,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        decoder_dropout: float = 0.1,
        activation: str = "relu",
        anticipate_len: int = 20,
        label_with: str = "fix",
        **kwargs,
    ) -> None:
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.pos_weight = torch.tensor(pos_weight)
        self.init_std = init_std
        self.num_clips = num_clips
        self.with_rnn = with_rnn
        self.with_decoder = with_decoder
        self.anticipate_len = anticipate_len
        self.label_with = label_with
        assert label_with in ["fix", "annotation", "constraint"]

        if self.with_rnn:
            self.rnn = nn.LSTM(
                input_size=self.in_channels,
                hidden_size=self.in_channels,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0,
                batch_first=True,
            )
        if self.with_decoder:
            self.decoder = TransformerDecoder(
                TransformerDecoderLayer(
                    d_model=self.in_channels,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=decoder_dropout,
                    activation=activation,
                    batch_first=True,
                ),
                num_layers=num_decoder_layers,
            )
            self.pos = PositionalEncoding(d_model=self.in_channels)

        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self) -> None:
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, **kwargs) -> None:
        # [N, C, T, H, W]
        if x.dim() == 4:
            x = self.avg_pool2d(x).squeeze(-1).squeeze(-1)
        elif x.dim() == 5:
            x = self.avg_pool3d(x).squeeze(-1).squeeze(-1).squeeze(-1)
        # [N, C]
        if self.with_rnn:
            x = x.reshape(-1, self.num_clips, self.in_channels)
            x, _ = self.rnn(x)
            x = x.reshape(-1, self.in_channels)
        # [N, C]
        if self.with_decoder:
            x = x.unsqueeze(1)
            tgt_pos = self.pos()[:, : self.anticipate_len, :]
            tgt = torch.zeros_like(tgt_pos).expand(x.shape[0], -1, -1)
            x = self.decoder(tgt, x, tgt_pos=tgt_pos, memory_pos=torch.zeros_like(x))
            # [N, anticipate_len, C]
        # [N, C]
        x = self.dropout(x)
        # [N, C]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score.squeeze()

    def loss_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> Dict:
        if self.label_with == "constraint":
            cls_scores = cls_scores.reshape(len(data_samples), -1)
            preds, labels = [], []
            for i, data_sample in enumerate(data_samples):
                if data_sample.target:
                    # preds.append(cls_scores[i, :5])
                    preds.append(cls_scores[i, -1:])
                    # labels.append(torch.zeros_like(cls_scores[i, :5]))
                    labels.append(torch.ones_like(cls_scores[i, -1:]))
                else:
                    preds.append(cls_scores[i, :])
                    labels.append(torch.zeros_like(cls_scores[i, :]))
            preds, labels = torch.concatenate(preds), torch.concatenate(labels)

            loss_cls = self.loss_cls(preds, labels, pos_weight=self.pos_weight)
            # loss_mse = ((cls_scores[:, 1:].detach() - cls_scores[:, :-1]) ** 2).mean()

            loss_mse = self.loss_cls(cls_scores[:, :-1], torch.sigmoid(cls_scores[:, 1:].detach()))

            return dict(loss_cls=loss_cls, loss_mse=loss_mse)
        else:
            labels = []
            for data_sample in data_samples:
                frame_inds = data_sample.frame_inds.reshape(-1, data_sample.clip_len)[:, -1]
                if data_sample.target:
                    if self.with_decoder:
                        index = np.ceil((data_sample.accident_frame - frame_inds) / data_sample.frame_interval)
                        label = np.eye(1000)[index.astype(int), : self.anticipate_len]
                        if self.label_with == "annotation":
                            label = label * (frame_inds >= data_sample.abnormal_start_frame)[:, None]
                    else:
                        if self.label_with == "fix":
                            label = (frame_inds >= data_sample.accident_frame - data_sample.frame_interval * 20) & (
                                frame_inds < data_sample.accident_frame + data_sample.frame_interval
                            )
                        elif self.label_with == "annotation":
                            label = (frame_inds >= data_sample.abnormal_start_frame) & (
                                frame_inds < data_sample.accident_frame + data_sample.frame_interval
                            )
                else:
                    if self.with_decoder:
                        label = np.zeros((len(frame_inds), self.anticipate_len))
                    else:
                        label = np.zeros_like(frame_inds)
                labels.append(torch.from_numpy(label))
            labels = torch.concatenate(labels).float().to(cls_scores.device)

            loss_cls = self.loss_cls(cls_scores, labels, pos_weight=self.pos_weight)

            return dict(loss_cls=loss_cls)

    def predict_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> SampleList:
        cls_scores = cls_scores.reshape(len(data_samples), -1, *cls_scores.shape[1:])
        for i, data_sample in enumerate(data_samples):
            data_sample.set_pred_score(F.sigmoid(cls_scores[i]))
            data_sample.frame_inds = data_sample.frame_inds.reshape(-1, data_sample.clip_len)[:, -1]
        return data_samples


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        # 创建位置编码张量
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model // 2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用 cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # 将位置编码注册为 Buffer
        self.register_buffer("pe", pe)

    def forward(self):
        return self.pe
