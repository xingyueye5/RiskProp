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
from mmengine.model import BaseModel
from mmaction.models import BaseRecognizer, BaseHead

from .models_transformer import TransformerDecoder, TransformerDecoderLayer
import random


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
        clip_len: int = 5,
        num_clips: int = 50,
        two_stream: bool = False,
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
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.two_stream = two_stream
        self.with_rnn = with_rnn
        self.with_decoder = with_decoder
        self.anticipate_len = anticipate_len
        self.label_with = label_with
        assert label_with in ["fix", "annotation", "constraint"]
        self.epoch = None

        if self.with_rnn:
            self.rnn = nn.LSTM(
                input_size=self.in_channels,
                hidden_size=self.in_channels,
                num_layers=rnn_num_layers,
                bidirectional=rnn_bidirectional,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0,
                batch_first=True,
            )
            if self.two_stream:
                self.rnn_flow = nn.LSTM(
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
            if self.two_stream:
                self.decoder_flow = TransformerDecoder(
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
        if self.two_stream:
            self.fc_cls_flow = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self) -> None:
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, **kwargs) -> None:
        # [N, C, T, H, W]
        x = self.avg_pool2d(x).squeeze(-1).squeeze(-1)
        if x.dim() == 2 and self.with_decoder:
            idx = torch.arange(x.shape[0], device=x.device).unsqueeze(-1) + torch.arange(self.clip_len, device=x.device)
            idx = (idx - self.clip_len + 1).clamp(0, x.shape[0] - 1)
            x = x[idx]
        elif x.dim() == 3 and not self.with_decoder:
            x = x.mean(dim=-1)
        elif x.dim() == 3 and self.with_decoder:
            x = x.permute(0, 2, 1)
        if self.two_stream:
            x_flow = x[:, self.in_channels :]
            x = x[:, : self.in_channels]
            # [N, C]
            if self.with_rnn:
                x_flow = x_flow.reshape(-1, self.num_clips, self.in_channels)
                x_flow, _ = self.rnn_flow(x_flow)
                x_flow = x_flow.reshape(-1, self.in_channels)
            # [N, C]
            if self.with_decoder:
                tgt_pos = self.pos()[:, self.clip_len - 1 : self.anticipate_len + self.clip_len - 1, :]
                memory_pos = self.pos()[:, : self.clip_len, :]
                tgt = torch.zeros_like(tgt_pos).expand(x_flow.shape[0], -1, -1)
                x_flow = self.decoder_flow(tgt, x_flow, tgt_pos=tgt_pos, memory_pos=memory_pos)
            # [N, C]
            x_flow = self.dropout(x_flow)
            # [N, C]
            cls_score_flow = self.fc_cls_flow(x_flow)
            # [N, num_classes]
        # [N, C]
        if self.with_rnn:
            x = x.reshape(-1, self.num_clips, self.in_channels)
            x, _ = self.rnn(x)
            x = x.reshape(-1, self.in_channels)
        # [N, C]
        if self.with_decoder:
            tgt_pos = self.pos()[:, self.clip_len - 1 : self.anticipate_len + self.clip_len - 1, :]
            memory_pos = self.pos()[:, : self.clip_len, :]
            tgt = torch.zeros_like(tgt_pos).expand(x.shape[0], -1, -1)
            x = self.decoder(tgt, x, tgt_pos=tgt_pos, memory_pos=memory_pos)
            # [N, anticipate_len, C]
        # [N, C]
        x = self.dropout(x)
        # [N, C]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return torch.cat([cls_score, cls_score_flow], dim=-1) if self.two_stream else cls_score.squeeze()


    def monotony_loss(self, cls_scores: torch.Tensor, data_samples: SampleList) -> torch.Tensor:
        # 添加全局排序损失（新采样策略）- 对概率进行约束
        M, l = 30, cls_scores.shape[1]  # M为采样对数
        margin_per_frame = 0.01  # 减小margin，适应概率空间
        loss_mono = torch.tensor(0.0, device=cls_scores.device)
        window_count = 0
        
        for i, data_sample in enumerate(data_samples):
            if data_sample.target:
                if l < 15:
                    continue
                global_i, global_j = [], []
                # gap采样范围：10%~90%区间
                min_gap = max(1, int(l * 0.1))
                max_gap = max(min_gap + 1, int(l * 0.9))
                #min_gap,max_gap=1,l-1
                for _ in range(M):
                    gap = random.randint(min_gap, max_gap)
                    p_max = l - gap - 1
                    if p_max <= 0:
                        continue
                    p = random.randint(0, p_max)
                    q = p + gap
                    if q < l:
                        global_i.append(p)
                        global_j.append(q)
                batch_loss = 0
                loss_count = 0
                if len(global_i) > 0:
                    global_i_tensor = torch.tensor(global_i, device=cls_scores.device)
                    global_j_tensor = torch.tensor(global_j, device=cls_scores.device)
                    # 对sigmoid后的概率进行单调性约束
                    prob_i = torch.sigmoid(cls_scores[i, global_i_tensor])
                    prob_j = torch.sigmoid(cls_scores[i, global_j_tensor])
                    # 动态margin：根据gap大小调整
                    dynamic_margin = margin_per_frame * (global_j_tensor - global_i_tensor) / l
                    global_loss = torch.clamp(
                        prob_i - prob_j + dynamic_margin,
                        min=0
                    ).mean()
                    batch_loss += global_loss
                    loss_count += 1
                if loss_count > 0:
                    loss_mono += batch_loss / loss_count
                    window_count += 1
        # 计算平均window loss
        if window_count > 0:
            loss_mono = loss_mono / window_count
        else:
            loss_mono = torch.tensor(0.0, device=cls_scores.device)

        return loss_mono


    def piecewise_mono_loss(self, cls_scores, data_samples, delta0=0.01, M=30):
        """
        cls_scores: [B, T] logits
        data_samples: 样本对象列表（需包含 target）
        delta0: 初始边际 δ0
        M: 每个样本采样对数
        """
        B, T = cls_scores.shape
        loss_total = torch.tensor(0.0, device=cls_scores.device)
        count_total = 0
        min_gap = max(1, int(T * 0.1))
        max_gap = max(min_gap + 1, int(T * 0.9))

        for i, data_sample in enumerate(data_samples):
            if not data_sample.target:
                continue
            if T < 5:
                continue

            pair_losses = []
            for _ in range(M):
                # 随机采样 (i,j)
                gap = random.randint(min_gap, max_gap)
                p_max = T - gap - 1
                if p_max <= 0:
                    continue
                p = random.randint(0, p_max)
                q = p + gap

                # 概率
                prob_i = torch.sigmoid(cls_scores[i, p])
                prob_j = torch.sigmoid(cls_scores[i, q])

                # ū_i:j : 区间不确定性均值 (示例: 基于logit方差或温度缩放)
                # 这里用简单近似: 1 - |p-0.5| * 2 作为单点不确定性，再取均值
                seg_probs = torch.sigmoid(cls_scores[i, p:q+1])
                u_seg = (1 - torch.abs(seg_probs - 0.4) * 2).mean()

                delta_ij = delta0 * gap * (1 - u_seg)

                # ---- 门控权重 w_i:j ----
                # 如果区间跨越自监督变点，则弱化/关闭单调约束
                w_ij = 1.0

                # 计算 hinge loss
                loss_ij = w_ij * torch.clamp(prob_i - prob_j + delta_ij, min=0)
                pair_losses.append(loss_ij)

            if len(pair_losses) > 0:
                loss_total += torch.stack(pair_losses).mean()
                count_total += 1

        if count_total > 0:
            return loss_total / count_total
        else:
            return torch.tensor(0.0, device=cls_scores.device)


    def loss_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> Dict:
        if self.two_stream:
            cls_scores_flow = cls_scores[..., 1]
            cls_scores = cls_scores[..., 0]
        if self.label_with == "constraint":
            cls_scores = cls_scores.reshape(len(data_samples), -1)
            preds, labels = [], []
            for i, data_sample in enumerate(data_samples):
                if data_sample.target:
                    preds.append(cls_scores[i, :5])
                    preds.append(cls_scores[i, -5:])
                    labels.append(torch.zeros_like(cls_scores[i, :5]))
                    labels.append(torch.ones_like(cls_scores[i, -5:]))
                else:
                    preds.append(cls_scores[i, :])
                    labels.append(torch.zeros_like(cls_scores[i, :]))
            preds, labels = torch.concatenate(preds), torch.concatenate(labels)

            loss_cls = self.loss_cls(preds, labels, pos_weight=self.pos_weight)
            loss_mse = ((cls_scores[:, 1:].detach() - cls_scores[:, :-1]) ** 2).mean()

            # loss_mse = self.loss_cls(cls_scores[:, :-1], torch.sigmoid(cls_scores[:, 1:].detach()))
            
            #loss_mono = self.monotony_loss(cls_scores, data_samples)
            loss_piecewise = self.piecewise_mono_loss(cls_scores, data_samples)

            # 损失权重配置
            loss_cls_weight= 1.0
            loss_mse_weight= 1.5
            loss_mono_weight= 1.1
            return dict(
                loss_cls=loss_cls * loss_cls_weight,
                loss_mse=loss_mse * loss_mse_weight,
                #loss_mono=loss_mono * loss_mono_weight
                loss_mono=loss_piecewise * loss_mono_weight
            )

            #return dict(loss_cls=loss_cls, loss_mse=loss_mse)
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
            #return dict(loss_cls=loss_cls)
            #cls_scores = cls_scores.reshape(len(data_samples), -1)
            #loss_mse = ((cls_scores[:, 1:].detach()- cls_scores[:, :-1]) ** 2).mean()
            #return dict(loss_cls=loss_cls, loss_mse=loss_mse)
            # loss_mse = self.loss_cls(cls_scores[:, :-1], torch.sigmoid(cls_scores[:, 1:].detach()))
            cls_scores = cls_scores.reshape(len(data_samples), -1)
            #loss_mono = self.monotony_loss(cls_scores, data_samples)

            # 损失权重配置
            loss_cls_weight= 1.0
            loss_mse_weight= 1.5
            loss_mono_weight= 1.1
            return dict(
                loss_cls=loss_cls * loss_cls_weight
                # loss_mse=loss_mse * loss_mse_weight,
                #loss_mono=loss_mono * loss_mono_weight
            )


    def predict_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> SampleList:
        if self.two_stream:
            cls_scores_flow = cls_scores[..., 1]
            cls_scores = cls_scores[..., 0]
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


@MODELS.register_module()
class Recognizer3DTwoStream(BaseRecognizer):
    """3D recognizer model framework."""

    def __init__(self, backbone: ConfigType, cls_head=None, neck=None, train_cfg=None, test_cfg=None, data_preprocessor=None) -> None:
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type="ActionDataPreprocessor")

        BaseModel.__init__(self, data_preprocessor=data_preprocessor)

        # Record the source of the backbone.
        self.backbone_from = "mmaction2"

        self.backbone = MODELS.build(backbone)
        self.backbone_flow = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if cls_head is not None:
            self.cls_head = MODELS.build(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, inputs: torch.Tensor, stage: str = "neck", data_samples=None, test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (torch.Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'neck'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                torch.Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``loss_aux``.
        """

        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()

        num_segs = inputs.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1,) + inputs.shape[2:])
        W = inputs.shape[-1] // 2

        # Check settings of test
        if test_mode:
            x = self.backbone(inputs[:, :, :, :, :W])
            x_flow = self.backbone_flow(inputs[:, :, :, :, W:])
            x = torch.cat([x, x_flow], dim=1)
            if self.with_neck:
                x, _ = self.neck(x)
            return x, loss_predict_kwargs
        else:
            x = self.backbone(inputs[:, :, :, :, :W])
            x_flow = self.backbone_flow(inputs[:, :, :, :, W:])
            x = torch.cat([x, x_flow], dim=1)
            if stage == "backbone":
                return x, loss_predict_kwargs

            loss_aux = dict()
            if self.with_neck:
                x, loss_aux = self.neck(x, data_samples=data_samples)

            # Return features extracted through neck
            loss_predict_kwargs["loss_aux"] = loss_aux
            if stage == "neck":
                return x, loss_predict_kwargs

            # Return raw logits through head.
            if self.with_cls_head and stage == "head":
                x = self.cls_head(x, **loss_predict_kwargs)
                return x, loss_predict_kwargs






