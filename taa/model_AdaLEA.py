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
        # AdaLEA损失函数参数
        use_adaea: bool = True,
        adaea_fps: float = 30.0,
        adaea_lambda_neg: float = 0.2,
        adaea_alpha: float = 1.0,
        adaea_beta: float = 0.1,
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
        
        # AdaLEA损失函数参数
        self.use_adaea = use_adaea
        self.adaea_fps = adaea_fps
        self.adaea_lambda_neg = adaea_lambda_neg
        self.adaea_alpha = adaea_alpha
        self.adaea_beta = adaea_beta

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

    def loss_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> Dict:
        if self.two_stream:
            cls_scores_flow = cls_scores[..., 1]
            cls_scores = cls_scores[..., 0]
        
        # 使用AdaLEA损失函数
        if self.use_adaea:
            loss_adaea = self._compute_adaea_loss(cls_scores, data_samples)
            return dict(loss_adaea=loss_adaea)
        
        # 原有的损失函数逻辑
        if self.label_with == "constraint":
            cls_scores = cls_scores.reshape(len(data_samples), -1)
            preds, labels = [], []
            for i, data_sample in enumerate(data_samples):
                if data_sample.target:
                    preds.append(cls_scores[i, :5])
                    preds.append(cls_scores[i, -5:])
                    labels.append(torch.zeros_like(cls_scores[i, :5]))
                    labels.append(torch.ones_like(cls_scores[i, :5]))
                else:
                    preds.append(cls_scores[i, :])
                    labels.append(torch.zeros_like(cls_scores[i, :]))
            preds, labels = torch.concatenate(preds), torch.concatenate(labels)

            loss_cls = self.loss_cls(preds, labels, pos_weight=self.pos_weight)
            loss_mse = ((cls_scores[:, 1:].detach() - cls_scores[:, :-1]) ** 2).mean()

            # 损失权重配置
            loss_cls_weight = 1.0
            loss_mse_weight = 1.0
            return dict(
                loss_cls=loss_cls * loss_cls_weight,
                loss_mse=loss_mse * loss_mse_weight,
            )
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

            cls_scores = cls_scores.reshape(len(data_samples), -1)
            loss_mse = ((cls_scores[:, 1:].detach()- cls_scores[:, :-1]) ** 2).mean()

            # 损失权重配置
            loss_cls_weight = 1.0
            loss_mse_weight = 1.0
            return dict(
                loss_cls=loss_cls * loss_cls_weight,
                loss_mse=loss_mse * loss_mse_weight,
            )

    def _compute_adaea_loss(self, cls_scores: torch.Tensor, data_samples: SampleList) -> torch.Tensor:
        """
        计算AdaLEA (Adaptive Early Anticipation) 损失函数
        基于论文 "Anticipating Traffic Accidents with Adaptive Loss and Large-scale Incident DB"
        
        数学公式：
        L_AdaLEA = α * L_pos + λ * L_neg + β * L_consistency
        其中：
        - L_pos = -exp(penalty) * log(p_pred)
        - L_neg = -log(1 - p_pred)
        - penalty = -max(0, (toa - time - 1) / fps)
        - L_consistency = max(0, p_t - p_{t+1})
        """
        device = cls_scores.device
        total_loss = torch.tensor(0.0, device=device)
        batch_count = 0
        
        cls_scores = cls_scores.reshape(len(data_samples), -1)

        for i, data_sample in enumerate(data_samples):
            # 获取当前样本的预测分数
            pred = cls_scores[i]  # [seq_len]
            
            if data_sample.target:
                # 正样本：计算时间到事故的帧数
                frame_inds = data_sample.frame_inds.reshape(-1, data_sample.clip_len)[:, -1]
                accident_frame = data_sample.accident_frame
                frame_interval = data_sample.frame_interval
                
                # 确保数据类型正确
                if isinstance(frame_inds, np.ndarray):
                    frame_inds = torch.from_numpy(frame_inds).to(device)
                if isinstance(accident_frame, (int, float)):
                    accident_frame = torch.tensor(accident_frame, device=device)
                if isinstance(frame_interval, (int, float)):
                    frame_interval = torch.tensor(frame_interval, device=device)
                
                # 计算每个时间步到事故的时间
                time_to_accident = (accident_frame - frame_inds) / frame_interval
                
                # 计算正样本的AdaLEA损失
                sample_loss = self._compute_single_adaea_loss(pred, time_to_accident, is_positive=True)
            else:
                # 负样本：计算负样本损失
                sample_loss = self._compute_single_adaea_loss(pred, None, is_positive=False)
            
            total_loss += sample_loss
            batch_count += 1
        
        if batch_count > 0:
            return total_loss / batch_count
        else:
            return torch.tensor(0.0, device=device)
    
    def _compute_single_adaea_loss(self, pred: torch.Tensor, time_to_accident: torch.Tensor, is_positive: bool = True) -> torch.Tensor:
        """
        计算单个样本的AdaLEA损失
        
        Args:
            pred: 预测分数 [seq_len]
            time_to_accident: 到事故的时间 [seq_len]，负样本时为None
            is_positive: 是否为正样本
        """
        device = pred.device
        seq_len = len(pred)
        
        # 使用sigmoid将预测转换为概率
        pred_probs = torch.sigmoid(pred)
        
        if is_positive:
            # 正样本：计算AdaLEA损失
            # 计算指数惩罚项
            # penalty = -max(0, (toa - time - 1) / fps)
            time_indices = torch.arange(seq_len, device=device, dtype=pred.dtype)
            penalty = -torch.max(
                torch.zeros_like(time_to_accident, device=device, dtype=pred.dtype),
                (time_to_accident - time_indices - 1) / self.adaea_fps
            )
            
            # 数值稳定性检查
            penalty = torch.clamp(penalty, min=-10, max=10)  # 防止exp爆炸
            
            # 正样本损失: -exp(penalty) * log(pred_prob)
            pos_loss = -torch.exp(penalty) * torch.log(pred_probs + 1e-8)
            
            # 负样本损失: -log(1 - pred_prob)
            neg_loss = -torch.log(1 - pred_probs + 1e-8)
            
            # 时序一致性损失: max(0, p_t - p_{t+1})
            if seq_len > 1:
                prob_diff = pred_probs[1:] - pred_probs[:-1]
                consistency_loss = torch.clamp(prob_diff, min=0).mean()
            else:
                consistency_loss = torch.tensor(0.0, device=device)
            
            # 总损失: α * L_pos + λ * L_neg + β * L_consistency
            total_loss = (self.adaea_alpha * pos_loss.mean() + 
                         self.adaea_lambda_neg * neg_loss.mean() + 
                         self.adaea_beta * consistency_loss)
            
        else:
            # 负样本：使用BCE损失
            target = torch.zeros_like(pred)
            bce_loss = self.loss_cls(pred, target, pos_weight=self.pos_weight)
            total_loss = bce_loss

        return total_loss


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
