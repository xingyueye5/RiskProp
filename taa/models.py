# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from typing import Dict
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList, get_str_type
from mmaction.models import BaseHead
from mmaction.models.heads.base import AvgConsensus

from .models_transformer import TransformerDecoder, TransformerDecoderLayer


@MODELS.register_module()
class FrameClsHead(BaseHead):
    """The classification head for frames.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str or ConfigDict): Pooling type in spatial dimension.
            Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_cls: ConfigType = dict(type="CrossEntropyLoss"),
        pos_weight: float = 1,
        spatial_type: str = "avg",
        consensus: ConfigType = dict(type="AvgConsensus", dim=1),
        dropout_ratio: float = 0.4,
        init_std: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop("type")
        if get_str_type(consensus_type) == "AvgConsensus":
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == "avg":
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.pos_weight = torch.tensor(pos_weight)

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            Tensor: The classification scores for input samples.
        """
        # Notice that num_segs = num_clips * clip_len!!!
        # In TSN, num_clips = 3, clip_len = 1, so num_segs = num_clips = 3
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f"x is tuple {shapes}"
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        # x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N * num_segs, in_channels, 1, 1]
        x = x.squeeze()
        # [N * num_segs, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N * num_segs, in_channels]
        cls_score = self.fc_cls(x)
        # [N * num_segs, num_classes]
        return cls_score.squeeze()

    def loss_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_label for x in data_samples]
        labels = torch.concatenate(labels).float()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(), labels.detach().cpu().numpy(), self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f"top{k}_acc"] = torch.tensor(a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = (1 - self.label_smooth_eps) * labels + self.label_smooth_eps / self.num_classes

        loss_cls = self.loss_cls(cls_scores, labels, pos_weight=self.pos_weight)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses["loss_cls"] = loss_cls
        return losses

    def predict_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        assert len(data_samples) == 1, "Test batch size must be 1!"
        data_samples[0].set_pred_score(F.sigmoid(cls_scores))
        return data_samples


@MODELS.register_module()
class FrameClsHeadWithRNN(FrameClsHead):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_cls: ConfigType = dict(type="CrossEntropyLoss"),
        spatial_type: str = "avg",
        consensus: ConfigType = dict(type="AvgConsensus", dim=1),
        dropout_ratio: float = 0.4,
        init_std: float = 0.01,
        rnn_hidden_size: int = 512,
        rnn_num_layers: int = 1,
        rnn_bidirectional: bool = False,
        rnn_dropout_ratio: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            spatial_type=spatial_type,
            consensus=consensus,
            dropout_ratio=dropout_ratio,
            init_std=init_std,
            **kwargs,
        )

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional
        self.rnn_dropout_ratio = rnn_dropout_ratio

        self.rnn = nn.LSTM(
            input_size=self.in_channels,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            bidirectional=self.rnn_bidirectional,
            dropout=self.rnn_dropout_ratio if self.rnn_num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_cls = nn.Linear(self.rnn_hidden_size, self.num_classes)

    def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            Tensor: The classification scores for input samples.
        """
        # Notice that num_segs = num_clips * clip_len!!!
        # In TSN, num_clips = 3, clip_len = 1, so num_segs = num_clips = 3
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f"x is tuple {shapes}"
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        # x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N * num_segs, in_channels, 1, 1]
        x = x.squeeze()
        # [N * num_segs, in_channels]
        x = x.reshape(-1, num_segs, self.in_channels)
        # [N, num_segs, in_channels]
        x, _ = self.rnn(x)
        # [N, num_segs, rnn_hidden_size]
        x = x.reshape(-1, self.rnn_hidden_size)
        # [N * num_segs, rnn_hidden_size]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N * num_segs, rnn_hidden_size]
        cls_score = self.fc_cls(x)
        # [N * num_segs, num_classes]
        return cls_score.squeeze()


@MODELS.register_module()
class SnippetClsHead(FrameClsHead):
    """The classification head for snippets.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def forward(self, x, **kwargs) -> None:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # [N, C, T, H, W]
        x = x[:, :, -1, :, :]
        # [N, C, H, W]
        x = self.avg_pool(x)
        # [N, C, 1, 1]
        x = x.squeeze()
        # [N, C]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, C]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score.squeeze()

    def predict_by_feat(self, cls_scores: torch.Tensor, data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        assert len(data_samples) == 1, "Test batch size must be 1!"
        data_samples[0].set_pred_score(F.sigmoid(cls_scores))
        data_samples[0].frame_inds = data_samples[0].frame_inds.reshape(-1, data_samples[0].clip_len)[:, -1]
        return data_samples


@MODELS.register_module()
class AnticipationHead(BaseHead):
    """The anticipation head for traffic accident.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str or ConfigDict): Pooling type in spatial dimension.
            Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2048,
        loss_cls: ConfigType = dict(type="CrossEntropyLoss"),
        pos_weight: float = 1,
        spatial_type: str = "avg",
        consensus: ConfigType = dict(type="AvgConsensus", dim=1),
        dropout_ratio: float = 0.4,
        init_std: float = 0.01,
        num_heads: int = 8,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        observed_len: int = 10,
        anticipated_len: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop("type")
        if get_str_type(consensus_type) == "AvgConsensus":
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == "avg":
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.pos_weight = torch.tensor(pos_weight)

        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=in_channels,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )
        self.pos = PositionalEncoding(d_model=in_channels)
        self.observed_len = observed_len
        self.anticipated_len = anticipated_len

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor, num_segs: int, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            Tensor: The classification scores for input samples.
        """
        # Notice that num_segs = num_clips * clip_len!!!
        # [N * num_segs, in_channels, 7, 7]
        x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = x.squeeze()
        # [N * num_segs, in_channels]
        x = x.reshape(-1, num_segs, self.in_channels)
        # [N, num_segs, in_channels]
        offsets = torch.arange(0, num_segs - self.observed_len + 1, device=x.device)  # [num_snippets]

        inds_observed = torch.arange(0, self.observed_len, device=x.device)  # [observed_len]
        inds_observed = (offsets[:, None] + inds_observed[None, :]).flatten()  # [num_snippets * observed_len]
        x_observed = x[:, inds_observed, :].reshape(-1, self.observed_len, self.in_channels)
        # [N * num_snippets, observed_len, in_channels]

        inds_anticipated = torch.arange(0, self.anticipated_len, device=x.device)  # [anticipated_len]
        inds_anticipated = (offsets[:, None] + inds_anticipated[None, :]).flatten()  # [num_snippets * anticipated_len]
        mask_anticipated = inds_anticipated >= num_segs  # [num_snippets * anticipated_len]
        inds_anticipated[mask_anticipated] = -1
        x_anticipated = x[:, inds_anticipated, :].reshape(-1, self.anticipated_len, self.in_channels)
        # [N * num_snippets, anticipated_len, in_channels]

        tgt_pos = self.pos()[:, : self.anticipated_len, :]
        memory_pos = self.pos()[:, : self.observed_len, :]
        tgt = torch.zeros_like(tgt_pos).expand(x_observed.shape[0], -1, -1)
        tgt = self.decoder(tgt, x_observed, tgt_pos=tgt_pos, memory_pos=memory_pos)
        # [N * num_snippets, anticipated_len, in_channels]

        tgt = tgt.reshape(-1, self.in_channels)  # [N * num_snippets * anticipated_len, in_channels]
        x_anticipated = x_anticipated.reshape(-1, self.in_channels)  # [N * num_snippets * anticipated_len, in_channels]
        mask_anticipated = mask_anticipated.repeat(x.shape[0])  # [N * num_snippets * anticipated_len]

        output = tgt
        if self.dropout is not None:
            output = self.dropout(output)
        cls_scores = self.fc_cls(output).squeeze()
        return tgt, x_anticipated, inds_anticipated, mask_anticipated, cls_scores

    def loss_by_feat(self, feats, data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        tgt, x, inds, mask, cls_scores = feats

        # loss_mse = (F.mse_loss(tgt, x.detach(), reduction="none").mean(dim=1) * ~mask).mean()

        labels = [data_sample.gt_label[inds] for data_sample in data_samples]
        labels = torch.concatenate(labels).float()

        # loss_cls = (self.loss_cls(cls_scores, labels, pos_weight=self.pos_weight, reduction="none") * ~mask).mean()
        loss_cls = self.loss_cls(cls_scores, labels, pos_weight=self.pos_weight)

        # return dict(loss_mse=loss_mse, loss_cls=loss_cls)
        return dict(loss_cls=loss_cls)

    def predict_by_feat(self, feats, data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        assert len(data_samples) == 1, "Test batch size must be 1!"
        _, _, _, _, cls_scores = feats
        data_samples[0].set_pred_score(
            F.sigmoid(cls_scores.reshape(-1, self.anticipated_len)[:, self.observed_len - 1 :])
        )
        data_samples[0].set_gt_label(data_samples[0].gt_label[self.observed_len - 1 :])
        data_samples[0].frame_inds = data_samples[0].frame_inds[self.observed_len - 1 :]
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
