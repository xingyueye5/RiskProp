# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from typing import Dict
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


@MODELS.register_module()
class FrameClsHead(BaseHead):
    """Class head for TSN.

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
        labels = torch.zeros_like(cls_scores)
        for i, data_sample in enumerate(data_samples):
            clip_len = data_sample.clip_len
            frame_inds = data_sample.frame_inds
            frame_interval = data_sample.frame_interval
            accident_frame = data_sample.accident_frame
            index = np.argmin(np.abs(frame_inds - accident_frame))
            if np.abs(frame_inds[index] - accident_frame) < frame_interval:
                labels[i * clip_len + index] = 1

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

        loss_cls = self.loss_cls(cls_scores, labels, pos_weight=torch.tensor(10))
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
class SnippetClsHead(BaseHead):
    """The classification head for Snippet.

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

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        loss_cls: ConfigType = dict(type="CrossEntropyLoss"),
        spatial_type: str = "avg",
        dropout_ratio: float = 0.8,
        init_std: float = 0.01,
        **kwargs,
    ) -> None:

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.spatial_type == "avg":
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, **kwargs) -> None:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        T = x.shape[2]
        # [N, C, T, H, W]
        x = x[:, :, int(np.ceil((T - 1) / 2)), :, :].unsqueeze(2)
        # [N, C, 1, H, W]
        x = self.avg_pool(x)
        # [N, C, 1, 1, 1]
        x = x.squeeze()
        # [N, C]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, C]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
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

        loss_cls = self.loss_cls(cls_scores, labels, pos_weight=torch.tensor(1))
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
