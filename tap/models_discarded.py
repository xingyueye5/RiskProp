# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model.weight_init import normal_init
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ConfigType, SampleList, get_str_type
from mmaction.models import BaseHead, ResNet3dSlowFast
from mmaction.models.backbones.resnet3d import Bottleneck3d
from mmaction.models.backbones.resnet3d_slowfast import ResNet3dPathway
from mmaction.models.heads.base import AvgConsensus
from .models import FrameClsHead


@MODELS.register_module()
class FrameClsHeadFromSlowFast(FrameClsHead):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_types: int,
        **kwargs,
    ) -> None:
        super().__init__(num_classes, in_channels, **kwargs)
        if self.dropout_ratio != 0:
            self.dropout_type = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout_type = None
        self.fc_type = nn.Linear(self.in_channels, num_types)

    def forward(self, x: Tuple[Tensor], **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (tuple[torch.Tensor]): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        # x = torch.cat(x, dim=1)
        # [N, in_channels, T, 7, 7]
        x = self.avg_pool(x)
        # [N, in_channels, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1)
        # [N, in_channels, T]
        x = x.permute(0, 2, 1)
        # [N, T, in_channels]
        x = x.reshape(-1, self.in_channels)
        # [N * T, in_channels]
        if self.dropout is not None:
            x_cls = self.dropout(x)
            x_type = self.dropout_type(x)
        # [N * T, in_channels]
        cls_score = self.fc_cls(x_cls)
        type_score = self.fc_type(x_type)
        # [N * T, num_classes]
        return cls_score.squeeze()


@MODELS.register_module()
class FrameResNet3dSlowFast(ResNet3dSlowFast):
    def __init__(
        self,
        pretrained: Optional[str] = None,
        resample_rate: int = 8,
        speed_ratio: int = 8,
        channel_ratio: int = 8,
        slow_pathway: Dict = dict(
            type="ResNet3dPathway",
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
        ),
        fast_pathway: Dict = dict(
            type="ResNet3dPathway",
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
        ),
        init_cfg: Optional[Union[Dict, List[Dict]]] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pretrained = pretrained
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if slow_pathway["lateral"]:
            slow_pathway["speed_ratio"] = speed_ratio
            slow_pathway["channel_ratio"] = channel_ratio

        self.slow_path = MODELS.build(slow_pathway)
        # self.fast_path = MODELS.build(fast_pathway)
        # self.slow_path = MODELS.build(
        #     dict(
        #         type="ResNet",
        #         pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        #         depth=50,
        #         norm_eval=False,
        #     )
        # )

    def forward(self, x: torch.Tensor) -> tuple:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples
                extracted by the backbone.
        """
        # x_slow = nn.functional.interpolate(x, mode="nearest", scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))

        # x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(-1, 3, 224, 224)
        x_slow = x
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        # x_fast = nn.functional.interpolate(
        #     x, mode="nearest", scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0, 1.0)
        # )
        # x_fast = x
        # x_fast = self.fast_path.conv1(x_fast)
        # x_fast = self.fast_path.maxpool(x_fast)

        # if self.slow_path.lateral:
        #     x_fast_lateral = x_fast
        #     x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            # res_layer_fast = getattr(self.fast_path, layer_name)
            # x_fast = res_layer_fast(x_fast)
            # if i != len(self.slow_path.res_layers) - 1 and self.slow_path.lateral:
            #     # No fusion needed in the final stage
            #     x_fast_lateral = x_fast
            #     x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        # out = (x_slow, x_fast)
        return x_slow


@MODELS.register_module()
class FrameResNet3dPathway(ResNet3dPathway):
    def __init__(self, conv1_padding, **kwargs) -> None:
        self.arch_settings[50] = (CustomBottleneck3d, (3, 4, 6, 3))
        super().__init__(**kwargs)
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=conv1_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            padding_mode="CustomPadding",
        )


@MODELS.register_module()
class CustomBottleneck3d(Bottleneck3d):
    def __init__(self, inplanes: int, planes: int, **kwargs) -> None:
        super().__init__(inplanes, planes, **kwargs)
        if self.inflate:
            if self.inflate_style == "3x1x1":
                self.conv1 = ConvModule(
                    self.inplanes,
                    self.planes,
                    kernel_size=(2, 1, 1),
                    stride=(
                        self.conv1_stride_t,
                        self.conv1_stride_s,
                        self.conv1_stride_s,
                    ),
                    padding=(0, 0, 0, 0, 1, 0),
                    bias=False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    padding_mode="CustomPadding",
                )


@MODELS.register_module()
class CustomPadding(nn.Module):
    def __init__(self, padding: Tuple[int, int, int]):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, self.padding)

    def extra_repr(self) -> str:
        return f"padding={self.padding}"
