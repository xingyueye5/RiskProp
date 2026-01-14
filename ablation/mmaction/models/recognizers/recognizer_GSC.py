from typing import Optional, Union

import numpy as np
import torch
from mmaction.registry import MODELS
from .base import BaseRecognizer


@MODELS.register_module()
class DRIVERecognizer(BaseRecognizer):
    def __init__(self,
                 backbone,
                 data_preprocessor=None,
                 args=None,
                 ):
        super().__init__(backbone=backbone, data_preprocessor=data_preprocessor)

        # 构建你的多模态模型
        self.model = None
        self.trainer = None

    def forward(self,
                inputs: torch.Tensor,       # [B, T, H, W, C]
                data_samples: Optional[list] = None,
                mode: str = 'tensor',
                **kwargs):



    def extract_feat(self, inputs: torch.Tensor, **kwargs):
        return inputs

    def _format_predictions(self, outputs, data_samples):
        """ 将输出转换为 MMAction2 的 ActionDataSample 格式 """
        data_samples[0].set_pred_score(outputs.squeeze())

        # 构造ActionDataSample
        return data_samples
