import time
from typing import Optional, Union

import numpy as np
import torch
from mmaction.registry import MODELS
from .base import BaseRecognizer
from CAP.src.model import accident
from mmengine.structures import LabelData
from mmengine.runner import load_checkpoint

from ...structures import ActionDataSample


# 自定义虚拟 Backbone 类
@MODELS.register_module()
class IdentityBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x  # 直接返回输入，不做任何处理


@MODELS.register_module()
class CAPRecognizer(BaseRecognizer):
    def __init__(self,
                 backbone,
                 data_preprocessor=None,
                 ):
        super().__init__(backbone=backbone, data_preprocessor=data_preprocessor)

        # 构建你的多模态模型
        # self.model = MODELS.build(model_cfg)
        self.model = accident(256,2,4,512,8,49,512,120,512,[3,4],2).cuda()
        self.model.eval()


    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor',
                **kwargs):
        # 数据预处理
        # N,C,H,W = inputs.shape
        # inputs = inputs.reshape(1, N, C, H, W)
        # focus = inputs['focus']      不再需要focus输入, 因为CapData只有RGB
        text = ["a video frame of { }"]  # ToDo: 修改固定text文本

        # ToDo 获取标签和时间信息
        labels = []
        toa = []
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # start_time = time.time()

        # 模型前向
        outputs = self.model(
            x=inputs, z=None, y=None, toa=None, w=text
        )

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # end_time = time.time()
        #
        # delta_time = end_time - start_time
        # size = inputs.shape[1]  # 获取批量大小
        # fps = size / delta_time  # 计算每秒处理的帧数
        #
        # print(str(fps) + '\n')

        if mode == 'predict':
            return self._format_predictions(outputs[1], data_samples)
        else:
            return outputs

    def extract_feat(self, inputs: torch.Tensor, **kwargs):
        return inputs

    def _format_predictions(self, outputs, data_samples):
        """ 将输出转换为 MMAction2 的 ActionDataSample 格式 """
        predictions = []
        num_frames = len(outputs)

        pred_frames = np.zeros((1, num_frames), dtype=np.float32)
        for t in range(len(outputs)):
            pred = outputs[t]
            # print( pred)
            pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
            pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)

            # 此时, 给出一个score加入data_sample
        pred_frames = pred_frames.squeeze(axis=0)
        pred_frames = torch.from_numpy(pred_frames).to(outputs[0].device)

        data_samples[0].set_pred_score(pred_frames)

        # 构造ActionDataSample
        return data_samples
