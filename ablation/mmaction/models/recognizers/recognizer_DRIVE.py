import time
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
        # self.model = MODELS.build(model_cfg)  # todo 是否有连续调用argparse的问题?
        self.env = None
        self.agent = None
        self.cfg = None

    def forward(self,
                inputs: torch.Tensor,       # [B, T, H, W, C]
                data_samples: Optional[list] = None,
                mode: str = 'tensor',
                **kwargs):

        all_pred_scores, all_gt_labels, all_pred_fixations, all_gt_fixations, all_toas, all_vids = [], [], [], [], [], []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()


        state = self.env.set_data(inputs)
        rnn_state = (torch.zeros((self.cfg.ENV.batch_size, self.cfg.SAC.hidden_size), dtype=torch.float32).to(self.cfg.device),
                     torch.zeros((self.cfg.ENV.batch_size, self.cfg.SAC.hidden_size), dtype=torch.float32).to(self.cfg.device))
        score_pred = torch.zeros((self.cfg.ENV.batch_size, self.env.max_steps), dtype=torch.float32)

        i_steps = 0
        while i_steps < self.env.max_steps:
            actions, rnn_state = self.agent.select_action(state, rnn_state, evaluate=True)
            # step
            state, reward, info = self.env.step(actions, isTraining=False)
            # gather actions
            score_pred[:, i_steps] = info['pred_score']  # shape=(B,)
            # next step
            i_steps += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        delta_time = end_time - start_time
        size = inputs.shape[1]  # 获取批量大小
        fps = size / delta_time  # 计算每秒处理的帧数

        print(str(fps) + '\n')

        if mode == 'predict':
            return self._format_predictions(score_pred, data_samples)
        else:
            return score_pred

    def extract_feat(self, inputs: torch.Tensor, **kwargs):
        return inputs

    def _format_predictions(self, outputs, data_samples):
        """ 将输出转换为 MMAction2 的 ActionDataSample 格式 """
        data_samples[0].set_pred_score(outputs.squeeze())

        # 构造ActionDataSample
        return data_samples
