# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class EpochHook(Hook):
    def before_val_epoch(self, runner) -> None:
        for metric in runner.val_evaluator.metrics:
            metric.epoch = runner.epoch
