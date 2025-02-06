# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import matplotlib.pyplot as plt
import os.path as osp


@HOOKS.register_module()
class EpochHook(Hook):
    def before_val_epoch(self, runner) -> None:
        for metric in runner.val_evaluator.metrics:
            metric.epoch = runner.epoch


@HOOKS.register_module()
class MetricHook(Hook):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.f_pre_train = []
        self.f_rec_train = []
        self.f_pre_val = []
        self.f_rec_val = []

    def after_val_epoch(self, runner, metrics) -> None:
        self.epochs.append(runner.epoch)
        if "f_pre#0.5" in metrics and "f_rec#0.5" in metrics:
            self.f_pre_train.append(metrics["f_pre#0.5"])
            self.f_rec_train.append(metrics["f_rec#0.5"])
            plt.plot(self.epochs, self.f_pre_train, label="f_pre#0.5 (train)", marker="o", color="blue")
            plt.plot(self.epochs, self.f_rec_train, label="f_rec#0.5 (train)", marker="o", color="green")
        if "f_pre@0.5" in metrics and "f_rec@0.5" in metrics:
            self.f_pre_val.append(metrics["f_pre@0.5"])
            self.f_rec_val.append(metrics["f_rec@0.5"])
            plt.plot(self.epochs, self.f_pre_val, label="f_pre@0.5 (val)", marker="o", color="red")
            plt.plot(self.epochs, self.f_rec_val, label="f_rec@0.5 (val)", marker="o", color="purple")
        plt.title("Frame level Precision and Recall")
        plt.xlabel("Epochs")
        plt.legend()
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics.png"))
        plt.close()
