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
class AnticipationMetricHook(Hook):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.fpr_train = []
        self.tta_train = []
        self.mtta_train = []
        self.AUC0_train = []
        self.AUC5_train = []
        self.AUC10_train = []
        self.AUC15_train = []
        self.mAUC_train = []
        self.fpr_val = []
        self.tta_val = []
        self.mtta_val = []
        self.AUC0_val = []
        self.AUC5_val = []
        self.AUC10_val = []
        self.AUC15_val = []
        self.mAUC_val = []

    def after_val_epoch(self, runner, metrics) -> None:
        self.epochs.append(runner.epoch)
        plt.figure()
        if "\nfpr#0.5" in metrics and "tta#0.5" in metrics and "mtta#0.1" in metrics:
            self.fpr_train.append(metrics["\nfpr#0.5"])
            self.tta_train.append(metrics["tta#0.5"])
            self.mtta_train.append(metrics["mtta#0.1"])
            plt.plot(self.epochs, self.fpr_train, label="fpr#0.5 (train)", marker="+", color="red")
            plt.plot(self.epochs, self.tta_train, label="tta#0.5 (train)", marker="+", color="blue")
            plt.plot(self.epochs, self.mtta_train, label="mtta#0.1 (train)", marker="+", color="green")
        if "\nfpr@0.5" in metrics and "tta@0.5" in metrics and "mtta@0.1" in metrics:
            self.fpr_val.append(metrics["\nfpr@0.5"])
            self.tta_val.append(metrics["tta@0.5"])
            self.mtta_val.append(metrics["mtta@0.1"])
            plt.plot(self.epochs, self.fpr_val, label="fpr@0.5 (val)", marker="o", color="red")
            plt.plot(self.epochs, self.tta_val, label="tta@0.5 (val)", marker="o", color="blue")
            plt.plot(self.epochs, self.mtta_val, label="mtta@0.1 (val)", marker="o", color="green")
        plt.title("Anticipation Metrics")
        plt.xlabel("Epochs")
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics_tta.png"))
        plt.close()

        plt.figure()
        if "mAUC#" in metrics:
            self.AUC0_train.append(metrics["AUC#0.0s"])
            self.AUC5_train.append(metrics["AUC#0.5s"])
            self.AUC10_train.append(metrics["AUC#1.0s"])
            self.AUC15_train.append(metrics["AUC#1.5s"])
            self.mAUC_train.append(metrics["mAUC#"])
            plt.plot(self.epochs, self.AUC0_train, label="AUC#0.0s (train)", marker="+", color="purple")
            plt.plot(self.epochs, self.AUC5_train, label="AUC#0.5s (train)", marker="+", color="blue")
            plt.plot(self.epochs, self.AUC10_train, label="AUC#1.0s (train)", marker="+", color="red")
            plt.plot(self.epochs, self.AUC15_train, label="AUC#1.5s (train)", marker="+", color="green")
            plt.plot(self.epochs, self.mAUC_train, label="mAUC# (train)", marker="+", color="orange")
        if "mAUC@" in metrics:
            self.AUC0_val.append(metrics["AUC@0.0s"])
            self.AUC5_val.append(metrics["AUC@0.5s"])
            self.AUC10_val.append(metrics["AUC@1.0s"])
            self.AUC15_val.append(metrics["AUC@1.5s"])
            self.mAUC_val.append(metrics["mAUC@"])
            plt.plot(self.epochs, self.AUC0_val, label="AUC@0.0s (val)", marker="o", color="purple")
            plt.plot(self.epochs, self.AUC5_val, label="AUC@0.5s (val)", marker="o", color="blue")
            plt.plot(self.epochs, self.AUC10_val, label="AUC@1.0s (val)", marker="o", color="red")
            plt.plot(self.epochs, self.AUC15_val, label="AUC@1.5s (val)", marker="o", color="green")
            plt.plot(self.epochs, self.mAUC_val, label="mAUC@ (val)", marker="o", color="orange")
        i_v = self.mAUC_val.index(max(self.mAUC_val))
        plt.title(f"mAUC_val@{i_v+1}={self.mAUC_val[i_v]:.4f}")
        plt.xlabel("Epochs")
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics_AUC.png"))
        plt.close()
