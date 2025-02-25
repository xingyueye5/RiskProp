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
class DetectionMetricHook(Hook):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.f_pre_train = []
        self.f_rec_train = []
        self.f_pre_val = []
        self.f_rec_val = []
        self.v_fpc_train = []
        self.v_fpr_train = []
        self.v_rec_train = []
        self.v_fpc_val = []
        self.v_fpr_val = []
        self.v_rec_val = []

    def after_val_epoch(self, runner, metrics) -> None:
        self.epochs.append(runner.epoch)
        plt.figure()
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
        plt.savefig(osp.join(runner.log_dir, "metrics_f.png"))
        plt.close()

        plt.figure()
        if "\nv_fpc#0.5" in metrics and "v_fpr#0.5" in metrics and "v_rec#0.5" in metrics:
            self.v_fpc_train.append(metrics["\nv_fpc#0.5"])
            self.v_fpr_train.append(metrics["v_fpr#0.5"])
            self.v_rec_train.append(metrics["v_rec#0.5"])
            plt.plot(self.epochs, self.v_fpc_train, label="v_fpc#0.5 (train)", marker="o", color="blue")
            plt.plot(self.epochs, self.v_fpr_train, label="v_fpr#0.5 (train)", marker="o", color="red")
            plt.plot(self.epochs, self.v_rec_train, label="v_rec#0.5 (train)", marker="o", color="green")
        if "\nv_fpc@0.5" in metrics and "v_fpr@0.5" in metrics and "v_rec@0.5" in metrics:
            self.v_fpc_val.append(metrics["\nv_fpc@0.5"])
            self.v_fpr_val.append(metrics["v_fpr@0.5"])
            self.v_rec_val.append(metrics["v_rec@0.5"])
            plt.plot(self.epochs, self.v_fpc_val, label="v_fpc@0.5 (val)", marker="o", color="purple")
            plt.plot(self.epochs, self.v_fpr_val, label="v_fpr@0.5 (val)", marker="o", color="yellow")
            plt.plot(self.epochs, self.v_rec_val, label="v_rec@0.5 (val)", marker="o", color="orange")
        plt.title("Video level Precision and Recall")
        plt.xlabel("Epochs")
        plt.legend()
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics_v.png"))
        plt.close()


@HOOKS.register_module()
class AnticipationMetricHook(Hook):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.a_fpr_train = []
        self.a_rec_train = []
        self.a_tta_train = []
        self.a_fpr_val = []
        self.a_rec_val = []
        self.a_tta_val = []
        self.d_fpr_train = []
        self.d_rec_1_train = []
        self.d_rec_5_train = []
        self.d_fpr_val = []
        self.d_rec_1_val = []
        self.d_rec_5_val = []

    def after_val_epoch(self, runner, metrics) -> None:
        self.epochs.append(runner.epoch)
        plt.figure()
        if "a_fpr#0.5" in metrics and "a_rec#0.5" in metrics and "a_tta#0.5" in metrics:
            self.a_fpr_train.append(metrics["a_fpr#0.5"])
            self.a_rec_train.append(metrics["a_rec#0.5"])
            self.a_tta_train.append(metrics["a_tta#0.5"] / 10)
            plt.plot(self.epochs, self.a_fpr_train, label="a_fpr#0.5 (train)", marker="o", color="blue")
            plt.plot(self.epochs, self.a_rec_train, label="a_rec#0.5 (train)", marker="o", color="red")
            plt.plot(self.epochs, self.a_tta_train, label="a_tta#0.5 (train)", marker="o", color="green")
        if "a_fpr@0.5" in metrics and "a_rec@0.5" in metrics and "a_tta@0.5" in metrics:
            self.a_fpr_val.append(metrics["a_fpr@0.5"])
            self.a_rec_val.append(metrics["a_rec@0.5"])
            self.a_tta_val.append(metrics["a_tta@0.5"] / 10)
            plt.plot(self.epochs, self.a_fpr_val, label="a_fpr@0.5 (val)", marker="o", color="purple")
            plt.plot(self.epochs, self.a_rec_val, label="a_rec@0.5 (val)", marker="o", color="yellow")
            plt.plot(self.epochs, self.a_tta_val, label="a_tta@0.5 (val)", marker="o", color="orange")
        plt.title("Anticipation Metrics")
        plt.xlabel("Epochs")
        plt.legend()
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics_anticipation.png"))
        plt.close()

        plt.figure()
        if "a_fpr#b_0" in metrics and "a_rec#b_0" in metrics and "a_tta#b_0" in metrics:
            self.a_fpr_train.append(metrics["a_fpr#b_0"])
            self.a_rec_train.append(metrics["a_rec#b_0"])
            self.a_tta_train.append(metrics["a_tta#b_0"] / 10)
            plt.plot(self.epochs, self.a_fpr_train, label="a_fpr#b_0 (train)", marker="o", color="blue")
            plt.plot(self.epochs, self.a_rec_train, label="a_rec#b_0 (train)", marker="o", color="red")
            plt.plot(self.epochs, self.a_tta_train, label="a_tta#b_0 (train)", marker="o", color="green")
        if "a_fpr@b_0" in metrics and "a_rec@b_0" in metrics and "a_tta@b_0" in metrics:
            self.a_fpr_val.append(metrics["a_fpr@b_0"])
            self.a_rec_val.append(metrics["a_rec@b_0"])
            self.a_tta_val.append(metrics["a_tta@b_0"] / 10)
            plt.plot(self.epochs, self.a_fpr_val, label="a_fpr@b_0 (val)", marker="o", color="purple")
            plt.plot(self.epochs, self.a_rec_val, label="a_rec@b_0 (val)", marker="o", color="yellow")
            plt.plot(self.epochs, self.a_tta_val, label="a_tta@b_0 (val)", marker="o", color="orange")
        plt.title("Anticipation Metrics")
        plt.xlabel("Epochs")
        plt.legend()
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics_anticipation_b_0.png"))
        plt.close()

        plt.figure()
        if "\nd_fpr#0.5" in metrics and "d_rec_1#0.5" in metrics and "d_rec_5#0.5" in metrics:
            self.d_fpr_train.append(metrics["\nd_fpr#0.5"])
            self.d_rec_1_train.append(metrics["d_rec_1#0.5"])
            self.d_rec_5_train.append(metrics["d_rec_5#0.5"])
            plt.plot(self.epochs, self.d_fpr_train, label="d_fpr#0.5 (train)", marker="o", color="blue")
            plt.plot(self.epochs, self.d_rec_1_train, label="d_rec_1#0.5 (train)", marker="o", color="red")
            plt.plot(self.epochs, self.d_rec_5_train, label="d_rec_5#0.5 (train)", marker="o", color="green")
        if "\nd_fpr@0.5" in metrics and "d_rec_1@0.5" in metrics and "d_rec_5@0.5" in metrics:
            self.d_fpr_val.append(metrics["\nd_fpr@0.5"])
            self.d_rec_1_val.append(metrics["d_rec_1@0.5"])
            self.d_rec_5_val.append(metrics["d_rec_5@0.5"])
            plt.plot(self.epochs, self.d_fpr_val, label="d_fpr@0.5 (val)", marker="o", color="purple")
            plt.plot(self.epochs, self.d_rec_1_val, label="d_rec_1@0.5 (val)", marker="o", color="yellow")
            plt.plot(self.epochs, self.d_rec_5_val, label="d_rec_5@0.5 (val)", marker="o", color="orange")
        plt.title("Detection Metrics")
        plt.xlabel("Epochs")
        plt.legend()
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics_detection.png"))
        plt.close()
