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
        self.a_fpr_b_0_train = []
        self.a_rec_b_0_train = []
        self.a_tta_b_0_train = []
        self.a_fpr_b_0_val = []
        self.a_rec_b_0_val = []
        self.a_tta_b_0_val = []
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
            self.a_tta_train.append(metrics["a_tta#0.5"])
            plt.plot(self.epochs, self.a_fpr_train, label="a_fpr#0.5 (train)", marker="o", color="blue")
            plt.plot(self.epochs, self.a_rec_train, label="a_rec#0.5 (train)", marker="o", color="red")
            plt.plot(self.epochs, self.a_tta_train, label="a_tta#0.5 (train)", marker="o", color="green")
        if "a_fpr@0.5" in metrics and "a_rec@0.5" in metrics and "a_tta@0.5" in metrics:
            self.a_fpr_val.append(metrics["a_fpr@0.5"])
            self.a_rec_val.append(metrics["a_rec@0.5"])
            self.a_tta_val.append(metrics["a_tta@0.5"])
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
            self.a_fpr_b_0_train.append(metrics["a_fpr#b_0"])
            self.a_rec_b_0_train.append(metrics["a_rec#b_0"])
            self.a_tta_b_0_train.append(metrics["a_tta#b_0"])
            plt.plot(self.epochs, self.a_fpr_b_0_train, label="a_fpr#b_0 (train)", marker="o", color="blue")
            plt.plot(self.epochs, self.a_rec_b_0_train, label="a_rec#b_0 (train)", marker="o", color="red")
            plt.plot(self.epochs, self.a_tta_b_0_train, label="a_tta#b_0 (train)", marker="o", color="green")
        if "a_fpr@b_0" in metrics and "a_rec@b_0" in metrics and "a_tta@b_0" in metrics:
            self.a_fpr_b_0_val.append(metrics["a_fpr@b_0"])
            self.a_rec_b_0_val.append(metrics["a_rec@b_0"])
            self.a_tta_b_0_val.append(metrics["a_tta@b_0"])
            plt.plot(self.epochs, self.a_fpr_b_0_val, label="a_fpr@b_0 (val)", marker="o", color="purple")
            plt.plot(self.epochs, self.a_rec_b_0_val, label="a_rec@b_0 (val)", marker="o", color="yellow")
            plt.plot(self.epochs, self.a_tta_b_0_val, label="a_tta@b_0 (val)", marker="o", color="orange")
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


@HOOKS.register_module()
class NewAnticipationMetricHook(Hook):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.fpr_train = []
        self.tta_train = []
        self.AP5_train = []
        self.AP10_train = []
        self.AP15_train = []
        self.AP20_train = []
        self.mAP_train = []
        self.fpr_val = []
        self.tta_val = []
        self.AP5_val = []
        self.AP10_val = []
        self.AP15_val = []
        self.AP20_val = []
        self.mAP_val = []

    def after_val_epoch(self, runner, metrics) -> None:
        self.epochs.append(runner.epoch)
        plt.figure()
        if "\nfpr#0.5" in metrics and "tta#0.5" in metrics:
            self.fpr_train.append(metrics["\nfpr#0.5"])
            self.tta_train.append(metrics["tta#0.5"])
            plt.plot(self.epochs, self.fpr_train, label="fpr#0.5 (train)", marker="+", color="red")
            plt.plot(self.epochs, self.tta_train, label="tta#0.5 (train)", marker="+", color="blue")
        if "\nfpr@0.5" in metrics and "tta@0.5" in metrics:
            self.fpr_val.append(metrics["\nfpr@0.5"])
            self.tta_val.append(metrics["tta@0.5"])
            plt.plot(self.epochs, self.fpr_val, label="fpr@0.5 (val)", marker="o", color="red")
            plt.plot(self.epochs, self.tta_val, label="tta@0.5 (val)", marker="o", color="blue")
        plt.title("Anticipation Metrics")
        plt.xlabel("Epochs")
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics_anticipation.png"))
        plt.close()

        plt.figure()
        if "mAP#" in metrics:
            self.AP5_train.append(metrics["AP#0.5s"])
            self.AP10_train.append(metrics["AP#1.0s"])
            self.AP15_train.append(metrics["AP#1.5s"])
            self.AP20_train.append(metrics["AP#2.0s"])
            self.mAP_train.append(metrics["mAP#"])
            plt.plot(self.epochs, self.AP5_train, label="AP#0.5s (train)", marker="+", color="blue")
            plt.plot(self.epochs, self.AP10_train, label="AP#1.0s (train)", marker="+", color="red")
            plt.plot(self.epochs, self.AP15_train, label="AP#1.5s (train)", marker="+", color="green")
            plt.plot(self.epochs, self.AP20_train, label="AP#2.0s (train)", marker="+", color="purple")
            plt.plot(self.epochs, self.mAP_train, label="mAP# (train)", marker="+", color="orange")
        if "mAP@" in metrics:
            self.AP5_val.append(metrics["AP@0.5s"])
            self.AP10_val.append(metrics["AP@1.0s"])
            self.AP15_val.append(metrics["AP@1.5s"])
            self.AP20_val.append(metrics["AP@2.0s"])
            self.mAP_val.append(metrics["mAP@"])
            plt.plot(self.epochs, self.AP5_val, label="AP@0.5s (val)", marker="o", color="blue")
            plt.plot(self.epochs, self.AP10_val, label="AP@1.0s (val)", marker="o", color="red")
            plt.plot(self.epochs, self.AP15_val, label="AP@1.5s (val)", marker="o", color="green")
            plt.plot(self.epochs, self.AP20_val, label="AP@2.0s (val)", marker="o", color="purple")
            plt.plot(self.epochs, self.mAP_val, label="mAP@ (val)", marker="o", color="orange")
        i_v = self.mAP_val.index(max(self.mAP_val))
        plt.title(f"mAP_val@{i_v+1}={self.mAP_val[i_v]:.4f}")
        plt.xlabel("Epochs")
        plt.legend()
        plt.xlim(0, max(self.epochs) + 1)
        plt.ylim(-0.1, 1.1)
        plt.xticks(range(1, max(self.epochs) + 1, 1))
        plt.yticks([i * 0.1 for i in range(0, 11)])
        plt.savefig(osp.join(runner.log_dir, "metrics_AP.png"))
        plt.close()
