# Copyright (c) OpenMMLab. All rights reserved.
import os
import copy
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Sequence, Tuple
from collections import OrderedDict
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS

from sklearn.metrics import average_precision_score
from .utils import visualize_pred_score


@METRICS.register_module()
class AnticipationMetric(BaseMetric):
    def __init__(self, fpr_max: float = 0.1, vis_list=[], output_dir=None) -> None:
        super().__init__()

        self.fpr_max = fpr_max
        self.vis_list = vis_list
        self.output_dir = output_dir
        self.epoch = None

    def process(self, data_batch: Sequence[Tuple[Any, Dict]], data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            result["pred"] = data_sample["pred_score"].cpu().numpy()
            result["target"] = data_sample["target"]
            if data_sample["target"] is True:
                abnormal_start_ind = np.where(data_sample["frame_inds"] >= data_sample["abnormal_start_frame"])[0]
                result["abnormal_start_ind"] = abnormal_start_ind[0] if abnormal_start_ind.size > 0 else 0
                accident_ind = np.where(data_sample["frame_inds"] >= data_sample["accident_frame"])[0]
                result["accident_ind"] = accident_ind[0] if accident_ind.size > 0 else 0
            else:
                result["abnormal_start_ind"] = 0
                result["accident_ind"] = 0
            result["video_id"] = data_sample["video_id"]
            result["dataset"] = data_sample["dataset"]
            result["frame_dir"] = data_sample["frame_dir"]
            result["filename_tmpl"] = data_sample["filename_tmpl"]
            result["type"] = data_sample["type"]
            result["frame_inds"] = data_sample["frame_inds"]
            result["abnormal_start_frame"] = data_sample["abnormal_start_frame"]
            result["accident_frame"] = data_sample["accident_frame"]
            result["is_val"] = data_sample["is_val"]
            result["is_test"] = data_sample["is_test"]
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        os.makedirs("outputs", exist_ok=True)

        preds = [x["pred"] for x in results if not x["is_val"] and not x["is_test"]]
        targets = [x["target"] for x in results if not x["is_val"] and not x["is_test"]]
        abnormal_start_inds = [x["abnormal_start_ind"] for x in results if not x["is_val"] and not x["is_test"]]
        accident_inds = [x["accident_ind"] for x in results if not x["is_val"] and not x["is_test"]]
        if len(preds) > 0:
            eval_results.update(self.calculate(preds, targets, abnormal_start_inds, accident_inds, "#"))

        preds = [x["pred"] for x in results if x["is_val"]]
        targets = [x["target"] for x in results if x["is_val"]]
        abnormal_start_inds = [x["abnormal_start_ind"] for x in results if x["is_val"]]
        accident_inds = [x["accident_ind"] for x in results if x["is_val"]]
        if len(preds) > 0:
            eval_results.update(self.calculate(preds, targets, abnormal_start_inds, accident_inds, "@"))

        data = dict(
            id=[x["video_id"] for x in results if x["is_test"]],
            target=[f'{np.max(x["pred"][-1]):.4f}' for x in results if x["is_test"]],
        )
        df = pd.DataFrame(data)
        if self.epoch is not None:
            df.to_csv(f"outputs/sample_submission_{self.epoch}.csv", index=False)
        else:
            df.to_csv("outputs/sample_submission.csv", index=False)

        for result in results:
            result["threshold"] = eval_results.get(f"threshold@{self.fpr_max:.2f}", 0.5)
            if result["video_id"] in self.vis_list:
                if result["target"] is True:
                    visualize_pred_score(result, self.output_dir, self.epoch)

        return eval_results

    def calculate(self, preds, labels, abnormal_start_inds, accident_inds, sep) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()

        if len(preds[0].shape) > 1:
            preds = [np.max(pred, axis=-1) for pred in preds]

        preds_n = -np.sort(-np.concatenate([pred for pred, label in zip(preds, labels) if not label]))
        eval_results[f"\nfpr{sep}0.5"] = sum(preds_n >= 0.5) / len(preds_n)
        eval_results[f"threshold{sep}{self.fpr_max:.2f}"] = preds_n[int(len(preds_n) * self.fpr_max)]

        ttas = [
            (i - np.argmax(pred[: i + 1] >= 0.5)) / 10 if np.any(pred[: i + 1] >= 0.5) else 0
            for pred, label, i in zip(preds, labels, accident_inds)
            if label
        ]
        eval_results[f"tta_old{sep}0.5"] = sum(ttas) / len(ttas)

        ttas = [
            (i - np.argmax(pred[: i + 1, None] >= preds_n, axis=0)) / 10 * np.any(pred[: i + 1, None] >= preds_n, axis=0)
            for pred, label, i in zip(preds, labels, accident_inds)
            if label
        ]
        ttas = np.array(ttas).mean(axis=0)
        eval_results[f"mtta_old{sep}0.1"] = ttas[: int(len(ttas) * self.fpr_max)].mean()

        df = pd.DataFrame(dict(fpr=[(i + 1) / len(ttas) for i in range(len(ttas))], tta=ttas))
        df.to_csv(f"outputs/fpr_tta_old_{self.epoch}.csv", index=False)

        ttas = [
            (20 - np.argmax(pred[i - 20 : i + 1] >= 0.5)) / 10 if np.any(pred[i - 20 : i + 1] >= 0.5) else 0
            for pred, label, i in zip(preds, labels, accident_inds)
            if label
        ]
        eval_results[f"tta_2s{sep}0.5"] = sum(ttas) / len(ttas)

        ttas = [
            (20 - np.argmax(pred[i - 20 : i + 1, None] >= preds_n, axis=0)) / 10 * np.any(pred[i - 20 : i + 1, None] >= preds_n, axis=0)
            for pred, label, i in zip(preds, labels, accident_inds)
            if label
        ]
        ttas = np.array(ttas).mean(axis=0)
        eval_results[f"mtta_2s{sep}0.1"] = ttas[: int(len(ttas) * self.fpr_max)].mean()

        df = pd.DataFrame(dict(fpr=[(i + 1) / len(ttas) for i in range(len(ttas))], tta=ttas))
        df.to_csv(f"outputs/fpr_tta_2s_{self.epoch}.csv", index=False)

        ttas = [
            (j - i - np.argmax(pred[i : j + 1] >= 0.5)) / 10 if np.any(pred[i : j + 1] >= 0.5) else 0
            for pred, label, i, j in zip(preds, labels, abnormal_start_inds, accident_inds)
            if label
        ]
        eval_results[f"tta{sep}0.5"] = sum(ttas) / len(ttas)

        ttas = [
            (j - i - np.argmax(pred[i : j + 1, None] >= preds_n, axis=0)) / 10 * np.any(pred[i : j + 1, None] >= preds_n, axis=0)
            for pred, label, i, j in zip(preds, labels, abnormal_start_inds, accident_inds)
            if label
        ]
        ttas = np.array(ttas).mean(axis=0)
        eval_results[f"mtta{sep}0.1"] = ttas[: int(len(ttas) * self.fpr_max)].mean()

        df = pd.DataFrame(dict(fpr=[(i + 1) / len(ttas) for i in range(len(ttas))], tta=ttas))
        df.to_csv(f"outputs/fpr_tta_{self.epoch}.csv", index=False)

        preds_0 = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if label]
        preds_5 = [np.max(pred[-10:-5]) for pred, label in zip(preds, labels) if label]
        preds_10 = [np.max(pred[-15:-10]) for pred, label in zip(preds, labels) if label]
        preds_15 = [np.max(pred[-20:-15]) for pred, label in zip(preds, labels) if label]
        preds_n = [np.max(pred[-5:]) for pred, label in zip(preds, labels) if not label]
        y = [1] * len(preds_0) + [0] * len(preds_n)
        eval_results[f"AUC{sep}0.0s"], fpr_0, tpr_0 = auc(np.array(y), np.array(preds_0 + preds_n), self.fpr_max)
        eval_results[f"AUC{sep}0.5s"], fpr_5, tpr_5 = auc(np.array(y), np.array(preds_5 + preds_n), self.fpr_max)
        eval_results[f"AUC{sep}1.0s"], fpr_10, tpr_10 = auc(np.array(y), np.array(preds_10 + preds_n), self.fpr_max)
        eval_results[f"AUC{sep}1.5s"], fpr_15, tpr_15 = auc(np.array(y), np.array(preds_15 + preds_n), self.fpr_max)
        eval_results[f"mAUC{sep}"] = (
            eval_results[f"AUC{sep}0.5s"] + eval_results[f"AUC{sep}1.0s"] + eval_results[f"AUC{sep}1.5s"]
        ) / 3

        eval_results[f"AUC_full{sep}0.0s"], fpr_0, tpr_0 = auc(np.array(y), np.array(preds_0 + preds_n), 1)
        eval_results[f"AUC_full{sep}0.5s"], fpr_5, tpr_5 = auc(np.array(y), np.array(preds_5 + preds_n), 1)
        eval_results[f"AUC_full{sep}1.0s"], fpr_10, tpr_10 = auc(np.array(y), np.array(preds_10 + preds_n), 1)
        eval_results[f"AUC_full{sep}1.5s"], fpr_15, tpr_15 = auc(np.array(y), np.array(preds_15 + preds_n), 1)
        eval_results[f"mAUC_full{sep}"] = (
            eval_results[f"AUC_full{sep}0.5s"] + eval_results[f"AUC_full{sep}1.0s"] + eval_results[f"AUC_full{sep}1.5s"]
        ) / 3

        eval_results[f"AP_full{sep}0.0s"] = average_precision_score(np.array(y), np.array(preds_0 + preds_n))
        eval_results[f"AP_full{sep}0.5s"] = average_precision_score(np.array(y), np.array(preds_5 + preds_n))
        eval_results[f"AP_full{sep}1.0s"] = average_precision_score(np.array(y), np.array(preds_10 + preds_n))
        eval_results[f"AP_full{sep}1.5s"] = average_precision_score(np.array(y), np.array(preds_15 + preds_n))
        eval_results[f"mAP_full{sep}"] = (
            eval_results[f"AP_full{sep}0.5s"] + eval_results[f"AP_full{sep}1.0s"] + eval_results[f"AP_full{sep}1.5s"]
        ) / 3

        df = pd.DataFrame(dict(fpr=fpr_0, tpr=tpr_0))
        df.to_csv(f"outputs/fpr_tpr_0_{self.epoch}.csv", index=False)
        df = pd.DataFrame(dict(fpr=fpr_5, tpr=tpr_5))
        df.to_csv(f"outputs/fpr_tpr_5_{self.epoch}.csv", index=False)
        df = pd.DataFrame(dict(fpr=fpr_10, tpr=tpr_10))
        df.to_csv(f"outputs/fpr_tpr_10_{self.epoch}.csv", index=False)
        df = pd.DataFrame(dict(fpr=fpr_15, tpr=tpr_15))
        df.to_csv(f"outputs/fpr_tpr_15_{self.epoch}.csv", index=False)

        eval_results[f"num_samples{sep}"] = len(preds)
        return eval_results


def auc(y_true, y_scores, fpr_max=0.1):
    # 1. 按预测得分降序排序，并记录真实标签
    sorted_indices = np.argsort(y_scores)[::-1]  # 从高到低排序
    y_true_sorted = y_true[sorted_indices]

    # 2. 计算正负样本数量
    P = np.sum(y_true == 1)  # 正样本数
    N = np.sum(y_true == 0)  # 负样本数

    # 3. 初始化TPR和FPR
    TPR = [0]  # 真正例率（初始为0）
    FPR = [0]  # 假正例率（初始为0）
    TP, FP = 0, 0  # 累积真正例和假正例数

    # 4. 遍历排序后的样本，动态更新TPR和FPR
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            TP += 1  # 真正例+1
        else:
            FP += 1  # 假正例+1
        TPR.append(TP / P)  # 计算当前TPR
        FPR.append(FP / N)  # 计算当前FPR

    # 5. 梯形法计算AUC（积分ROC曲线下面积）
    auc = 0
    for i in range(1, len(FPR)):
        if FPR[i] > fpr_max:
            break
        dx = FPR[i] - FPR[i - 1]  # x轴宽度
        dy = TPR[i] + TPR[i - 1]  # y轴平均高度
        auc += dx * dy / 2  # 梯形面积累加

    return auc / fpr_max, FPR, TPR
