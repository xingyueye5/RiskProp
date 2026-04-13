# Copyright (c) OpenMMLab. All rights reserved.
import os
import copy
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Sequence, Tuple
from collections import OrderedDict
from mmengine.evaluator import BaseMetric
from mmengine import dump

from mmaction.registry import METRICS

from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve
from .utils import (
    visualize_pred_score,
    visualize_pred_line,
    plot_roc_curves_from_csvs,
    plot_tta_curves_from_csvs,
    plot_mean_accident_aligned_curve_from_results,
)


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

        vis_results=[]
        for result in results:
            # Try to find the threshold on the validation set; if it is unavailable, then use the threshold from the training set.
            threshold_key = f"threshold@{self.fpr_max:.2f}"
            if threshold_key not in eval_results:
                threshold_key = f"threshold#{self.fpr_max:.2f}"
            result["threshold"] = eval_results.get(threshold_key, 0.5)
            if result["target"] is True and result["is_val"] is True:
                continue
            if result["video_id"] in self.vis_list:
                vis_results.append(result)

        
        # Generate the accident-aligned average risk curve based on the validation set.
        # try:
        #     plot_mean_accident_aligned_curve_from_results(
        #         results=results,
        #         output_dir="visualizations",
        #         epoch=self.epoch,
        #         max_pre_frames=50,
        #         include_accident_frame=True,
        #         only_non_test=True,
        #         auto_max_pre_frames=True,
        #     )
        # except Exception as e:
        #     print(f"Dataset-level plotting failed: {e}")

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
        eval_results[f"threshold{sep}{self.fpr_max:.2f}"] = preds_n[int(len(preds_n) * self.fpr_max)]


        ttas = [
            (j - i - np.argmax(pred[i : j + 1, None] >= preds_n, axis=0)) / 10 * np.any(pred[i : j + 1, None] >= preds_n, axis=0)
            for pred, label, i, j in zip(preds, labels, abnormal_start_inds, accident_inds)
            if label
        ]
        ttas = np.array(ttas).mean(axis=0)
        eval_results[f"tta{sep}0.01"] = ttas[int(len(ttas) * 0.01)]
        eval_results[f"tta{sep}0.05"] = ttas[int(len(ttas) * 0.05)]
        eval_results[f"tta{sep}0.1"] = ttas[int(len(ttas) * self.fpr_max)]
        eval_results[f"mtta{sep}0.1"] = ttas[: int(len(ttas) * self.fpr_max)].mean()
        eval_results[f"tta{sep}1"] = ttas[int(len(ttas) * 0.99)]

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
        eval_results[f"mAUC{sep}0.1"] = (
            eval_results[f"AUC{sep}0.5s"] + eval_results[f"AUC{sep}1.0s"] + eval_results[f"AUC{sep}1.5s"]
        ) / 3

        eval_results[f"AUC_full{sep}0.0s"], fpr_0, tpr_0 = auc(np.array(y), np.array(preds_0 + preds_n), 1)
        eval_results[f"AUC_full{sep}0.5s"], fpr_5, tpr_5 = auc(np.array(y), np.array(preds_5 + preds_n), 1)
        eval_results[f"AUC_full{sep}1.0s"], fpr_10, tpr_10 = auc(np.array(y), np.array(preds_10 + preds_n), 1)
        eval_results[f"AUC_full{sep}1.5s"], fpr_15, tpr_15 = auc(np.array(y), np.array(preds_15 + preds_n), 1)
        eval_results[f"mAUC{sep}"] = (
            eval_results[f"AUC_full{sep}0.5s"] + eval_results[f"AUC_full{sep}1.0s"] + eval_results[f"AUC_full{sep}1.5s"]
        ) / 3

        eval_results[f"AP{sep}0.0s"] = average_precision_score(np.array(y), np.array(preds_0 + preds_n))
        eval_results[f"AP{sep}0.5s"] = average_precision_score(np.array(y), np.array(preds_5 + preds_n))
        eval_results[f"AP{sep}1.0s"] = average_precision_score(np.array(y), np.array(preds_10 + preds_n))
        eval_results[f"AP{sep}1.5s"] = average_precision_score(np.array(y), np.array(preds_15 + preds_n))
        eval_results[f"mAP{sep}"] = (
            eval_results[f"AP{sep}0.5s"] + eval_results[f"AP{sep}1.0s"] + eval_results[f"AP{sep}1.5s"]
        ) / 3

        eval_results[f"num_samples{sep}"] = len(preds)
        return eval_results


def auc(y_true, y_scores, fpr_max=0.1):
    score = roc_auc_score(y_true, y_scores, max_fpr=fpr_max)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return score, fpr, tpr
