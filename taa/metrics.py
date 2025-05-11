# Copyright (c) OpenMMLab. All rights reserved.
import os
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pandas as pd
import torch
from mmengine.evaluator import BaseMetric

from mmaction.evaluation import (
    get_weighted_score,
    mean_average_precision,
    mean_class_accuracy,
    mmit_mean_average_precision,
    top_k_accuracy,
)
from mmaction.registry import METRICS

from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score
from .utils import visualize_pred_score


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f"{type(value)} is not an available argument.")
    return value


@METRICS.register_module()
class DetectionMetric(BaseMetric):

    default_prefix: Optional[str] = ""

    def __init__(
        self,
        thresholds=[0.5],
        a_fpr_benchmarks=[0.01],
        vis_list=[],
        output_dir=None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.thresholds = thresholds
        self.a_fpr_benchmarks = a_fpr_benchmarks
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
            result["label"] = data_sample["gt_label"].cpu().numpy()
            abnormal_start_ind = np.where(data_sample["frame_inds"] >= data_sample["abnormal_start_frame"])[0]
            result["abnormal_start_ind"] = abnormal_start_ind[0] if abnormal_start_ind.size > 0 else 0
            accident_ind = np.where(data_sample["frame_inds"] >= data_sample["accident_frame"])[0]
            result["accident_ind"] = accident_ind[0] if accident_ind.size > 0 else 0
            result["frame_dir"] = data_sample["frame_dir"]
            result["filename_tmpl"] = data_sample["filename_tmpl"]
            result["frame_inds"] = data_sample["frame_inds"]
            result["abnormal_start_frame"] = data_sample["abnormal_start_frame"]
            result["abnormal_end_frame"] = data_sample["abnormal_end_frame"]
            result["accident_frame"] = data_sample["accident_frame"]
            result["video_id"] = data_sample["video_id"]
            result["type"] = data_sample["type"]
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
        preds = [x["pred"] for x in results if not x["is_test"]]
        labels = [x["label"] for x in results if not x["is_test"]]
        abnormal_start_inds = [x["abnormal_start_ind"] for x in results if not x["is_test"]]
        accident_inds = [x["accident_ind"] for x in results if not x["is_test"]]
        if len(preds) > 0:
            eval_results.update(self.calculate(preds, labels, abnormal_start_inds, accident_inds, False))

        preds = [x["pred"] for x in results if x["is_test"]]
        labels = [x["label"] for x in results if x["is_test"]]
        abnormal_start_inds = [x["abnormal_start_ind"] for x in results if x["is_test"]]
        accident_inds = [x["accident_ind"] for x in results if x["is_test"]]
        eval_results.update(self.calculate(preds, labels, abnormal_start_inds, accident_inds, True))

        for result in results:
            result["threshold"] = eval_results.get("threshold@b_0", 0.5)
            if result["video_id"] in self.vis_list:
                visualize_pred_score(result, self.output_dir, self.epoch)

        return eval_results

    def calculate(self, preds, labels, abnormal_start_inds, accident_inds, is_test) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        sep = "@" if is_test else "#"
        for t in self.thresholds:
            preds_t = [pred >= t for pred in preds]
            labels_t = [label == 1 for label in labels]
            pred_videos = np.array(
                [np.any(pred[np.argmax(label) - 2 : np.argmax(label) + 3]) for pred, label in zip(preds_t, labels_t)]
            )
            pred_before_accident = np.concatenate(
                [pred[: np.argmax(label) - 2] for pred, label in zip(preds_t, labels_t)]
            )
            false_positive_count = np.array(
                [
                    len(np.where(np.diff(np.concatenate(([0], pred[: np.argmax(label) - 2]))) == 1)[0])
                    for pred, label in zip(preds_t, labels_t)
                ]
            )
            label_videos = np.array([np.any(label) for label in labels_t])
            pred_frames = np.concatenate(preds_t)
            label_frames = np.concatenate(labels_t)
            # Only calculate tta when there is a true positive sample
            ttas = [
                (np.where(label)[0][0] - np.where(pred)[0][0]) / 10
                for pred, label in zip(preds_t, labels_t)
                if np.any(label) and np.any(pred)
            ]

            eval_results[f"\nv_fpc{sep}{t:.1f}"] = false_positive_count.sum() / len(false_positive_count)
            eval_results[f"v_fpr{sep}{t:.1f}"] = pred_before_accident.sum() / len(pred_before_accident)
            eval_results[f"v_rec{sep}{t:.1f}"] = recall_score(label_videos, pred_videos, zero_division=0)
            eval_results[f"f_pre{sep}{t:.1f}"] = precision_score(label_frames, pred_frames, zero_division=0)
            eval_results[f"f_rec{sep}{t:.1f}"] = recall_score(label_frames, pred_frames, zero_division=0)
            eval_results[f"tta{sep}{t:.1f}"] = sum(ttas) / len(ttas) if len(ttas) else 0

        metrics = ["\nv_fpc", "v_fpr", "v_rec", "f_pre", "f_rec"]
        for metric in metrics:
            eval_results[f"{metric}{sep}max"] = max([eval_results[f"{metric}{sep}{t:.1f}"] for t in self.thresholds])
        eval_results[f"tta{sep}min"] = min([eval_results[f"tta{sep}{t:.1f}"] for t in self.thresholds])
        eval_results[f"\n{sep}num_samples"] = len(labels)
        return eval_results


@METRICS.register_module()
class AnticipationMetric(DetectionMetric):
    def calculate(self, preds, labels, abnormal_start_inds, accident_inds, is_test) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        sep = "@" if is_test else "#"

        labels_video = np.array([np.any(label) for label in labels])

        if len(preds[0].shape) == 1:
            a = np.concatenate([pred[:i] for pred, i in zip(preds, abnormal_start_inds)])
        else:
            a = np.concatenate([np.max(pred[:i], axis=1) for pred, i in zip(preds, abnormal_start_inds)])
        thresholds = [np.partition(a, -int(b * len(a)))[-int(b * len(a))] for b in self.a_fpr_benchmarks]

        for index, t in enumerate(self.thresholds + thresholds):
            preds_t = [pred >= t for pred in preds]

            t = f"{t:.1f}" if index < len(self.thresholds) else f"b_{index - len(self.thresholds)}"

            if len(preds[0].shape) == 1:
                alarms_video = np.array(
                    [np.any(pred[i:j]) for pred, i, j in zip(preds_t, abnormal_start_inds, accident_inds)]
                )
                alarms_before_abnormal = np.concatenate([pred[:i] for pred, i in zip(preds_t, abnormal_start_inds)])
                ttas = np.array(
                    [
                        (j - i - np.argmax(pred[i:j])) / 10 if np.any(pred[i:j]) else 0
                        for pred, i, j in zip(preds_t, abnormal_start_inds, accident_inds)
                    ]
                )

                eval_results[f"\nd_fpr{sep}{t}"] = 0
                eval_results[f"d_rec_1{sep}{t}"] = 0
                eval_results[f"d_rec_5{sep}{t}"] = 0
                eval_results[f"a_fpr{sep}{t}"] = alarms_before_abnormal.sum() / len(alarms_before_abnormal)
                eval_results[f"a_rec{sep}{t}"] = recall_score(labels_video, alarms_video, zero_division=0)
                eval_results[f"a_tta{sep}{t}"] = sum(ttas) / len(ttas) if len(ttas) else 0
                continue

            preds_video_1 = np.array([pred[i, 0] for pred, i in zip(preds_t, accident_inds)])
            preds_video_5 = np.array([np.any(pred[i - 2 : i + 3, 0]) for pred, i in zip(preds_t, accident_inds)])
            preds_before_accident = np.concatenate([pred[:i, 0] for pred, i in zip(preds_t, accident_inds)])
            alarms_video = np.array(
                [np.any(pred[i:j]) for pred, i, j in zip(preds_t, abnormal_start_inds, accident_inds)]
            )
            alarms_before_abnormal = np.concatenate(
                [np.any(pred[:i], axis=1) for pred, i in zip(preds_t, abnormal_start_inds)]
            )
            ttas = np.array(
                [
                    (j - i - np.argmax(np.any(pred[i:j], axis=1))) / 10 if np.any(pred[i:j]) else 0
                    for pred, i, j in zip(preds_t, abnormal_start_inds, accident_inds)
                ]
            )

            eval_results[f"\nd_fpr{sep}{t}"] = preds_before_accident.sum() / len(preds_before_accident)
            eval_results[f"d_rec_1{sep}{t}"] = recall_score(labels_video, preds_video_1, zero_division=0)
            eval_results[f"d_rec_5{sep}{t}"] = recall_score(labels_video, preds_video_5, zero_division=0)
            eval_results[f"a_fpr{sep}{t}"] = alarms_before_abnormal.sum() / len(alarms_before_abnormal)
            eval_results[f"a_rec{sep}{t}"] = recall_score(labels_video, alarms_video, zero_division=0)
            eval_results[f"a_tta{sep}{t}"] = sum(ttas) / len(ttas) if len(ttas) else 0

        eval_results[f"\n{sep}num_samples"] = len(labels)
        for index, t in enumerate(thresholds):
            eval_results[f"threshold{sep}b_{index}"] = t
        return eval_results


@METRICS.register_module()
class NewAnticipationMetric(DetectionMetric):
    def calculate(self, preds, labels, abnormal_start_inds, accident_inds, is_test) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        sep = "@" if is_test else "#"

        if len(preds[0].shape) > 1:
            preds = [np.max(pred, axis=-1) for pred in preds]

        preds_before_abnormal = -np.sort(-np.concatenate([pred[:i] for pred, i in zip(preds, abnormal_start_inds)]))
        eval_results[f"\nfpr{sep}0.5"] = sum(preds_before_abnormal >= 0.5) / len(preds_before_abnormal)

        ttas = [
            (i - np.argmax(pred[:i] >= 0.5)) / 10 if np.any(pred[:i] >= 0.5) else 0
            for pred, i in zip(preds, accident_inds)
        ]
        eval_results[f"tta{sep}0.5"] = sum(ttas) / len(ttas)

        tta_list = []
        for t in preds_before_abnormal:
            ttas = [
                (i - np.argmax(pred[:i] >= t)) / 10 if np.any(pred[:i] >= t) else 0
                for pred, i in zip(preds, accident_inds)
            ]
            tta_list.append(sum(ttas) / len(ttas))
        eval_results[f"auc{sep}0.1"] = sum(tta_list[: int(len(tta_list) * 0.1)]) / int(len(tta_list) * 0.1)

        df = pd.DataFrame(dict(fpr=[(i + 1) / len(tta_list) for i in range(len(tta_list))], tta=tta_list))
        os.makedirs("outputs", exist_ok=True)
        df.to_csv(f"outputs/fpr_tta_{self.epoch}.csv", index=False)

        preds_0 = [np.max(pred[i - 5 : i]) for pred, i in zip(preds, accident_inds) if i >= 20]
        preds_5 = [np.max(pred[i - 10 : i - 5]) for pred, i in zip(preds, accident_inds) if i >= 20]
        preds_10 = [np.max(pred[i - 15 : i - 10]) for pred, i in zip(preds, accident_inds) if i >= 20]
        preds_15 = [np.max(pred[i - 20 : i - 15]) for pred, i in zip(preds, accident_inds) if i >= 20]
        preds_n = [np.max(pred[:5]) for pred, i in zip(preds, accident_inds) if i >= 20]
        labels = [1] * len(preds_n) + [0] * len(preds_n)
        eval_results[f"AP{sep}0.0s"] = average_precision_score(labels, preds_0 + preds_n)
        eval_results[f"AP{sep}0.5s"] = average_precision_score(labels, preds_5 + preds_n)
        eval_results[f"AP{sep}1.0s"] = average_precision_score(labels, preds_10 + preds_n)
        eval_results[f"AP{sep}1.5s"] = average_precision_score(labels, preds_15 + preds_n)
        eval_results[f"mAP{sep}"] = (
            eval_results[f"AP{sep}0.0s"]
            + eval_results[f"AP{sep}0.5s"]
            + eval_results[f"AP{sep}1.0s"]
            + eval_results[f"AP{sep}1.5s"]
        ) / 4

        eval_results[f"num_samples{sep}"] = len(preds)
        return eval_results
