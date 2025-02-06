# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
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

from sklearn.metrics import accuracy_score, precision_score, recall_score
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
class EvaluationMetric(BaseMetric):
    """Accuracy evaluation metric."""

    default_prefix: Optional[str] = ""

    def __init__(
        self,
        thresholds=[0.5],
        vis_list=[],
        output_dir=None,
        test_mode=True,
        metric_list: Optional[Union[str, Tuple[str]]] = ("top_k_accuracy", "mean_class_accuracy"),
        collect_device: str = "cpu",
        metric_options: Optional[Dict] = dict(top_k_accuracy=dict(topk=(1, 5))),
        prefix: Optional[str] = None,
    ) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError("metric_list must be str or tuple of str, " f"but got {type(metric_list)}")

        if isinstance(metric_list, str):
            metrics = (metric_list,)
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                "top_k_accuracy",
                "mean_class_accuracy",
                "mmit_mean_average_precision",
                "mean_average_precision",
            ]

        self.metrics = metrics
        self.metric_options = metric_options
        self.thresholds = thresholds
        self.vis_list = vis_list
        self.output_dir = output_dir
        self.epoch = None
        self.test_mode = test_mode

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
            if self.test_mode != data_sample["is_test"]:
                continue
            result = dict()
            pred = data_sample["pred_score"].cpu().numpy()
            label = np.zeros_like(pred)
            frame_inds = data_sample["frame_inds"]
            frame_interval = data_sample["frame_interval"]
            accident_frame = data_sample["accident_frame"]
            index = np.argmin(np.abs(frame_inds - accident_frame))
            if np.abs(frame_inds[index] - accident_frame) < frame_interval:
                label[index] = 1
            result["pred"] = pred
            result["label"] = label
            self.results.append(result)
            video_id = data_sample["video_id"]
            if video_id in self.vis_list:
                visualize_pred_score(data_sample, result, self.output_dir, epoch=self.epoch)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x["label"] for x in results]
        preds = [x["pred"] for x in results]
        return self.calculate(preds, labels)

    def calculate(self, preds: List[np.ndarray], labels: List[Union[int, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        sep = "@" if self.test_mode else "#"
        for t in self.thresholds:
            preds_t = [pred >= t for pred in preds]
            labels_t = [label == 1 for label in labels]
            pred_videos = np.array([np.any(pred) for pred in preds_t])
            label_videos = np.array([np.any(label) for label in labels_t])
            pred_frames = np.concatenate(preds_t)
            label_frames = np.concatenate(labels_t)
            # Only calculate tta when there is a true positive sample
            ttas = [
                (np.where(label)[0][0] - np.where(pred)[0][0]) / 10
                for pred, label in zip(preds_t, labels_t)
                if np.any(label) and np.any(pred)
            ]

            eval_results[f"\nv_acc{sep}{t:.1f}"] = accuracy_score(label_videos, pred_videos)
            eval_results[f"v_pre{sep}{t:.1f}"] = precision_score(label_videos, pred_videos, zero_division=0)
            eval_results[f"v_rec{sep}{t:.1f}"] = recall_score(label_videos, pred_videos, zero_division=0)
            eval_results[f"f_acc{sep}{t:.1f}"] = accuracy_score(label_frames, pred_frames)
            eval_results[f"f_pre{sep}{t:.1f}"] = precision_score(label_frames, pred_frames, zero_division=0)
            eval_results[f"f_rec{sep}{t:.1f}"] = recall_score(label_frames, pred_frames, zero_division=0)
            eval_results[f"tta{sep}{t:.1f}"] = sum(ttas) / len(ttas) if len(ttas) else 0

        metrics = ["\nv_acc", "v_pre", "v_rec", "f_acc", "f_pre", "f_rec"]
        for metric in metrics:
            eval_results[f"{metric}{sep}max"] = max([eval_results[f"{metric}{sep}{t:.1f}"] for t in self.thresholds])
        eval_results[f"tta{sep}min"] = min([eval_results[f"tta{sep}{t:.1f}"] for t in self.thresholds])
        eval_results[f"\n{sep}num_samples"] = len(labels)
        return eval_results


@METRICS.register_module()
class SnippetMetric(EvaluationMetric):
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
            if self.test_mode != data_sample["is_test"]:
                continue
            result = dict()
            result["pred"] = data_sample["pred_score"].cpu().numpy()
            result["label"] = data_sample["gt_label"].cpu().numpy()
            self.results.append(result)

            if data_sample["video_id"] in self.vis_list:
                data_sample["frame_inds"] = data_sample["frame_inds"].reshape(-1, data_sample["clip_len"])[
                    :, int(np.ceil((data_sample["clip_len"] - 1) / 2))
                ]
                visualize_pred_score(data_sample, result, self.output_dir, epoch=self.epoch)
