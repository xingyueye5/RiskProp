import os
import math
import json
import random
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import average_precision_score
import pandas as pd


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, save_path):
    torch.save(state, save_path)


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)


def compute_ap(y_true, y_score):
    # y_true and y_score are 1D arrays
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return 0.0


def compute_mTTA(predictions_by_video, annotations, threshold=0.5):
    # predictions_by_video: dict video_id -> list of scores per frame (length num_frames)
    # annotations: dict video_id -> collision_frame (1-indexed or -1 if none)
    tta_list = []
    for vid, scores in predictions_by_video.items():
        coll = annotations.get(vid, -1)
        if coll <= 0:
            continue
        # find first frame index where score >= threshold
        pred_idx = None
        for i, s in enumerate(scores, start=1):
            if s >= threshold:
                pred_idx = i
                break
        if pred_idx is None:
            # never predicted
            continue
        tta = coll - pred_idx
        tta_list.append(tta)
    if len(tta_list) == 0:
        return None
    return float(sum(tta_list) / len(tta_list))


def eval_suite_from_sequences(preds_list, labels_list, abnormal_start_inds, accident_inds, fpr_max=0.1):
    """Replicate key metrics from taa/metrics.py without mmaction.

    Inputs are lists over samples:
      - preds_list: list[np.ndarray shape (T,)] per sample
      - labels_list: list[bool or 0/1 array] per sample (bool indicates target)
      - abnormal_start_inds: list[int] per sample
      - accident_inds: list[int] per sample
    Returns dict of metrics including thresholds, tta/mtta, AUC/AP slices.
    """
    results = {}

    preds = [p if p.ndim == 1 else p.max(axis=-1) for p in preds_list]
    labels = [bool(l) if (isinstance(l, (bool, np.bool_))) else bool(np.max(l) > 0.5) for l in labels_list]

    preds_n = -np.sort(-np.concatenate([p for p, l in zip(preds, labels) if not l]))
    if len(preds_n) == 0:
        thr = 0.5
    else:
        idx = min(int(len(preds_n) * fpr_max), len(preds_n) - 1)
        thr = float(preds_n[idx])
    results[f"threshold@{fpr_max:.2f}"] = thr

    # tta style metrics at 0.5
    ttas = [
        (i - np.argmax(p[: i + 1] >= 0.5)) / 10 if np.any(p[: i + 1] >= 0.5) else 0
        for p, l, i in zip(preds, labels, accident_inds) if l
    ]
    if len(ttas) > 0:
        results[f"tta_old@0.5"] = float(sum(ttas) / len(ttas))

    # AUC/AP slices similar to taa/metrics.py
    def _auc(y_true, y_score, fpr_max_local=0.1):
        sorted_idx = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_idx]
        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)
        TPR = [0]
        FPR = [0]
        TP = 0
        FP = 0
        for v in y_true_sorted:
            if v == 1:
                TP += 1
            else:
                FP += 1
            TPR.append(TP / max(P, 1))
            FPR.append(FP / max(N, 1))
        area = 0.0
        for k in range(1, len(FPR)):
            if FPR[k] > fpr_max_local:
                break
            dx = FPR[k] - FPR[k - 1]
            dy = TPR[k] + TPR[k - 1]
            area += dx * dy / 2
        return float(area / max(fpr_max_local, 1e-6))

    preds_0 = [np.max(p[-5:]) for p, l in zip(preds, labels) if l]
    preds_5 = [np.max(p[-10:-5]) for p, l in zip(preds, labels) if l]
    preds_10 = [np.max(p[-15:-10]) for p, l in zip(preds, labels) if l]
    preds_15 = [np.max(p[-20:-15]) for p, l in zip(preds, labels) if l]
    preds_n_end = [np.max(p[-5:]) for p, l in zip(preds, labels) if not l]
    y = np.array([1] * len(preds_0) + [0] * len(preds_n_end))
    if len(y) > 1 and len(preds_n_end) > 0:
        results[f"AUC@0.5s"] = _auc(y, np.array(preds_5 + preds_n_end), fpr_max)
        results[f"AUC@1.0s"] = _auc(y, np.array(preds_10 + preds_n_end), fpr_max)
        results[f"AUC@1.5s"] = _auc(y, np.array(preds_15 + preds_n_end), fpr_max)
        try:
            results[f"AP_full@0.5s"] = float(average_precision_score(y, np.array(preds_5 + preds_n_end)))
            results[f"AP_full@1.0s"] = float(average_precision_score(y, np.array(preds_10 + preds_n_end)))
            results[f"AP_full@1.5s"] = float(average_precision_score(y, np.array(preds_15 + preds_n_end)))
        except Exception:
            pass
    return results


def write_inference_csv(preds_by_vid: dict, anns: dict, out_path: str):
    rows = []
    for vid, scores in preds_by_vid.items():
        rows.append({'video_id': vid, 'target': f"{max(scores[-1], 0.0):.4f}"})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)