# RiskProp: Collision-Anchored Self-supervised Temporal Constraints for Early Accident Anticipation

> **CVPR 2026**
>
> [[Paper](#)] · [[Project Page](#)] · [[Video](#)]

---

## Overview

**RiskProp** is a dashcam-based traffic accident anticipation framework that predicts frame-level collision risk before an accident occurs. It introduces collision-anchored self-supervised temporal constraints to enforce monotonically increasing risk scores as the accident approaches, without requiring dense per-frame annotations.

Built on [MMAction2](https://github.com/open-mmlab/mmaction2), RiskProp supports multi-dataset training across CAP, DADA, D²-City, and Nexar, and provides both frame-level and snippet-level prediction modes.

![Data Pipeline](resources/data_pipeline.png)

---

## Installation

```bash
conda create -n riskprop python=3.8.5 -y
conda activate riskprop

pip install -r requirements.txt
pip install torch torchvision
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.2.0
mim install mmaction2==1.2.0
pip install -v -e .
```

---

## Data Preparation

Organize datasets under `data/` as follows:

```
data/
├── MM-AU/
│   ├── CAP-DATA/
│   │   ├── cap_text_annotations.xls
│   │   ├── 1-10/
│   │   ├── 11/
│   │   ├── 12-42/
│   │   └── ...
│   └── DADA-DATA/
│       ├── dada_text_annotations.xlsx
│       └── ...
├── D_square-City/
│   ├── annotations.csv
│   └── raw/
└── nexar-collision-prediction/
    ├── annotations.csv
    ├── train/
    ├── test/
    ├── train_raw_frames/
    └── test_raw_frames/
```

Dataset paths are configured at the top of each config file under `configs/`.

---

## Training

Edit `dist_train.sh` to set the config `name` and `CUDA_VISIBLE_DEVICES`, then run:

```bash
bash dist_train.sh
```

This saves a timestamped code snapshot to `codes/` and launches distributed training. To run directly:

```bash
# Snippet-level anticipation (SlowOnly-R50, constraint label) — main model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 \
    tools/dist_train.sh configs/predict_anomaly_snippet.py 8

# Frame-level anticipation (ResNet-50 + LSTM, annotation label)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 \
    tools/dist_train.sh configs/predict_anomaly_frame.py 8

# Occurrence prediction (SlowOnly-R50 + Transformer decoder)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 \
    tools/dist_train.sh configs/predict_occurrence_snippet.py 8
```

Checkpoints and logs are saved to `work_dirs/<config_name>/`. The best checkpoint is selected by `mAUC@` (partial AUC under FPR ≤ 0.1).

**Useful training flags:**

| Flag | Description |
|------|-------------|
| `--resume` | Auto-resume from the latest checkpoint |
| `--amp` | Enable automatic mixed precision |
| `--auto-scale-lr` | Scale LR proportionally to actual batch size |
| `--cfg-options key=val` | Override any config field inline |

For Slurm clusters:

```bash
tools/slurm_train.sh <partition> <job_name> configs/predict_anomaly_snippet.py --gpus 8
```

---

## Evaluation

Edit `dist_test.sh` to set the config `name` and checkpoint path, then run:

```bash
bash dist_test.sh
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 \
    tools/dist_test.sh configs/predict_anomaly_snippet.py \
    work_dirs/predict_anomaly_snippet/<checkpoint>.pth 8
```

To dump predictions for offline analysis:

```bash
tools/dist_test.sh configs/predict_anomaly_snippet.py <checkpoint> 8 \
    --dump outputs/predictions.pkl
```

Results are saved to `work_dirs/`. Check `metrics_AUC.png` to identify the best epoch, then inspect the corresponding log for detailed metrics.

---

## Configs

| Config | Backbone | Clip | Label | Notes |
|--------|----------|------|-------|-------|
| `predict_anomaly_snippet.py` | SlowOnly-R50 | 5-frame × 30 clips | constraint | Main model |
| `predict_anomaly_frame.py` | ResNet-50 + LSTM | 1-frame × 30 clips | annotation | Frame-level |
| `predict_occurrence_snippet.py` | SlowOnly-R50 | 5-frame × 30 clips | annotation | With Transformer decoder |

---

## Metrics

| Metric | Description |
|--------|-------------|
| `mAUC@` | Mean partial AUC (FPR ≤ 0.1) at 0.5s, 1.0s, 1.5s before accident **(primary)** |
| `mAUC` | Mean full-curve AUC at 0.5s, 1.0s, 1.5s |
| `mAP` | Mean Average Precision at 0.5s, 1.0s, 1.5s |
| `mTTA@0.1` | Mean Time-To-Accident at FPR ≤ 0.1 |

---

## Model Zoo

| Config | Dataset | mAUC@ | mAUC | mAP | Download |
|--------|---------|-------|------|-----|----------|
| `predict_anomaly_snippet` | CAP | - | - | - | [model](#) \| [log](#) |
| `predict_anomaly_frame` | CAP | - | - | - | [model](#) \| [log](#) |

---

## Citation

```bibtex
@inproceedings{riskprop2026,
  title     = {RiskProp: Collision-Anchored Self-supervised Temporal Constraints for Early Accident Anticipation},
  author    = {zyy, zth},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Acknowledgements

This project is built on [MMAction2](https://github.com/open-mmlab/mmaction2). We thank the OpenMMLab team for their excellent codebase.

## License

This project is released under the [Apache 2.0 License](LICENSE).
