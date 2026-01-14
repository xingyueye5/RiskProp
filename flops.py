#!/usr/bin/env python3
"""Utility script to measure MACs and parameters for MMACTION configs."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - guard for missing deps
    raise SystemExit(
        "PyTorch is required to compute FLOPs. "
        "Install it first, e.g. `pip install torch`."
    ) from exc

try:
    from mmengine import Config
    from mmengine.analysis import get_model_complexity_info
    from mmengine.registry import init_default_scope
    from mmaction.registry import MODELS
    from mmaction.utils import register_all_modules
except ModuleNotFoundError as exc:  # pragma: no cover - guard for missing deps
    missing = getattr(exc, "name", None) or str(exc)
    raise SystemExit(
        f'Required module "{missing}" is not installed. '
        "Install the OpenMMLab runtime stack first, e.g. "
        "`pip install mmengine>=0.7.1 mmcv>=2.0.0rc4`."
    ) from exc


PIPELINE_KEYS = [
    "train_pipeline_video",
    "val_pipeline_video",
    "test_pipeline_video",
    "train_pipeline_frame",
    "val_pipeline_frame",
    "test_pipeline_frame",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FLOPs and params for a given mmaction config."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/predict_anomaly_snippet.py",
        help="Path to the config file (default: configs/predict_anomaly_snippet.py)",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        help=(
            "Explicit input shape tuple. "
            "Example for 3D recognizer: --shape 1 30 3 5 224 224 "
            "(batch, num_clips, C, T, H, W)."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for the dummy input."
    )
    parser.add_argument("--clip-len", type=int, help="Override clip length.")
    parser.add_argument("--num-clips", type=int, help="Override number of clips/views.")
    parser.add_argument("--channels", type=int, help="Override input channels.")
    parser.add_argument("--height", type=int, help="Override input height.")
    parser.add_argument("--width", type=int, help="Override input width.")
    parser.add_argument(
        "--device", default="cpu", help="Device used for analysis (default: cpu)."
    )
    parser.add_argument(
        "--per-layer", action="store_true", help="Print per-layer statistics."
    )
    return parser.parse_args()


def import_custom_modules(cfg: Config) -> None:
    """Respect ``custom_imports`` in config so registries are populated."""
    custom = cfg.get("custom_imports")
    if not custom:
        return
    modules = custom.get("imports", [])
    if isinstance(modules, (str, Path)):
        modules = [modules]
    allow_failure = custom.get("allow_failed_imports", False)
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            if allow_failure:
                continue
            raise


def disable_pretrained_fields(node) -> None:
    """Prevent downloading checkpoints when instantiating the model."""
    if isinstance(node, dict):
        if "pretrained" in node:
            node["pretrained"] = None
        for value in node.values():
            disable_pretrained_fields(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            disable_pretrained_fields(item)


def infer_resolution(cfg: Config, height: int | None, width: int | None) -> Tuple[int, int]:
    if height is not None and width is not None:
        return height, width

    def _normalize_pair(pair: Sequence[int]) -> Tuple[int, int]:
        if len(pair) == 2:
            return int(pair[1]), int(pair[0])
        val = int(pair[0])
        return val, val

    for key in PIPELINE_KEYS:
        pipeline = cfg.get(key)
        if not pipeline:
            continue
        for transform in pipeline:
            if not isinstance(transform, dict):
                continue
            if "scale" in transform:
                scale = transform["scale"]
                if isinstance(scale, int):
                    return scale, scale
                if isinstance(scale, (tuple, list)):
                    return _normalize_pair(scale)
            if "crop_size" in transform:
                crop = transform["crop_size"]
                if isinstance(crop, int):
                    return crop, crop
                if isinstance(crop, (tuple, list)):
                    return _normalize_pair(crop)

    fallback = cfg.get("input_size")
    if isinstance(fallback, int):
        return fallback, fallback
    if isinstance(fallback, (tuple, list)) and fallback:
        return _normalize_pair(fallback)
    return 224, 224


def infer_channels(cfg: Config, override: int | None) -> int:
    if override is not None:
        return override
    modality = str(cfg.get("modality", "rgb")).lower()
    if modality in {"rgb", "both", "two_stream"}:
        return 3
    if modality == "flow":
        return 3  # stacked optical flow (u, v, magnitude)
    data_preprocessor = cfg.model.get("data_preprocessor")
    if data_preprocessor and "mean" in data_preprocessor:
        return len(data_preprocessor["mean"])
    return 3


def infer_clip_meta(cfg: Config, key: str, override: int | None, default: int) -> int:
    if override is not None:
        return override
    if key in cfg:
        return int(cfg[key])
    return default


def build_input_shape(cfg: Config, args: argparse.Namespace) -> Tuple[int, ...]:
    if args.shape:
        return tuple(args.shape)

    model_type = str(cfg.model.get("type", "")).lower()
    is_video = "3d" in model_type or "slowfast" in model_type

    height, width = infer_resolution(cfg, args.height, args.width)
    channels = infer_channels(cfg, args.channels)
    if is_video:
        clip_len = infer_clip_meta(cfg, "clip_len", args.clip_len, default=1)
        num_clips = infer_clip_meta(cfg, "num_clips", args.num_clips, default=1)
        return (args.batch_size, num_clips, channels, clip_len, height, width)
    return (args.batch_size, channels, height, width)


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)

    register_all_modules(init_default_scope=False)
    import_custom_modules(cfg)
    init_default_scope(cfg.get("default_scope", "mmaction"))

    disable_pretrained_fields(cfg.model)

    model = MODELS.build(cfg.model)
    model.eval()
    model.to(args.device)

    if hasattr(model, "extract_feat"):
        extractor = model.extract_feat

        def tensor_forward(inputs: torch.Tensor) -> torch.Tensor:
            feats, _ = extractor(inputs)
            return feats if isinstance(feats, torch.Tensor) else feats[0]

        model.forward = tensor_forward  # type: ignore[assignment]

    input_shape = build_input_shape(cfg, args)
    print(f"Using dummy input shape: {input_shape}")

    with torch.no_grad():
        analysis = get_model_complexity_info(
            model,
            input_shape,
            as_strings=True,
            print_per_layer_stat=args.per_layer,
        )

    table = analysis.get("out_table")
    if table:
        print(table)

    flops = analysis.get("flops_str", analysis.get("flops"))
    params = analysis.get("params_str", analysis.get("params"))
    print(f"MACs: {flops}")
    print(f"Params: {params}")


if __name__ == "__main__":
    main()
