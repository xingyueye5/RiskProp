# -*- coding: utf-8 -*-
"""Distributed training script (CAP baseline, revision 2).

Highlights
---------
- Handles per-sample metadata (labels, TOA, texts) instead of borrowing the
  first element in a batch.
- Preserves multi-modal text descriptions by wiring them from the dataset to
  the CAP model.
- Keeps time-to-accident penalties consistent with sampled clip intervals.
- Compatible with both single-GPU and torchrun DDP launches.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


def _normalize_env_pre_import() -> None:
    """Standardise NCCL-related env vars before torch/mmengine imports."""

    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    local_rank_raw = os.environ.get("LOCAL_RANK", "0")
    try:
        int(local_rank_raw)
    except Exception:
        os.environ["LOCAL_RANK"] = "0"

    cvd_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd_raw:
        devices = [d.strip() for d in cvd_raw.split(",") if d.strip()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    try:
        cuda_cnt = torch.cuda.device_count()
    except Exception:
        cuda_cnt = 0
    print(
        f"[pre-import] pid={os.getpid()} LR={os.environ.get('LOCAL_RANK')} "
        f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} cuda_count={cuda_cnt}",
        file=sys.stderr,
    )


_normalize_env_pre_import()

import importlib

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> int:
    """Initialise torch distributed if env vars indicate torchrun launch."""

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(
            f"[DDP] rank={dist.get_rank()} / {dist.get_world_size()} "
            f"(local_rank={local_rank})",
            file=sys.stderr,
        )
        return local_rank
    print("[Single GPU] non-distributed mode.", file=sys.stderr)
    return 0


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# monkey patch mmengine logger device detection (avoids CVD parsing issues)
mm_logger_mod = importlib.import_module("mmengine.logging.logger")


def _safe_get_device_id() -> int:
    if torch.cuda.is_available():
        try:
            return torch.cuda.current_device()
        except Exception:
            return 0
    return 0


mm_logger_mod._get_device_id = _safe_get_device_id  # type: ignore[attr-defined]

# remaining heavy deps (after env normalisation)
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.registry import init_default_scope

import torch.optim as optim

from mmaction.utils import register_all_modules

from taa.metrics import AnticipationMetric
from ablation.CAP.src.model import accident
from ablation.CAP.src.bert import opt


def _gather_attr(ds: Any, key: str, default: Any = None) -> Any:
    if hasattr(ds, key):
        value = getattr(ds, key)
        if value is not None:
            return value
    if hasattr(ds, "metainfo") and key in ds.metainfo and ds.metainfo[key] is not None:
        return ds.metainfo[key]
    if hasattr(ds, "algorithms") and key in ds.algorithms and ds.algorithms[key] is not None:
        return ds.algorithms[key]
    return default


def _ensure_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, Sequence):
        tensor = torch.tensor(value)
    elif value is None:
        tensor = torch.tensor([])
    else:
        tensor = torch.tensor([value])
    return tensor.to(device=device, dtype=dtype)


def extract_batch_fields(
    data_batch: Dict[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Collate inputs plus metadata for one optimiser/eval step."""

    inputs = torch.stack(tuple(data_batch["inputs"]), dim=0)

    B, num_clips, C, T, H, W = inputs.shape
    inputs = inputs.reshape(B, num_clips * T, C, H, W)
    x_cap = inputs.to(device=device, non_blocking=True).float() / 255.0

    data_samples = data_batch["data_samples"]
    batch = len(data_samples)

    targets = torch.zeros(batch, 2, device=device, dtype=torch.float32)
    toa_steps = torch.zeros(batch, device=device, dtype=torch.float32)
    step_rates = torch.zeros(batch, device=device, dtype=torch.float32)
    texts: List[str] = []

    for idx, ds in enumerate(data_samples):
        target_flag = bool(_gather_attr(ds, "target", False))
        if target_flag:
            targets[idx, 1] = 1.0
        else:
            targets[idx, 0] = 1.0

        fps = float(_gather_attr(ds, "fps", 30) or 30)
        accident_frame = _gather_attr(ds, "accident_frame", None)

        frame_inds = _gather_attr(ds, "frame_inds", None)
        frame_inds_tensor = _ensure_tensor(frame_inds, device=device, dtype=torch.int64)
        if frame_inds_tensor.ndim > 1:
            frame_inds_tensor = frame_inds_tensor.flatten()
        if frame_inds_tensor.numel() == 0:
            frame_inds_tensor = torch.arange(x_cap.size(1), device=device, dtype=torch.int64)

        if frame_inds_tensor.numel() == 1:
            frame_interval = 1
        else:
            diffs = torch.diff(frame_inds_tensor)
            frame_interval = int(torch.mode(diffs)[0].item()) if diffs.numel() > 0 else 1
            frame_interval = max(frame_interval, 1)

        steps_per_second = fps / frame_interval
        step_rates[idx] = torch.tensor(steps_per_second, device=device, dtype=torch.float32)

        if target_flag and accident_frame is not None:
            accident_frame = int(accident_frame)
            indices = torch.nonzero(frame_inds_tensor >= accident_frame, as_tuple=False)
            if indices.numel() > 0:
                toa_idx = int(indices[0].item())
            else:
                toa_idx = frame_inds_tensor.numel() - 1
        else:
            toa_idx = frame_inds_tensor.numel() + 5

        toa_steps[idx] = torch.tensor(float(toa_idx), device=device)

        text_value = _gather_attr(ds, "text", "") or ""
        if isinstance(text_value, (list, tuple)):
            text_value = " ".join(str(part) for part in text_value)
        texts.append(str(text_value))

    return x_cap, targets, toa_steps, step_rates, texts


@torch.no_grad()
def _eval_step_to_metric(
    preds_np: np.ndarray, data_batch: Dict[str, Any], metric: AnticipationMetric
) -> None:
    for i in range(preds_np.shape[0]):
        ds = data_batch["data_samples"][i]
        result = dict(
            pred_score=torch.from_numpy(preds_np[i]),
            target=bool(_gather_attr(ds, "target", False)),
            abnormal_start_frame=_gather_attr(ds, "abnormal_start_frame", 0),
            accident_frame=_gather_attr(ds, "accident_frame", 0),
            frame_inds=_gather_attr(ds, "frame_inds", torch.arange(preds_np.shape[1]).numpy()),
            video_id=_gather_attr(ds, "video_id", f"video_{i}"),
            dataset=_gather_attr(ds, "dataset", "unknown"),
            frame_dir=_gather_attr(ds, "frame_dir", ""),
            filename_tmpl=_gather_attr(ds, "filename_tmpl", "{:06}.jpg"),
            type=_gather_attr(ds, "type", ""),
            is_val=bool(_gather_attr(ds, "is_val", False)),
            is_test=bool(_gather_attr(ds, "is_test", False)),
        )
        metric.process(None, [result])


def run_epoch(
    cap_model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    metric: AnticipationMetric | None = None,
    rank: int = 0,
) -> float:
    training = optimizer is not None
    cap_model.train(training)

    total_loss = 0.0
    step_count = 0

    for data_batch in loader:
        x_cap, targets, toa_steps, step_rates, texts = extract_batch_fields(data_batch, device)
        losses, outputs = cap_model(x_cap, targets, toa_steps, texts, step_rate=step_rates)
        loss = losses["total_loss"]

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        step_count += 1

        if metric is not None and rank == 0:
            logits = torch.stack(outputs, dim=1)
            preds_np = logits.softmax(dim=-1)[..., 1].detach().cpu().numpy()
            _eval_step_to_metric(preds_np, data_batch, metric)

    avg_loss = total_loss / max(step_count, 1)
    avg_loss_tensor = torch.tensor([avg_loss], device=device)

    if dist.is_initialized():
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_tensor /= dist.get_world_size()
        avg_loss = float(avg_loss_tensor.item())

    return avg_loss


def main() -> None:
    local_rank = setup_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    register_all_modules()

    cfg = Config.fromfile("configs/predict_anomaly_snippet.py")
    init_default_scope("mmaction")

    if not hasattr(cfg, "launcher"):
        cfg.launcher = "pytorch"
    if "work_dir" not in cfg:
        cfg.work_dir = "./work_dirs/cap_baseline"

    for key in ("train_dataloader", "val_dataloader"):
        if hasattr(cfg, key):
            dataloader_cfg = getattr(cfg, key)
            if isinstance(dataloader_cfg, dict):
                dataloader_cfg["num_workers"] = min(int(dataloader_cfg.get("num_workers", 4)), 4)
                dataloader_cfg["persistent_workers"] = False
                sampler_cfg = dataloader_cfg.get("sampler", {})
                if isinstance(sampler_cfg, dict):
                    sampler_cfg["round_up"] = True

    runner = Runner.from_cfg(cfg)

    global_rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = global_rank == 0
    if is_main:
        os.makedirs(cfg.work_dir, exist_ok=True)

    train_loader = runner.build_dataloader(cfg.train_dataloader)
    val_loader = runner.build_dataloader(cfg.val_dataloader)

    if dist.is_initialized():
        dist.barrier()

    cap_model = accident(
        h_dim=opt.s_dim2,
        n_layers=1,
        depth=opt.tran_num_layers,
        adim=opt.adim,
        heads=opt.heads,
        num_tokens=opt.num_tokens,
        c_dim=opt.c_dim,
        s_dim1=opt.s_dim1,
        s_dim2=opt.s_dim2,
        keral=opt.keral,
        num_class=opt.num_class,
    ).to(device)

    if dist.is_initialized():
        cap_model = DDP(
            cap_model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    optimizer = optim.AdamW(cap_model.parameters(), lr=1e-4, weight_decay=1e-4)
    metric = AnticipationMetric(fpr_max=0.1, output_dir="outputs") if is_main else None

    max_epochs = 30
    best_mauc = -1.0
    best_epoch = -1
    save_dir = "outputs"
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    try:
        for epoch in range(1, max_epochs + 1):
            if dist.is_initialized():
                for loader in (train_loader, val_loader):
                    sampler = getattr(loader, "sampler", None)
                    if hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(epoch)

        train_loss = run_epoch(cap_model, train_loader, device, optimizer=optimizer, metric=None, rank=global_rank)
        if is_main:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        if dist.is_initialized():
            dist.barrier()

        if is_main and metric is not None:
            if hasattr(metric, "results"):
                metric.results.clear()
            metric.epoch = epoch

        val_metric = metric if is_main else None
        with torch.no_grad():
            _ = run_epoch(cap_model, val_loader, device, optimizer=None, metric=val_metric, rank=global_rank)

        if dist.is_initialized():
            dist.barrier()

        if is_main and metric is not None:
            raw_results = metric.compute_metrics(metric.results)
            metric.results.clear()
            print(f"[Epoch {epoch}] Val Results: {raw_results}")

            mauc_key = next((k for k in raw_results if k.startswith("mAUC")), None)
            current_mauc = float(raw_results.get(mauc_key, 0.0)) if mauc_key else 0.0
            if current_mauc > best_mauc:
                best_mauc = current_mauc
                best_epoch = epoch
                state_dict = cap_model.module.state_dict() if hasattr(cap_model, "module") else cap_model.state_dict()
                best_path = os.path.join(save_dir, f"best_mAUC_epoch_{epoch:03d}_{best_mauc:.4f}.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_mAUC": best_mauc,
                    },
                    best_path,
                )
                print(f"✅ [Epoch {epoch}] New best mAUC = {best_mauc:.4f}, saved to {best_path}")

        if is_main:
            print(f"\n🎯 Training done. Best mAUC = {best_mauc:.4f} at epoch {best_epoch}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
