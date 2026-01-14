# -*- coding: utf-8 -*-
"""
compare_xai.py — 分布式训练 AccidentXai 模型，复用 compare_dis.py 的数据/评估流程。
- 统一 NCCL 环境设置，避免分布式超时
- 使用 mmengine Runner 构建数据加载，并复用 AnticipationMetric
- 保持原 XAI 模型与损失逻辑不变，仅桥接输入数据与标签格式
"""

import os
import sys
import pathlib
import contextlib
from typing import Dict, Any, List

import torch
import numpy as np


def _normalize_env_pre_import():
    """提前设置常见 NCCL 变量，提升稳定性。"""
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')

    lr_raw = os.environ.get('LOCAL_RANK', '0')
    try:
        int(lr_raw)
    except Exception:
        os.environ['LOCAL_RANK'] = '0'

    cvd_raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    parts = [p.strip() for p in cvd_raw.split(',') if p.strip()]
    if parts:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(parts)

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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> int:
    """初始化分布式后端，返回 local_rank。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(
            f"[DDP] initialized rank={dist.get_rank()} / {dist.get_world_size()} "
            f"(local_rank={local_rank})",
            file=sys.stderr,
        )
        return local_rank
    print("[Single GPU] non-distributed mode.", file=sys.stderr)
    return 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


import importlib

mm_logger_mod = importlib.import_module('mmengine.logging.logger')


def _safe_get_device_id():
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return 0
    return 0


mm_logger_mod._get_device_id = _safe_get_device_id  # type: ignore

from mmengine.runner import Runner
from mmengine.config import Config as MMEngineConfig
from mmengine.registry import init_default_scope
from mmaction.utils import register_all_modules
import torch.optim as optim

# 共享 AnticipationMetric
cap_metrics_mod = importlib.import_module('taa.metrics')
AnticipationMetric = cap_metrics_mod.AnticipationMetric

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
XAI_ROOT = PROJECT_ROOT / 'ablation' / 'xai'
if str(XAI_ROOT) not in sys.path:
    sys.path.insert(0, str(XAI_ROOT))

from src.model import AccidentXai  # type: ignore  # noqa: E402


DEFAULT_XAI_CFG = dict(
    num_epochs=10,
    lr=1e-4,
    clip_grad=10.0,
    feature_mean=0.5,
    feature_std=0.5,
)


def _get_sample_attr(ds, key, default=None):
    if hasattr(ds, key) and getattr(ds, key) is not None:
        return getattr(ds, key)
    if hasattr(ds, 'metainfo') and ds.metainfo is not None and key in ds.metainfo:
        return ds.metainfo[key]
    if hasattr(ds, 'algorithms') and ds.algorithms is not None and key in ds.algorithms:
        return ds.algorithms[key]
    return default


def one_hot(targets: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    out = torch.zeros(targets.size(0), num_classes, device=targets.device, dtype=torch.float32)
    out.scatter_(1, targets.view(-1, 1), 1.0)
    return out


def prepare_batch(
    data_batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    inputs = data_batch['inputs']
    if isinstance(inputs, (list, tuple)):
        x = torch.cat(inputs, dim=0)
    else:
        x = inputs
    x = x.permute(2, 0, 1, 3, 4).contiguous()  # -> (B, T, C, H, W)
    data_samples: List[Any] = data_batch['data_samples']
    batch_size = len(data_samples)
    if x.size(0) != batch_size:
        raise RuntimeError(f"Mismatched batch size: tensor {x.size(0)} vs samples {batch_size}")

    frames = x.float().to(device) / 255.0
    # 原代码 Normalize(mean=0.5, std=0.5)
    frames = (frames - DEFAULT_XAI_CFG['feature_mean']) / DEFAULT_XAI_CFG['feature_std']

    T = frames.size(1)
    targets, toas = [], []
    for ds in data_samples:
        target = 1 if bool(_get_sample_attr(ds, 'target', False)) else 0
        frame_inds = _get_sample_attr(ds, 'frame_inds', None)
        if frame_inds is None:
            frame_inds = np.arange(T)
        else:
            frame_inds = np.array(frame_inds)
        accident_frame = _get_sample_attr(ds, 'accident_frame', None)
        if target == 1 and accident_frame is not None:
            hit = np.where(frame_inds >= accident_frame)[0]
            toa_idx = float(hit[0]) if hit.size > 0 else float(T - 1)
        elif target == 1:
            toa_idx = float(T - 1)
        else:
            toa_idx = float(T + 1)
        targets.append(target)
        toas.append(toa_idx)

    y = one_hot(torch.tensor(targets, device=device, dtype=torch.long))
    toa_tensor = torch.tensor(toas, device=device, dtype=torch.float32)
    return dict(frames=frames, labels=y, toa=toa_tensor)


@torch.no_grad()
def _eval_step_to_metric(preds_np: np.ndarray, data_batch: Dict[str, Any], metric: AnticipationMetric, T: int):
    data_samples = data_batch['data_samples']
    for i in range(preds_np.shape[0]):
        ds = data_samples[i]

        def get_k(k, default=None):
            return _get_sample_attr(ds, k, default)

        result = dict(
            pred_score=torch.from_numpy(preds_np[i]),
            target=bool(get_k('target', False)),
            abnormal_start_frame=get_k('abnormal_start_frame', 0),
            accident_frame=get_k('accident_frame', 0),
            frame_inds=get_k('frame_inds', torch.arange(T).numpy()),
            video_id=get_k('video_id', f"video_{i}"),
            dataset=get_k('dataset', 'unknown'),
            frame_dir=get_k('frame_dir', ''),
            filename_tmpl=get_k('filename_tmpl', '{:06}.jpg'),
            type=get_k('type', ''),
            is_val=bool(get_k('is_val', False)),
            is_test=bool(get_k('is_test', False)),
        )
        metric.process(None, [result])


def configure_trainable_params(model: AccidentXai):
    """对齐原脚本的冻结策略。"""
    for name, param in model.features.named_parameters():
        if "fc.0.weight" in name or "fc.0.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in model.gru_net.named_parameters():
        if 'gru.weight' in name or 'gru.bias' in name:
            param.requires_grad = True
        elif 'dense1' in name or 'dense2' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def run_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    optimizer: optim.Optimizer = None,
    metric: AnticipationMetric = None,
    rank: int = 0,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: Dict[str, float] = {}
    batches = 0
    ctx = contextlib.nullcontext() if train_mode else torch.no_grad()

    for data_batch in loader:
        batch = prepare_batch(data_batch, device)
        frames, labels, toa = batch['frames'], batch['labels'], batch['toa']

        with ctx:
            losses, all_outputs = model(frames, labels, toa)

        total_loss = losses['total_loss']
        if train_mode:
            optimizer.zero_grad()
            total_loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_XAI_CFG['clip_grad'])
            optimizer.step()

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().mean().cpu())
        batches += 1

        if metric is not None and rank == 0:
            logits = torch.stack(all_outputs, dim=1).to(device)  # (B, T, 2)
            probs = torch.softmax(logits, dim=-1)[..., 1].detach().cpu().numpy()
            _eval_step_to_metric(probs, data_batch, metric, probs.shape[1])

    if batches == 0:
        return {key: 0.0 for key in totals}

    avg_losses: Dict[str, float] = {}
    for key, value in totals.items():
        avg = torch.tensor([value / batches], device=device)
        if dist.is_initialized():
            dist.all_reduce(avg, op=dist.ReduceOp.SUM)
            avg /= dist.get_world_size()
        avg_losses[key] = avg.item()
    return avg_losses


def build_model(device: torch.device):
    num_classes = 2
    x_dim, h_dim, z_dim, n_layers = 2048, 256, 128, 1
    model = AccidentXai(num_classes, x_dim, h_dim, z_dim, n_layers).to(device)
    configure_trainable_params(model)
    return model


def main():
    local_rank = setup_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    register_all_modules()

    mm_cfg = MMEngineConfig.fromfile("configs/predict_anomaly_snippet.py")
    init_default_scope('mmaction')
    if not hasattr(mm_cfg, 'launcher'):
        mm_cfg.launcher = 'pytorch'
    if 'work_dir' not in mm_cfg:
        mm_cfg.work_dir = './work_dirs/xai_baseline'

    if isinstance(mm_cfg.train_dataloader, dict):
        mm_cfg.train_dataloader['num_workers'] = min(int(mm_cfg.train_dataloader.get('num_workers', 4)), 4)
        mm_cfg.train_dataloader['persistent_workers'] = False
        sampler_cfg = mm_cfg.train_dataloader.get('sampler', {})
        if isinstance(sampler_cfg, dict):
            sampler_cfg['round_up'] = True
            mm_cfg.train_dataloader['sampler'] = sampler_cfg
    if isinstance(mm_cfg.val_dataloader, dict):
        mm_cfg.val_dataloader['num_workers'] = min(int(mm_cfg.val_dataloader.get('num_workers', 4)), 4)
        mm_cfg.val_dataloader['persistent_workers'] = False
        sampler_cfg = mm_cfg.val_dataloader.get('sampler', {})
        if isinstance(sampler_cfg, dict):
            sampler_cfg['round_up'] = True
            mm_cfg.val_dataloader['sampler'] = sampler_cfg

    runner = Runner.from_cfg(mm_cfg)

    global_rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = (global_rank == 0)
    if is_main:
        os.makedirs(mm_cfg.work_dir, exist_ok=True)

    train_loader = runner.build_dataloader(mm_cfg.train_dataloader)
    val_loader = runner.build_dataloader(mm_cfg.val_dataloader)

    if dist.is_initialized():
        dist.barrier()

    model = build_model(device)
    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=DEFAULT_XAI_CFG['lr'])
    metric = AnticipationMetric(fpr_max=0.1, output_dir="outputs") if is_main else None

    best_mAUC = -1.0
    best_epoch = -1
    save_dir = "outputs"
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    max_epochs = DEFAULT_XAI_CFG['num_epochs']
    for epoch in range(1, max_epochs + 1):
        if dist.is_initialized():
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
            val_sampler = getattr(val_loader, 'sampler', None)
            if hasattr(val_sampler, 'set_epoch'):
                val_sampler.set_epoch(epoch)

        train_losses = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            metric=None,
            rank=global_rank,
        )
        if is_main:
            train_msg = " ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            print(f"[Epoch {epoch}] Train {train_msg}")

        if dist.is_initialized():
            dist.barrier()

        if is_main and metric is not None:
            if hasattr(metric, "reset"):
                metric.reset()
            elif hasattr(metric, "results"):
                metric.results.clear()
            metric.epoch = epoch

        if dist.is_initialized():
            dist.barrier()

        val_losses = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            metric=metric,
            rank=global_rank,
        )

        if dist.is_initialized():
            dist.barrier()

        if is_main and metric is not None:
            results = metric.evaluate()
            val_msg = " ".join(f"{k}: {v:.4f}" for k, v in val_losses.items())
            print(f"[Epoch {epoch}] Val losses {val_msg}")
            print(f"[Epoch {epoch}] Val results: {results}")

            mauc_key = "mAUC@" if "mAUC@" in results else ("mAUC#" if "mAUC#" in results else None)
            current_mAUC = results.get(mauc_key, 0.0) if mauc_key else 0.0
            if current_mAUC > best_mAUC:
                best_mAUC = current_mAUC
                best_epoch = epoch
                best_path = f"{save_dir}/xai_best_mAUC_epoch_{epoch:03d}_{best_mAUC:.4f}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_mAUC": best_mAUC,
                }, best_path)
                print(f"✅ [Epoch {epoch}] New best mAUC = {best_mAUC:.4f}, saved to {best_path}")

    if is_main:
        print(f"\n🎯 Training done. Best mAUC = {best_mAUC:.4f} at epoch {best_epoch}")

    cleanup_distributed()


if __name__ == '__main__':
    main()

