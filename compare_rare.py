# -*- coding: utf-8 -*-
"""
compare_rare.py — 分布式训练 RARE-ICIP 模型，数据管线与指标复用 mmaction/mmengine（同 compare_dis.py）
- 规范 NCCL 环境、避免 watchdog 超时
- 复用 compare_dis.py 的 mmengine Runner / AnticipationMetric 数据管线
- 模型与损失继承 ablation/RARE-ICIP2025/main_taa_ddp.py 的实现
"""

import os
import sys
import pathlib
import contextlib
from typing import Dict, Any, List

import torch
import numpy as np


def _normalize_env_pre_import():
    """在 import 前设置常用的 NCCL 环境变量，规避常见的超时/死锁。"""
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')

    lr_raw = os.environ.get('LOCAL_RANK', '0')
    try:
        lr = int(lr_raw)
    except Exception:
        lr = 0
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

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent

# 先导入原项目中的 AnticipationMetric，之后再切换到 RARE 的 taa 包
cap_metrics_mod = importlib.import_module('taa.metrics')
AnticipationMetric = cap_metrics_mod.AnticipationMetric

# 移除原 taa 包，避免与 RARE 的同名包冲突
for _name in list(sys.modules.keys()):
    if _name == 'taa' or _name.startswith('taa.'):
        sys.modules.pop(_name)

RARE_ROOT = PROJECT_ROOT / 'ablation' / 'RARE-ICIP2025'
if str(RARE_ROOT) not in sys.path:
    sys.path.insert(0, str(RARE_ROOT))

from taa.configs.config import Config as RareConfig  # type: ignore  # noqa: E402
from taa.model.taa_yolov10 import YOLOv10TAADetectionModel, TAATemporal, AdaLEA  # type: ignore  # noqa: E402
try:  # type: ignore  # noqa: E402
    from taa.model.taa_yolov10_neuflow import YOLOv10TAANeuFlowDetectionModel  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    YOLOv10TAANeuFlowDetectionModel = None  # type: ignore


def _get_sample_attr(ds, key, default=None):
    if hasattr(ds, key) and getattr(ds, key) is not None:
        return getattr(ds, key)
    if hasattr(ds, 'metainfo') and ds.metainfo is not None and key in ds.metainfo:
        return ds.metainfo[key]
    if hasattr(ds, 'algorithms') and ds.algorithms is not None and key in ds.algorithms:
        return ds.algorithms[key]
    return default


def _ensure_batch_first(x: torch.Tensor, expected_batch: int) -> torch.Tensor:
    """尝试把 tensor 调整成 [B, T, C, H, W] 格式。"""
    if x.size(0) == expected_batch:
        return x
    if x.size(1) == expected_batch:
        return x.permute(1, 0, 2, 3, 4).contiguous()
    raise RuntimeError(
        f"Unexpected input shape {tuple(x.shape)} (expected batch {expected_batch})."
    )


def prepare_batch(data_batch: Dict[str, Any], device: torch.device, criterion: torch.nn.Module):
    inputs = data_batch['inputs']
    if isinstance(inputs, (list, tuple)):
        x = torch.cat(inputs, dim=0)
    else:
        x = inputs
    x = x.permute(2, 0, 1, 3, 4).contiguous()  # 先对齐 compare_dis.py 的写法
    data_samples: List[Any] = data_batch['data_samples']
    batch_size = len(data_samples)
    x = _ensure_batch_first(x, batch_size)
    frames = x.float().to(device) / 255.0

    is_positive, toa_list, annotations = [], [], []
    for idx, ds in enumerate(data_samples):
        target = bool(_get_sample_attr(ds, 'target', False))
        fps = _get_sample_attr(ds, 'fps', 30) or 30
        accident_frame = _get_sample_attr(ds, 'accident_frame', None)
        start_index = _get_sample_attr(ds, 'start_index', 0)
        if target and accident_frame is not None:
            toa_val = max(int(accident_frame) - int(start_index), 0)
        else:
            toa_val = frames.size(1) * fps
        is_positive.append(1 if target else 0)
        toa_list.append(float(toa_val))
        annotations.append([])  # 若无标注则保持空列表

    batch_info = {
        'frames': frames,
        'is_positive': torch.tensor(is_positive, device=device, dtype=torch.long),
        'toa': torch.tensor(toa_list, device=device, dtype=torch.float32),
        'annotations': annotations,
    }
    if hasattr(criterion, 'last_mtta'):
        batch_info['prev_mtta'] = getattr(criterion, 'last_mtta', 0.0)
    return frames, batch_info


@torch.no_grad()
def _eval_step_to_metric(preds_np: np.ndarray, data_batch: Dict[str, Any], metric: AnticipationMetric, T: int):
    B = preds_np.shape[0]
    data_samples = data_batch['data_samples']
    for i in range(B):
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


def run_epoch(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    criterion: torch.nn.Module,
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
        frames, batch_info = prepare_batch(data_batch, device, criterion)
        with ctx:
            predictions = model(frames)
        losses = criterion(predictions, batch_info)
        loss_total = losses['total_loss']
        if train_mode:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        batches += 1

        if metric is not None and rank == 0:
            risk_scores = predictions['risk_score']
            probs = torch.stack(risk_scores, dim=1).softmax(dim=-1)[..., 1].detach().cpu().numpy()
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


def build_model(cfg: Dict[str, Any], device: torch.device):
    model_cfg = cfg['model']
    opticalflow = model_cfg.get('opticalflow', 'None')
    if opticalflow == 'neuflow' and YOLOv10TAANeuFlowDetectionModel is not None:
        model = YOLOv10TAANeuFlowDetectionModel(
            yolo_id=model_cfg['detector'],
            yolo_ft=model_cfg['yolo_ft'],
            conf_thresh=model_cfg.get('conf_thresh', 0.1),
        )
    else:
        model = YOLOv10TAADetectionModel(
            yolo_id=model_cfg['detector'],
            yolo_ft=model_cfg['yolo_ft'],
            conf_thresh=model_cfg.get('conf_thresh', 0.1),
        )
    return model.to(device)


def build_criterion(cfg: Dict[str, Any], device: torch.device):
    loss_cfg = cfg['loss']
    if loss_cfg['name'] == 'AdaLEA':
        criterion = AdaLEA(
            lambda_temporal=loss_cfg['lambda_temporal'],
            lambda_attn=loss_cfg['lambda_attn'],
            fps=cfg['dataset']['fps'],
            iou_thresh=loss_cfg['iou_thresh'],
            gamma=loss_cfg.get('gamma', 5.0),
        )
    else:
        criterion = TAATemporal(
            lambda_temporal=loss_cfg['lambda_temporal'],
            lambda_attn=loss_cfg['lambda_attn'],
            fps=cfg['dataset']['fps'],
            iou_thresh=loss_cfg['iou_thresh'],
        )
    return criterion.to(device)


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]):
    trainable = [
        p for n, p in model.named_parameters()
        if p.requires_grad and 'detector' not in n and 'flow_model' not in n
    ]
    opt_cfg = cfg['train']
    lr = float(opt_cfg['learning_rate'])
    opt_type = opt_cfg.get('optimizer', 'Adam')
    if opt_type == 'Adam':
        return optim.Adam(trainable, lr=lr)
    if opt_type == 'AdamW':
        return optim.AdamW(trainable, lr=lr)
    if opt_type == 'SGD':
        return optim.SGD(trainable, lr=lr)
    raise ValueError(f"Unsupported optimizer {opt_type}")


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
        mm_cfg.work_dir = './work_dirs/rare_baseline'

    # dataloader 设置与 compare_dis.py 一致，限制并发并保持 round_up
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

    rare_cfg_path = RARE_ROOT / 'taa' / 'configs' / 'taa_DAD_yolov10.yaml'
    rare_cfg = RareConfig.load_config(str(rare_cfg_path))

    model = build_model(rare_cfg, device)
    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    criterion = build_criterion(rare_cfg, device)
    optimizer = build_optimizer(model, rare_cfg)
    metric = AnticipationMetric(fpr_max=0.1, output_dir="outputs") if is_main else None

    max_epochs = int(rare_cfg['train'].get('num_epochs', 30))
    best_mAUC = -1.0
    best_epoch = -1
    save_dir = "outputs"
    if is_main:
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        if dist.is_initialized():
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
            val_sampler = getattr(val_loader, 'sampler', None)
            if hasattr(val_sampler, 'set_epoch'):
                val_sampler.set_epoch(epoch)

        train_losses = run_epoch(model, train_loader, device, criterion, optimizer=optimizer, metric=None, rank=global_rank)
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

        val_losses = run_epoch(model, val_loader, device, criterion, optimizer=None, metric=metric, rank=global_rank)

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
                best_path = f"{save_dir}/rare_best_mAUC_epoch_{epoch:03d}_{best_mAUC:.4f}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_mAUC": best_mAUC,
                }, best_path)
                print(f"✅ [Epoch {epoch}] New best mAUC = {best_mAUC:.4f}, saved to {best_path}")

            criterion.last_mtta = results.get('mtta_2s@0.1', results.get('mtta_2s#0.1', 0.0))

    if is_main:
        print(f"\n🎯 Training done. Best mAUC = {best_mAUC:.4f} at epoch {best_epoch}")

    cleanup_distributed()


if __name__ == '__main__':
    main()

