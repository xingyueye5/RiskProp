# -*- coding: utf-8 -*-
"""
compare_dis.py — 分布式训练稳健版
- 不再动态改写 CUDA_VISIBLE_DEVICES，避免设备映射紊乱
- 在 import 前设置常见 NCCL 环境变量，降低超时/死锁概率
- 分布式初始化后按 local_rank 设定当前 device
- monkey‑patch mmengine.logger 的设备获取为当前 CUDA 设备（不依赖 CVD）
  单机4卡：
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 compare_dis.py
  单卡：
    CUDA_VISIBLE_DEVICES=0 python compare_dis.py
"""

# ========= ① 在 import mmengine/mmaction 之前，先规范化环境 =========
import os
import sys
import torch
import numpy as np

def _normalize_env_pre_import():
    # 常见 NCCL 稳定性环境变量（如未显式设置则给默认值）
    os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
    # 单机多卡常见问题：禁用 IB/P2P 可规避部分环境不兼容导致的超时（如需要可手动覆盖为 0）
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')

    # LOCAL_RANK 清洗（不基于它去扩展/改写 CVD）
    lr_raw = os.environ.get('LOCAL_RANK', '0')
    try:
        lr = int(lr_raw)
    except Exception:
        lr = 0
        os.environ['LOCAL_RANK'] = '0'

    # 清洗 CVD：仅去空格/空项，不做“复制扩展”等破坏性改写
    cvd_raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    parts = [p.strip() for p in cvd_raw.split(',') if p.strip() != '']
    if parts:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(parts)

    # 诊断打印（不触碰实际 CUDA 设备枚举）
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

# ========= ② 分布式初始化与再次兜底 =========
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """初始化分布式（torchrun 注入时启用），返回 local_rank。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # env:// 从 torchrun 注入的环境变量获取初始化信息
        # 使用较新的 NCCL watchdog 策略由上面的环境变量控制
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(
            f"[DDP] initialized rank={dist.get_rank()} / {dist.get_world_size()} (local_rank={local_rank})",
            file=sys.stderr,
        )
        return local_rank
    else:
        print("[Single GPU] non-distributed mode.", file=sys.stderr)
        return 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ========= ③ 现在 import mmengine.logger 并 monkey-patch _get_device_id =========
import importlib
mm_logger_mod = importlib.import_module('mmengine.logging.logger')

def _safe_get_device_id():
    """返回当前 CUDA 设备编号；CPU 时返回 0。不修改任何环境变量。"""
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.current_device())
        except Exception:
            return 0
    return 0

# 仅替换设备获取逻辑，避免因解析 CVD 越界而报错/改写环境
mm_logger_mod._get_device_id = _safe_get_device_id  # type: ignore

# ========= ④ 其余依赖再导入（Runner 等） =========
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmaction.utils import register_all_modules

import torch.optim as optim

# 你的工程依赖
from taa.metrics import AnticipationMetric
from ablation.CAP.src.model import accident  # CAP 模型
from ablation.CAP.src.bert import opt



# ========= ⑤ 训练/评估逻辑 =========
def one_hot_from_bool(target_bool_tensor, num_classes=2):
    t = target_bool_tensor.long().clamp(min=0, max=1)
    y = torch.zeros(t.size(0), num_classes, device=t.device, dtype=torch.float32)
    y.scatter_(1, t.view(-1, 1), 1.0)
    return y

def extract_batch_fields(data_batch, device):
    x = data_batch['inputs']                  # [B, C, T, H, W]
    data_samples = data_batch['data_samples'] # list[ActionDataSample]
    x = torch.cat(x, dim=0)
    x_cap = x.permute(2, 0, 1, 3, 4).contiguous().to(device)  # [B, T, C, H, W] =[1,60,3,224,224]
    x_cap = x_cap.float() / 255.0
    B = x_cap.size(0)
    targets, toa_frames = [], []

    ds=data_samples[0]
    def get_k(k, default=None):
        if hasattr(ds, k) and getattr(ds, k) is not None:
            return getattr(ds, k)
        if hasattr(ds, 'metainfo') and k in ds.metainfo:
            return ds.metainfo[k]
        if hasattr(ds, 'algorithms') and k in ds.algorithms:
            return ds.algorithms[k]
        return default


    target = int(bool(get_k('target', False)))
    fps = get_k('fps', 30) or 30
    accident_frame = get_k('accident_frame', None)
    start_index = get_k('start_index', 0)
    T = x_cap.size(1)
    toa = T * fps if accident_frame is None else max(int(accident_frame) - int(start_index), 0)
    targets.append(target)
    toa_frames.append(float(toa))

    y_onehot = one_hot_from_bool(torch.tensor(targets, device=device))
    toa = torch.tensor(toa_frames, device=device, dtype=torch.float32)
    texts = [""] * B
    return x_cap, y_onehot, toa, texts

@torch.no_grad()
def _eval_step_to_metric(preds_np, data_batch, metric, T):
    B = preds_np.shape[0]
    data_samples = []
    for i in range(B):
        ds = data_batch['data_samples'][i]
        def get_k(k, default=None):
            if hasattr(ds, k) and getattr(ds, k) is not None:
                return getattr(ds, k)
            if hasattr(ds, 'metainfo') and k in ds.metainfo:
                return ds.metainfo[k]
            if hasattr(ds, 'algorithms') and k in ds.algorithms:
                return ds.algorithms[k]
            return default
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
        data_samples.append(result)
    metric.process(None, data_samples)

def run_epoch(cap_model, loader, device, optimizer=None, metric=None, rank=0):
    train_mode = optimizer is not None
    cap_model.train(train_mode)
    total_loss, n = 0.0, 0
    for data_batch in loader:
        x_cap, y, toa, texts = extract_batch_fields(data_batch, device)
        losses, outputs = cap_model(x_cap, y, toa, texts)
        loss = losses['total_loss']

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        n += 1

        if (metric is not None) and (rank == 0):
            preds = torch.stack(outputs, dim=1).softmax(dim=-1)[..., 1].detach().cpu().numpy()
            _, T = preds.shape
            _eval_step_to_metric(preds, data_batch, metric, T)

    avg_loss = torch.tensor([total_loss / max(1, n)], device=device)
    if dist.is_initialized():
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss /= dist.get_world_size()
    return float(avg_loss.item())

# ========= ⑥ 主函数 =========
def main():
    local_rank = setup_distributed()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    from mmaction.utils import register_all_modules
    register_all_modules()

    cfg = Config.fromfile("configs/predict_anomaly_snippet.py")
    init_default_scope('mmaction')
    if not hasattr(cfg, 'launcher'):
        cfg.launcher = 'pytorch'
    if 'work_dir' not in cfg:
        cfg.work_dir = './work_dirs/cap_baseline'

    # 进一步提高稳定性：限制 DataLoader 并发和关闭持久化 worker（常见死锁源）
    if hasattr(cfg, 'train_dataloader') and isinstance(cfg.train_dataloader, dict):
        cfg.train_dataloader['num_workers'] = min(int(cfg.train_dataloader.get('num_workers', 4)), 4)
        cfg.train_dataloader['persistent_workers'] = False
        # 保障各 rank 步数一致
        if isinstance(cfg.train_dataloader.get('sampler', {}), dict):
            cfg.train_dataloader['sampler']['round_up'] = True
    if hasattr(cfg, 'val_dataloader') and isinstance(cfg.val_dataloader, dict):
        cfg.val_dataloader['num_workers'] = min(int(cfg.val_dataloader.get('num_workers', 4)), 4)
        cfg.val_dataloader['persistent_workers'] = False
        if isinstance(cfg.val_dataloader.get('sampler', {}), dict):
            cfg.val_dataloader['sampler']['round_up'] = True

    # 关键：现在才创建 Runner（logger 已被 monkey-patch）
    runner = Runner.from_cfg(cfg)

    global_rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = (global_rank == 0)
    if is_main:
        os.makedirs(cfg.work_dir, exist_ok=True)

    train_loader = runner.build_dataloader(cfg.train_dataloader)
    val_loader = runner.build_dataloader(cfg.val_dataloader)

    # 保证所有 rank 在进入训练循环前同步，避免某些 rank 先进入反向导致的等待
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
        num_class=opt.num_class
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

        train_loss = run_epoch(cap_model, train_loader, device, optimizer=optimizer, metric=None, rank=global_rank)
        if is_main:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

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

        val_metric = metric if is_main else None
        with torch.no_grad():
            _ = run_epoch(cap_model, val_loader, device, optimizer=None, metric=val_metric, rank=global_rank)

        if dist.is_initialized():
            dist.barrier()

        if is_main and metric is not None:
            raw_results = metric.compute_metrics(metric.results)
            if metric.prefix:
                results = {f"{metric.prefix}/{k}": v for k, v in raw_results.items()}
            else:
                results = raw_results
            metric.results.clear()
            print(f"[Epoch {epoch}] Val Results: {results}")

            mauc_key = "mAUC@" if "mAUC@" in results else ("mAUC#" if "mAUC#" in results else None)
            current_mAUC = results.get(mauc_key, 0.0) if mauc_key else 0.0
            if current_mAUC > best_mAUC:
                best_mAUC = current_mAUC
                best_epoch = epoch
                best_path = f"{save_dir}/best_mAUC_epoch_{epoch:03d}_{best_mAUC:.4f}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": cap_model.module.state_dict() if hasattr(cap_model, 'module') else cap_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_mAUC": best_mAUC
                }, best_path)
                print(f"✅ [Epoch {epoch}] New best mAUC = {best_mAUC:.4f}, saved to {best_path}")

    if is_main:
        print(f"\n🎯 Training done. Best mAUC = {best_mAUC:.4f} at epoch {best_epoch}")

    cleanup_distributed()

if __name__ == '__main__':
    main()
