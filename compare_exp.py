import os
import torch
import torch.optim as optim
from taa.metrics import AnticipationMetric
from ablation.CAP.src.model import accident  # CAP 模型
from mmengine.runner import Runner
from ablation.CAP.src.bert import opt 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def one_hot_from_bool(target_bool_tensor, num_classes=2):
    t = target_bool_tensor.long().clamp(min=0, max=1)
    y = torch.zeros(t.size(0), num_classes, device=t.device, dtype=torch.float32)
    y.scatter_(1, t.view(-1, 1), 1.0)
    return y


def extract_batch_fields(data_batch):
    """ 从 MMAction dataloader 批次中提取 CAP 模型需要的输入 """
    x = data_batch['inputs']                  # [B, C, T, H, W]
    data_samples = data_batch['data_samples'] # list[ActionDataSample]

    x_cap = x.permute(0, 2, 1, 3, 4).contiguous().to(device)  # [B, T, C, H, W]
    B = x_cap.size(0)
    targets, toa_frames = [], []

    for i in range(B):
        ds = data_samples[i]

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
    texts = [""] * B  # 无文本时占位
    return x_cap, y_onehot, toa, texts


def run_epoch(cap_model, loader, optimizer=None, metric=None):
    train_mode = optimizer is not None
    cap_model.train(train_mode)
    total_loss, n = 0.0, 0

    for data_batch in loader:
        x_cap, y, toa, texts = extract_batch_fields(data_batch)
        losses, outputs = cap_model(x_cap, y, toa, texts)
        loss = losses['total_loss']

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.detach().cpu().item()
        n += 1

        # 评估阶段
        if metric is not None:
            preds = torch.stack(outputs, dim=1).softmax(dim=-1)[..., 1].detach().cpu().numpy()  # [B,T]
            B, T = preds.shape

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
                    pred_score=torch.from_numpy(preds[i]),
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

    return total_loss / max(1, n)


def main():
    # --- 1️⃣ 直接构建 dataloaders（不依赖 cfg） ---
    from mmaction.utils import register_all_modules
    register_all_modules()
    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    # 手动加载你之前的 config，用于复用 dataloader 配置
    cfg = Config.fromfile("configs/predict_anomaly_snippet.py")  # 改成你实际的 cfg 路径
    init_default_scope('mmaction')

    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    train_loader = runner.build_dataloader(cfg.train_dataloader)
    val_loader = runner.build_dataloader(cfg.val_dataloader)

    # --- 2️⃣ 初始化 CAP 模型 ---
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

    optimizer = optim.AdamW(cap_model.parameters(), lr=1e-4, weight_decay=1e-4)
    metric = AnticipationMetric(fpr_max=0.1, output_dir="outputs")

    # --- 3️⃣ 训练与评估 ---
    max_epochs = 50
    best_mAUC = -1.0
    best_epoch = -1
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, max_epochs + 1):
        cap_model.train()
        train_loss = run_epoch(cap_model, train_loader, optimizer)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        metric.reset()
        metric.epoch = epoch
        with torch.no_grad():
            val_loss = run_epoch(cap_model, val_loader, optimizer=None, metric=metric)
        results = metric.evaluate()
        print(f"[Epoch {epoch}] Val Results: {results}")

        # === ✅ 记录 & 保存 best mAUC ===
        mauc_key = "mAUC@" if "mAUC@" in results else "mAUC#"
        current_mAUC = results.get(mauc_key, 0.0)

        if current_mAUC > best_mAUC:
            best_mAUC = current_mAUC
            best_epoch = epoch
            best_path = f"{save_dir}/best_mAUC_epoch_{epoch:03d}_{best_mAUC:.4f}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": cap_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mAUC": best_mAUC
            }, best_path)
            print(f"✅ [Epoch {epoch}] New best mAUC = {best_mAUC:.4f}, model saved to {best_path}")

    print(f"\n🎯 Training done. Best mAUC = {best_mAUC:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()
