import torch.optim as optim

model = BaselineModel().cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(30):
    model.train()
    for batch in train_loader:
        x, target = batch['inputs'].cuda(), batch['target'].float().cuda()
        pred = model(x)

        # 只取最后一个时间点的预测作为训练目标 (也可以平均)
        loss = criterion(pred[:, -1], target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done, loss={loss.item():.4f}")

    # 每个 epoch 用 val_loader 验证
    model.eval()
    val_results = []
    with torch.no_grad():
        for batch in val_loader:
            x, target = batch['inputs'].cuda(), batch['target']
            pred = model(x)

            for i in range(len(target)):
                val_results.append(dict(
                    pred_score=pred[i].cpu(),
                    target=target[i],
                    video_id=batch['video_id'][i],
                    frame_inds=batch['frame_inds'][i],
                    abnormal_start_frame=batch['abnormal_start_frame'][i],
                    accident_frame=batch['accident_frame'][i],
                    is_val=batch['is_val'][i],
                    is_test=batch['is_test'][i],
                    dataset=batch['dataset'][i],
                    frame_dir=batch['frame_dir'][i],
                    filename_tmpl=batch['filename_tmpl'][i],
                    type=batch['type'][i]
                ))

    from taa.metrics import AnticipationMetric
    metric = AnticipationMetric(fpr_max=0.1, output_dir="outputs")
    eval_res = metric.compute_metrics(val_results)
    print(f"Val results: {eval_res}")


torch.save(model.state_dict(), "baseline.pth")