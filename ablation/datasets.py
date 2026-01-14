from taa.datasets import MultiDataset
from taa.metrics import AnticipationMetric
import torch
from torch.utils.data import DataLoader

# 定义 train/val/test dataset
train_dataset = MultiDataset(
    cap=cap,  # 这里就是你 config 里的数据集设置
    pipeline_video=train_pipeline_video,
    pipeline_frame=train_pipeline_frame,
    modality="rgb",
    test_mode=False,
    train_with_val=False
)

val_dataset = MultiDataset(
    cap=cap,
    pipeline_video=val_pipeline_video,
    pipeline_frame=val_pipeline_frame,
    modality="rgb",
    test_mode=True,   # 验证集
    val_train=False
)

test_dataset = MultiDataset(
    cap=cap,
    pipeline_video=test_pipeline_video,
    pipeline_frame=test_pipeline_frame,
    modality="rgb",
    test_mode=True,   # 测试集
    val_train=False
)

# 构造 dataloader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8)



# # baseline 模型 forward
# preds, results = [], []
# for batch in dataloader:
#     frames, target = batch['inputs'], batch['target']
#     output = model(frames)
#     results.append(dict(
#         pred=output.detach().cpu().numpy(),
#         target=target,
#         video_id=batch['video_id'],
#         frame_inds=batch['frame_inds'],
#         accident_frame=batch['accident_frame'],
#         abnormal_start_frame=batch['abnormal_start_frame'],
#         is_val=batch['is_val'],
#         is_test=batch['is_test'],
#         dataset=batch['dataset'],
#         frame_dir=batch['frame_dir'],
#         filename_tmpl=batch['filename_tmpl'],
#         type=batch['type']
#     ))

# metric = AnticipationMetric(fpr_max=0.1, output_dir="outputs")
# eval_res = metric.compute_metrics(results)
# print(eval_res)
