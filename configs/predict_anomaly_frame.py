_base_ = ["_base_/schedules/sgd_50e.py", "_base_/default_runtime.py"]

custom_imports = dict(imports="taa")

# dataset settings
cap = dict(data_root="data/MM-AU/CAP-DATA", ann_file="cap_text_annotations.xls", filename_tmpl="{:06}.jpg", start_index=1)
dada = dict(data_root="data/MM-AU/DADA-DATA", ann_file="dada_text_annotations.xlsx", filename_tmpl="{:04}.png", start_index=1)
d2city = dict(data_root="data/D_square-City", ann_file="annotations.csv")
nexar = dict(data_root="data/nexar-collision-prediction", ann_file="annotations.csv", filename_tmpl="{:06}.jpg", start_index=0)
clip_len = 1
num_clips = 30
modality = "rgb"
assert modality in ["rgb", "flow"], f"modality {modality} is not supported"
vis_list = []

algorithm_keys = (
    "dataset",
    "frame_dir",
    "filename_tmpl",
    "img_shape",
    "sample_idx",
    "video_id",
    "type",
    "start_index",
    "total_frames",
    "target",
    "abnormal_start_frame",
    "accident_frame",
    "frame_inds",
    "clip_len",
    "num_clips",
    "frame_interval",
    "fps",
    "is_val",
    "is_test",
)

file_client_args = dict(io_backend="disk")

train_pipeline_video = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFramesBeforeAccident", clip_len=clip_len, num_clips=num_clips, test_mode=False),
    dict(type="DecordDecode"),
    dict(type="RandomResizedCrop", area_range=(0.8, 1.0), aspect_ratio_range=(4 / 3, 16 / 9)),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Flow", modality=modality),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]
val_pipeline_video = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFramesBeforeAccident", clip_len=clip_len, num_clips=num_clips, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flow", modality=modality),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]
test_pipeline_video = val_pipeline_video

train_pipeline_frame = [
    dict(type="SampleFramesBeforeAccident", clip_len=clip_len, num_clips=num_clips, test_mode=False),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="RandomResizedCrop", area_range=(0.8, 1.0), aspect_ratio_range=(4 / 3, 16 / 9)),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Flow", modality=modality),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]
val_pipeline_frame = [
    dict(type="SampleFramesBeforeAccident", clip_len=clip_len, num_clips=num_clips, test_mode=True),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flow", modality=modality),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]
test_pipeline_frame = val_pipeline_frame

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="MultiDataset",
        # cap=cap,
        # dada=dada,
        # d2city=d2city,
        nexar=nexar,
        pipeline_video=train_pipeline_video,
        pipeline_frame=train_pipeline_frame,
        modality=modality,
        test_mode=False,
        train_with_val=False,
        # indices=list(range(20)),
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="MultiDataset",
        # cap=cap,
        # dada=dada,
        # d2city=d2city,
        nexar=nexar,
        pipeline_video=val_pipeline_video,
        pipeline_frame=val_pipeline_frame,
        modality=modality,
        test_mode=True,
        val_train=False,
        # indices=list(range(20)),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="AnticipationMetric", fpr_max=0.1, vis_list=vis_list, output_dir="visualizations")
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_begin=1, val_interval=1)

# 每轮都保存权重，并且只保留最新的权重
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=50, save_best="mAUC@", rule="greater"))
custom_hooks = [dict(type="EpochHook"), dict(type="AnticipationMetricHook")]

model = dict(
    type="Recognizer2D",
    backbone=dict(type="ResNet", pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth", depth=50, norm_eval=False),
    cls_head=dict(
        type="AnticipationHead",
        pos_weight=1,
        clip_len=clip_len,
        num_clips=num_clips,
        two_stream=modality in ["both", "two_stream"],
        with_rnn=True,
        with_decoder=False,
        label_with="annotation",
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], format_shape="NCHW"
    ),
    train_cfg=None,
    test_cfg=None,
)

load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth"
