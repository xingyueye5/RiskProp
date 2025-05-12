_base_ = ["_base_/schedules/sgd_50e.py", "_base_/default_runtime.py"]

custom_imports = dict(imports="taa")

# dataset settings
dataset_type = "CapData"
data_root = "data/MM-AU/CAP-DATA"
ann_file = "cap_text_annotations.xls"
filename_tmpl = "{:06}.jpg"
vis_list = {
    "001537": 1,
    "004251": 1,
    "005968": 2,
    "009677": 2,
    "005184": 3,
    "005721": 3,
    "004619": 4,
    "006078": 4,
    "001101": 5,
    "003483": 5,
    "001366": 6,
    "002074": 6,
    "001604": 7,
    "006577": 7,
    "000966": 8,
    "003171": 8,
    "001752": 9,
    "003119": 9,
    "000001": 10,
    "000859": 10,
    "006390": 11,
    "006403": 11,
    "001017": 12,
    "001040": 12,
    "002182": 13,
    "002363": 13,
    "001676": 14,
    "001772": 14,
    "001293": 15,
    "001929": 15,
    "001931": 16,
    "007024": 16,
    "009645": 17,
    "010302": 17,
    "004131": 18,
    "009831": 18,
}

# dataset_type = "DadaData"
# data_root = "data/MM-AU/DADA-DATA"
# ann_file = "dada_text_annotations.xlsx"
# filename_tmpl = "{:04}.png"
# vis_list = {
#     "1_001": 1,
#     "1_004": 1,
#     "2_002": 2,
#     "2_004": 2,
#     "3_003": 3,
#     "3_007": 3,
#     "4_003": 4,
#     "4_005": 4,
#     "5_008": 5,
#     "5_009": 5,
#     "6_007": 6,
#     "6_017": 6,
#     "7_002": 7,
#     "7_005": 7,
#     "8_005": 8,
#     "8_006": 8,
#     "9_006": 9,
#     "9_008": 9,
#     "10_002": 10,
#     "10_010": 10,
#     "11_001": 11,
#     "11_010": 11,
#     "12_004": 12,
#     "12_005": 12,
#     "13_004": 13,
#     "13_009": 13,
#     "14_002": 14,
#     "14_004": 14,
#     "15_003": 15,
#     "18_002": 18,
#     "18_008": 18,
# }


algorithm_keys = (
    "frame_dir",
    "filename_tmpl",
    "img_shape",
    "sample_idx",
    "video_id",
    "type",
    "label",
    "start_index",
    "total_frames",
    "abnormal_start_frame",
    "abnormal_end_frame",
    "accident_frame",
    "frame_inds",
    "clip_len",
    "num_clips",
    "frame_interval",
    "fps",
    "is_test",
)

file_client_args = dict(io_backend="disk")

train_pipeline = [
    dict(type="SampleSnippetsBeforeAccident", snippet_len=5, num_snippets=10, test_mode=False),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="RandomResizedCrop", area_range=(0.8, 1.0), aspect_ratio_range=(4 / 3, 16 / 9)),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
    # dict(type="VisualizeInputsAsVideos", output_dir="visualizations/inputs_train"),
]
val_pipeline = [
    dict(type="SampleSnippetsBeforeAccident", snippet_len=5, num_snippets=None, test_mode=True),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    # dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=3,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        filename_tmpl=filename_tmpl,
        pipeline=train_pipeline,
        test_mode=False,
        # indices=list(range(20)),
    ),
)
val_dataloader = dict(
    batch_size=1,  # Batch size can only be 1 for testing
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        filename_tmpl=filename_tmpl,
        pipeline=val_pipeline,
        test_mode=True,
        # indices=list(range(20)),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="NewAnticipationMetric",
    thresholds=[x * 0.1 for x in range(1, 10)],
    # vis_list=vis_list,
    output_dir="visualizations",
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_begin=1, val_interval=1)

# 每轮都保存权重，并且只保留最新的权重
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=50, save_best="mAUC@", rule="greater")
)
custom_hooks = [dict(type="EpochHook"), dict(type="NewAnticipationMetricHook")]

model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="ResNet3dSlowOnly",
        depth=50,
        pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False,
    ),
    cls_head=dict(
        type="AnomalyHeadFromSnippets",
        num_classes=1,
        loss_cls=dict(type="BCELossWithLogits"),
        pos_weight=10,
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], format_shape="NCTHW"
    ),
    train_cfg=None,
    test_cfg=None,
)

load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb_20230612-12ce977c.pth"
