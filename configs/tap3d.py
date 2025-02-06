_base_ = ["_base_/schedules/sgd_50e.py", "_base_/default_runtime.py"]

custom_imports = dict(imports="tap")

# dataset settings
dataset_type = "CapData"
data_root = "data/MM-AU/CAP-DATA"
ann_file = "cap_text_annotations.xls"

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
    dict(type="CustomSampleSnippets", snippet_len=5, num_snippets=10, test_mode=False),
    dict(type="RawFrameDecode", **file_client_args),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(type="RandomResizedCrop", area_range=(0.8, 1.0), aspect_ratio_range=(1.0, 5 / 4)),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    # dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
    # dict(type="VisualizeInputsAsVideos", output_dir="visualizations/inputs_train"),
]
val_pipeline = [
    dict(type="CustomSampleSnippets", snippet_len=5, num_snippets=None, test_mode=True),
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
        filename_tmpl="{:06}.jpg",
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
        filename_tmpl="{:06}.jpg",
        pipeline=val_pipeline,
        test_mode=True,
        # indices=list(range(20)),
    ),
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type="SnippetMetric",
        thresholds=[x * 0.1 for x in range(1, 10)],
        test_mode=False,
    ),
    dict(
        type="SnippetMetric",
        thresholds=[x * 0.1 for x in range(1, 10)],
        vis_list={
            "001537": 1,
            "004251": 1,
            # "006517": 1,
            # "006713": 1,
            # "006986": 1,
            # "006993": 1,
            # "006994": 1,
            # "006998": 1,
            # "007460": 1,
            # "009733": 1,
            # "010261": 1,
            # "010361": 1,
            # "011211": 1,
            # "011215": 1,
            # "011309": 1,
            # "013336": 1,
            "005968": 2,
            "009677": 2,
            # "011083": 2,
            "005184": 3,
            "005721": 3,
            # "006438": 3,
            # "009725": 3,
            # "010135": 3,
            # "010741": 3,
            "004619": 4,
            "006078": 4,
            # "006459": 4,
            # "010093": 4,
            "001101": 5,
            "003483": 5,
            # "004244": 5,
            # "004695": 5,
            # "004801": 5,
            # "005321": 5,
            # "005546": 5,
            # "006382": 5,
            # "006393": 5,
            # "006450": 5,
            "001366": 6,
            "002074": 6,
            # "003821": 6,
            # "004745": 6,
            # "005438": 6,
            # "005781": 6,
            # "006432": 6,
            # "006502": 6,
            # "007211": 6,
            # "009126": 6,
            "001604": 7,
            "006577": 7,
            # "007492": 7,
            # "008219": 7,
            # "009256": 7,
            # "009689": 7,
            # "010066": 7,
            # "011072": 7,
            # "011398": 7,
            # "013001": 7,
            "000966": 8,
            "003171": 8,
            # "003228": 8,
            # "003535": 8,
            # "003686": 8,
            # "005097": 8,
            # "005170": 8,
            # "005365": 8,
            # "005779": 8,
            # "005856": 8,
            "001752": 9,
            "003119": 9,
            # "003335": 9,
            # "003903": 9,
            # "008208": 9,
            # "009444": 9,
            # "010112": 9,
            # "010554": 9,
            # "013409": 9,
            "000001": 10,
            "000859": 10,
            # "000888": 10,
            # "000922": 10,
            # "000969": 10,
            # "001078": 10,
            # "001130": 10,
            # "001152": 10,
            # "001160": 10,
            # "001186": 10,
            "006390": 11,
            "006403": 11,
            # "006413": 11,
            # "006439": 11,
            # "006453": 11,
            # "006473": 11,
            # "006492": 11,
            # "006505": 11,
            # "006520": 11,
            # "006603": 11,
            "001017": 12,
            "001040": 12,
            # "001171": 12,
            # "001197": 12,
            # "001360": 12,
            # "001447": 12,
            # "001483": 12,
            # "001547": 12,
            # "001632": 12,
            # "001656": 12,
            "002182": 13,
            "002363": 13,
            # "002372": 13,
            # "002445": 13,
            # "002596": 13,
            # "002620": 13,
            # "002700": 13,
            # "002728": 13,
            # "004494": 13,
            # "004502": 13,
            "001676": 14,
            "001772": 14,
            # "001974": 14,
            # "002103": 14,
            # "002527": 14,
            # "002819": 14,
            # "003121": 14,
            # "003706": 14,
            # "004114": 14,
            # "004662": 14,
            "001293": 15,
            "001929": 15,
            # "002690": 15,
            # "002936": 15,
            # "004577": 15,
            # "006012": 15,
            # "007447": 15,
            # "009318": 15,
            # "009545": 15,
            # "009634": 15,
            "001931": 16,
            "007024": 16,
            # "009915": 16,
            # "010326": 16,
            # "010426": 16,
            # "011298": 16,
            "009645": 17,
            "010302": 17,
            # "011616": 17,
            # "011710": 17,
            "004131": 18,
            "009831": 18,
            # "010891": 18,
            # "010925": 18,
        },
        output_dir="visualizations/outputs_test",
    ),
]
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_begin=1, val_interval=1)

# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=50,
#         by_epoch=True,
#         milestones=[5, 10, 15, 20],
#         gamma=0.1)
# ]

# 每轮都保存权重，并且只保留最新的权重
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=1, save_best="f_pre@max", rule="greater")
)
custom_hooks = [dict(type="EpochHook"), dict(type="MetricHook")]

model = dict(
    type="Recognizer3D",
    # backbone=dict(type="ResNet3d", depth=50, pretrained=None, with_pool2=False),
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
        type="SnippetClsHead",
        num_classes=1,
        in_channels=2048,
        spatial_type="avg",
        loss_cls=dict(type="BCELossWithLogits"),
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], format_shape="NCTHW"
    ),
    train_cfg=None,
    test_cfg=None,
)

load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb_20230612-12ce977c.pth"
# load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
# load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth"
# find_unused_parameters = True
