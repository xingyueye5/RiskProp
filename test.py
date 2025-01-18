from mmaction.datasets.transforms import SampleFrames

# 定义采样配置
sample_frames = SampleFrames(
    clip_len=1,          # 每个片段包含16帧
    frame_interval=1,     # 每隔2帧采样1帧
    num_clips=1,          # 采样1个片段
    test_mode=False       # 训练模式
)

# 模拟视频帧索引
results = {'total_frames': 50, 'start_index': 0}

# 应用采样
results = sample_frames(results)
print(results)  # 输出采样帧的索引
