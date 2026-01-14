#!/usr/bin/env python3
"""
AdaLEA损失函数使用示例
"""

import torch
import torch.nn as nn
from taa.models import AnticipationHead


def create_simple_model():
    """创建一个简单的模型用于演示"""
    
    class SimpleBackbone(nn.Module):
        def __init__(self, in_channels=3, out_channels=2048):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            self.conv5 = nn.Conv3d(512, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
            self.relu = nn.ReLU(inplace=True)
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            
        def forward(self, x):
            # x: [batch_size, channels, time, height, width]
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.relu(self.conv5(x))
            x = self.avg_pool(x)
            return x.squeeze(-1).squeeze(-1)  # [batch_size, channels, time]
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SimpleBackbone()
            self.head = AnticipationHead(
                num_classes=1,
                in_channels=2048,
                use_adaea=True,
                adaea_fps=30.0,
                adaea_lambda_neg=0.2,
                adaea_alpha=1.0,
                adaea_beta=0.1,
                label_with="constraint"
            )
            
        def forward(self, x, data_samples=None):
            features = self.backbone(x)
            if self.training and data_samples is not None:
                return self.head(features, data_samples)
            else:
                return self.head(features)
    
    return SimpleModel()


def create_mock_data(batch_size=4, time_steps=20, height=224, width=224):
    """创建模拟数据"""
    
    # 创建视频数据
    video_data = torch.randn(batch_size, 3, time_steps, height, width)
    
    # 创建数据样本
    class MockDataSample:
        def __init__(self, target, frame_inds, accident_frame, frame_interval):
            self.target = target
            self.frame_inds = frame_inds
            self.accident_frame = accident_frame
            self.frame_interval = frame_interval
    
    data_samples = []
    for i in range(batch_size):
        # 模拟帧索引
        frame_inds = torch.arange(0, time_steps * 5, 5)  # 每5帧一个clip
        accident_frame = 100 + i * 10  # 不同的事故帧
        frame_interval = 5
        
        data_sample = MockDataSample(
            target=True,  # 所有样本都是事故样本
            frame_inds=frame_inds,
            accident_frame=accident_frame,
            frame_interval=frame_interval
        )
        data_samples.append(data_sample)
    
    return video_data, data_samples


def train_step(model, optimizer, video_data, data_samples):
    """执行一个训练步骤"""
    
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    loss_dict = model(video_data, data_samples)
    
    # 计算总损失
    total_loss = sum(loss_dict.values())
    
    # 反向传播
    total_loss.backward()
    
    # 更新参数
    optimizer.step()
    
    return loss_dict, total_loss


def main():
    """主函数"""
    print("AdaLEA损失函数使用示例")
    print("=" * 50)
    
    # 创建模型
    model = create_simple_model()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建模拟数据
    video_data, data_samples = create_mock_data()
    print(f"输入数据形状: {video_data.shape}")
    print(f"数据样本数量: {len(data_samples)}")
    
    # 训练几个步骤
    print("\n开始训练...")
    for step in range(5):
        loss_dict, total_loss = train_step(model, optimizer, video_data, data_samples)
        
        print(f"步骤 {step + 1}:")
        for loss_name, loss_value in loss_dict.items():
            print(f"  {loss_name}: {loss_value.item():.6f}")
        print(f"  总损失: {total_loss.item():.6f}")
        print()
    
    print("训练完成！")
    
    # 测试推理
    print("\n测试推理...")
    model.eval()
    with torch.no_grad():
        predictions = model(video_data)
        print(f"预测形状: {predictions.shape}")
        print(f"预测值范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    
    print("\n示例运行完成！")


if __name__ == "__main__":
    main() 