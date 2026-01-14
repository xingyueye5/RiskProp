# AdaLEA损失函数实现

## 概述

AdaLEA (Adaptive Early Anticipation) 损失函数是基于论文 "Anticipating Traffic Accidents with Adaptive Loss and Large-scale Incident DB" 实现的自适应早期待测损失函数。该损失函数通过指数惩罚机制来鼓励模型更早地预测交通事故。

## 核心思想

AdaLEA损失函数的主要特点：

1. **指数惩罚机制**: 通过指数函数对距离事故时间较远的预测给予更大的惩罚
2. **自适应权重**: 根据时间到事故的距离动态调整损失权重
3. **时序一致性**: 确保预测概率在时间上具有单调性
4. **正负样本平衡**: 通过λ参数平衡正样本和负样本的损失贡献

## 数学公式

AdaLEA损失函数的核心公式：

```
L_AdaLEA = α * L_pos + λ * L_neg + β * L_consistency

其中：
- L_pos = -exp(penalty) * log(p_pred)
- L_neg = -log(1 - p_pred)
- penalty = -max(0, (toa - time - 1) / fps)
- L_consistency = max(0, p_t - p_{t+1})
```

## 参数说明

在 `AnticipationHead` 中，AdaLEA损失函数支持以下参数：

- `use_adaea` (bool): 是否使用AdaLEA损失函数，默认为True
- `adaea_fps` (float): 帧率，用于计算时间惩罚，默认为30.0
- `adaea_lambda_neg` (float): 负样本损失权重，默认为0.2
- `adaea_alpha` (float): 正样本损失权重，默认为1.0
- `adaea_beta` (float): 时序一致性损失权重，默认为0.1

## 使用方法

### 1. 基本配置

在配置文件中使用AdaLEA损失函数：

```python
model = dict(
    type="Recognizer3D",
    cls_head=dict(
        type="AnticipationHead",
        num_classes=1,
        # AdaLEA损失函数参数
        use_adaea=True,
        adaea_fps=30.0,
        adaea_lambda_neg=0.2,
        adaea_alpha=1.0,
        adaea_beta=0.1,
        label_with="constraint",
    ),
)
```

### 2. 训练命令

```bash
python tools/train.py configs/predict_occurrence_snippet_adaea.py
```

### 3. 参数调优建议

- **adaea_fps**: 根据实际视频帧率调整，通常为30fps
- **adaea_lambda_neg**: 控制负样本损失权重，建议范围[0.1, 0.5]
- **adaea_alpha**: 控制正样本损失权重，通常设为1.0
- **adaea_beta**: 控制时序一致性损失权重，建议范围[0.05, 0.2]

## 实现细节

### 1. 指数惩罚计算

```python
penalty = -torch.max(
    torch.zeros_like(time_to_accident),
    (time_to_accident - time_indices - 1) / fps
)
```

### 2. 正样本损失

```python
pos_loss = -torch.exp(penalty) * torch.log(pred_probs + 1e-8)
```

### 3. 负样本损失

```python
neg_loss = -torch.log(1 - pred_probs + 1e-8)
```

### 4. 时序一致性损失

```python
prob_diff = pred_probs[1:] - pred_probs[:-1]
consistency_loss = torch.clamp(-prob_diff, min=0).mean()
```

## 优势

1. **早期预测**: 通过指数惩罚机制鼓励模型更早地预测事故
2. **自适应权重**: 根据时间距离动态调整损失权重
3. **时序一致性**: 确保预测的单调性，避免预测概率的剧烈波动
4. **平衡性**: 通过参数调节平衡正负样本的贡献

## 注意事项

1. **数据格式**: 确保输入数据包含正确的时间信息（frame_inds, accident_frame等）
2. **参数调优**: 根据具体数据集和任务需求调整参数
3. **计算效率**: AdaLEA损失函数比标准BCE损失计算更复杂，可能影响训练速度
4. **内存使用**: 指数计算可能增加内存使用量

## 实验结果

在交通事故事件预测任务中，AdaLEA损失函数相比标准BCE损失函数通常能够：

- 提高早期预测的准确性
- 减少误报率
- 提升整体预测性能
- 增强模型的时序理解能力

## 参考文献

1. "Anticipating Traffic Accidents with Adaptive Loss and Large-scale Incident DB"
2. 相关代码实现参考了CAP、DSTA、GSC等项目的损失函数设计 