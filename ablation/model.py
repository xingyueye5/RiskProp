import torch
import torch.nn as nn
import torchvision.models as models

class BaselineModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        # CNN backbone (提取帧特征)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 去掉分类层
        feat_dim = resnet.fc.in_features

        # 时间建模 (LSTM)
        self.rnn = nn.LSTM(feat_dim, hidden_dim, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_dim, 1)  # 事故 vs 正常
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

        feats = []
        for t in range(T):
            f = self.feature_extractor(x[:, t])  # [B, feat_dim, 1, 1]
            f = f.view(B, -1)
            feats.append(f)
        feats = torch.stack(feats, dim=1)  # [B, T, feat_dim]

        out, _ = self.rnn(feats)   # [B, T, hidden_dim]
        out = self.fc(out)         # [B, T, 1]
        out = self.sigmoid(out).squeeze(-1)  # [B, T]
        return out