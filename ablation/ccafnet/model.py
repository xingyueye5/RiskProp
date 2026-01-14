import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)


class CrossAttentionFusion(nn.Module):
    """Simple cross-attention: Q from global(scene) features, K/V from agent features.
    Here we implement a spatial-pooled version for simplicity.
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
        self.q = nn.Linear(dim, hidden_dim)
        self.k = nn.Linear(dim, hidden_dim)
        self.v = nn.Linear(dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, dim)

    def forward(self, scene_feat, agent_feat):
        # scene_feat: [B, C]
        # agent_feat: [B, N_agents, C]
        # produce fused scene-aware agent aggregation and residual update
        q = self.q(scene_feat).unsqueeze(1)  # [B,1,H]
        k = self.k(agent_feat)               # [B,N,H]
        v = self.v(agent_feat)               # [B,N,H]
        attn = torch.softmax(torch.bmm(q, k.transpose(1,2)) / (k.size(-1) ** 0.5), dim=-1)  # [B,1,N]
        agg = torch.bmm(attn, v).squeeze(1)  # [B,H]
        out = self.proj(agg) + scene_feat
        return out


class CascadeFusionBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.se = SEBlock(dim, reduction)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, C]
        y = self.se(x.unsqueeze(-1).unsqueeze(-1)).view(x.size())
        y = self.ff(y)
        return self.norm(x + y)


class CCAFNetBaseline(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, feat_dim=512, rnn_hidden=512, num_layers=1, cascade_stages=2):
        super().__init__()
        assert backbone in ['resnet50']
        # Robust pretrained loading across torchvision versions
        try:
            # torchvision>=0.13 uses weights enum
            if pretrained:
                res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                res = models.resnet50(weights=None)
        except Exception:
            # fallback to legacy API
            res = models.resnet50(pretrained=pretrained)
        # remove fc and avgpool
        modules = list(res.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(2048, feat_dim)

        # optional agent branch placeholder (we keep it simple: no detector required)
        # We'll implement agent features as the result of a 3x3 conv over feature maps followed by pooling
        self.agent_conv = nn.Conv2d(2048, feat_dim, kernel_size=3, padding=1)

        # Cascade fusion stages - each stage refines scene feature
        self.cascade_stages = cascade_stages
        self.fusion_stages = nn.ModuleList([CascadeFusionBlock(feat_dim) for _ in range(cascade_stages)])
        self.cross_attention = CrossAttentionFusion(feat_dim)

        # temporal module
        self.rnn = nn.GRU(input_size=feat_dim, hidden_size=rnn_hidden, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc_out = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_hidden//2, 1),
            nn.Sigmoid()
        )

    def forward(self, frames):
        # frames: [B, T, 3, H, W]
        B, T, C, H, W = frames.shape
        frames = frames.view(B*T, C, H, W)
        feats_map = self.encoder(frames)  # [B*T, 2048, h, w]
        # scene pooled feature
        scene_pool = self.pool(feats_map).view(B*T, -1)  # [B*T, 2048]
        scene_feat = self.project(scene_pool)            # [B*T, feat_dim]

        # agent features (simplified): conv + pool -> simulate multiple agents by spatial grid
        agent_map = self.agent_conv(feats_map)  # [B*T, feat_dim, h, w]
        # flatten spatial locations as "agents"
        bs_tw, c_a, h_a, w_a = agent_map.shape
        agent_feat = agent_map.view(bs_tw, c_a, h_a*w_a).permute(0,2,1)  # [B*T, N_agents, C]

        # perform cascade fusion per time step
        fused = scene_feat
        for i in range(self.cascade_stages):
            fused = self.cross_attention(fused, agent_feat)
            fused = self.fusion_stages[i](fused)

        # reshape to temporal sequence
        fused_seq = fused.view(B, T, -1)  # [B, T, feat_dim]
        rnn_out, _ = self.rnn(fused_seq)
        logits = self.fc_out(rnn_out)  # [B, T, 1]
        logits = logits.squeeze(-1)
        return logits