import torch
import torch.nn as nn
from detectron2.layers import Conv2d

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # mean on spatial dim
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        # max on spatial dim
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        feat = torch.cat([avg_feat, max_feat], dim=1)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * x

