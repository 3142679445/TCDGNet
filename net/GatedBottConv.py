import math
import torch
import torch.nn as nn

class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)


class CAGU(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2 // reduction, dim),
            nn.Sigmoid()
        )
        self.dir_predictor = nn.Sequential(
            nn.Conv2d(dim, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )
        self.bd_attn = nn.ModuleDict({
            'horizontal': nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2)),
            'vertical': nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0))
        })
        self.compensate_conv = nn.Conv2d(dim, dim, kernel_size=3,
                                         padding=1, groups=dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x_cat = torch.cat([x1.flatten(2).mean(-1),
                           x2.flatten(2).mean(-1)], dim=1)  # [B, 2C]
        gate = self.gate_fc(x_cat).view(B, C, 1, 1)  # [B, C, 1, 1]
        fused = x1 * gate + x2 * (1 - gate)
        dir_prob = self.dir_predictor(fused)  # [B, 2, H, W]
        dir_h, dir_v = dir_prob[:, 0], dir_prob[:, 1]
        horiz_feat = self.bd_attn['horizontal'](fused) * dir_h.unsqueeze(1)
        vert_feat = self.bd_attn['vertical'](fused) * dir_v.unsqueeze(1)
        compensated = self.compensate_conv(fused + horiz_feat + vert_feat)
        return torch.sigmoid(compensated) * fused
