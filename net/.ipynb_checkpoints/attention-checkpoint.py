import math
import torch
from torch import nn
import torch.fft as fft
from torch.cuda import max_memory_reserved
from torch.nn import init
from GatedBottConv import CAGU
import torch.nn.functional as F


class HybridSEBlock(nn.Module):
    def __init__(self, mode, channels, ratio=16, expansion=4):
        super(HybridSEBlock, self).__init__()
        self.mode = mode
        self.channels = channels
        self.ratio = ratio
        self.expansion = expansion

        self.original_pool = self._create_pooling(mode)
        self.original_fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid()
        )

        self.enhanced_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels // expansion, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // expansion, channels, 1),
            nn.Sigmoid()
        )

    def _create_pooling(self, mode):
        if mode == "max":
            return nn.AdaptiveMaxPool2d(1)
        elif mode == "avg":
            return nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError("Invalid pooling mode")

    def forward(self, x):
        b, c, h, w = x.shape

        original_pool = self.original_pool(x)  # [B, C, 1, 1]
        original_flat = original_pool.view(b, -1)  # [B, C]
        original_weights = self.original_fc(original_flat).view(b, c, 1, 1)  # [B, C, 1, 1]

        avg_pool = F.adaptive_avg_pool2d(x, 1)  # [B, C, 1, 1]
        max_pool = F.adaptive_max_pool2d(x, 1)  # [B, C, 1, 1]
        concatenated = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2C, 1, 1]
        enhanced_weights = self.enhanced_conv(concatenated)  # [B, C, 1, 1]

        combined_weights = original_weights * enhanced_weights
        return x * combined_weights

class SEBlock(nn.Module):
    def __init__(self, mode, channels, ratio):
        super(SEBlock, self).__init__()
        self.mode = mode
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        if mode == "max":
            self.global_pooling = self.max_pooling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.avg_pooling(x).view(b, c) + self.max_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=3)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return x*output

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class MultiHeadAttentionA2(nn.Module):
    def __init__(self, in_channel, num_heads):
        super(MultiHeadAttentionA2, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channel // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(in_channel, in_channel)
        self.key = nn.Linear(in_channel, in_channel)
        self.value = nn.Linear(in_channel, in_channel)

    def forward(self, x):
        B, C = x.size(0), x.size(1)
        q = self.query(x).view(B, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, B, head_dim)
        k = self.key(x).view(B, self.num_heads, self.head_dim).transpose(0, 1)    # (num_heads, B, head_dim)
        v = self.value(x).view(B, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, B, head_dim)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # (num_heads, B, B)
        attention_weights = torch.softmax(attention_scores, dim=-1)           # (num_heads, B, B)

        A2_heads = torch.matmul(attention_weights, v)  # (num_heads, B, head_dim)
        A2_heads = A2_heads.transpose(0, 1).contiguous().view(B, C)  # (B, C)

        return A2_heads


class MixMaxPool(nn.Module):
    def __init__(self, num_groups, pool_sizes):
        super().__init__()
        assert len(pool_sizes) == num_groups,
        self.num_groups = num_groups


        self.pools = nn.ModuleList()
        for size in pool_sizes:
            kernel_size = (size, size) if isinstance(size, int) else size
            assert kernel_size[0] % 2 == 0 and kernel_size[1] % 2 == 0, 

            stride = 2
            padding = ((kernel_size[0] - stride) // 2, (kernel_size[1] - stride) // 2)

            self.pools.append(
                nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )

    def forward(self, x):
        x_split = torch.chunk(x, self.num_groups, dim=1)
        pooled = [pool(group) for pool, group in zip(self.pools, x_split)]
        return torch.cat(pooled, dim=1)


class MixAvgPool(nn.Module):
    def __init__(self, num_groups, pool_sizes):
        super().__init__()
        assert len(pool_sizes) == num_groups
        self.num_groups = num_groups

        self.pools = nn.ModuleList()
        for size in pool_sizes:
            kernel_size = (size, size) if isinstance(size, int) else size
            assert kernel_size[0] % 2 == 0 and kernel_size[1] % 2 == 0

            stride = 2
            padding = ((kernel_size[0] - stride) // 2, (kernel_size[1] - stride) // 2)

            self.pools.append(
                nn.AvgPool2d(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    count_include_pad=False
                )
            )

    def forward(self, x):
        x_split = torch.chunk(x, self.num_groups, dim=1)
        pooled = [pool(group) for pool, group in zip(self.pools, x_split)]
        return torch.cat(pooled, dim=1)

class CGSA(nn.Module):
    def __init__(self, in_channels, ratio, dilation=2, reduction_ratio=4):
        super().__init__()
        hide_channel = in_channels // ratio
        self.mix_max_pool = MixMaxPool(num_groups=4, pool_sizes=[2, 4, 6, 8])
        self.mix_avg_pool = MixAvgPool(num_groups=4, pool_sizes=[2, 4, 6, 8])
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * in_channels, max(2 * in_channels // reduction_ratio, 1), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(2 * in_channels // reduction_ratio, 1), 2 * in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3,
                      dilation=dilation, padding=dilation, groups=2 * in_channels),
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        max_result = self.mix_max_pool(x)
        avg_result = self.mix_avg_pool(x)
        max_result = F.interpolate(max_result, size=x.shape[2:], mode='bilinear')
        avg_result = F.interpolate(avg_result, size=x.shape[2:], mode='bilinear')
        result = torch.cat([max_result, avg_result], dim=1)
        channel_weights = self.channel_att(result)
        weighted_result = result * channel_weights
        outputs = self.conv(weighted_result)
        outputs = self.sigmoid(outputs)
        return x * outputs + x

class TDAGCA(nn.Module):
    def __init__(self, in_channel, ratio,num_heads=4):
        super(TDAGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.ratio = ratio
        self.A0 = torch.eye(hide_channel).to('cuda')
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.delta_gen = nn.Sequential(
            nn.Linear(hide_channel, max(hide_channel // 4, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(hide_channel // 4, 4), 2 * hide_channel),
        )
        self.multihead_attention = MultiHeadAttentionA2(hide_channel, num_heads=num_heads)
        self.weight_A2_base = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        self.weight_A2_attention = nn.Parameter(torch.tensor(2.0), requires_grad=True)
        self.weight_A2_delta = nn.Parameter(torch.tensor(6.0), requires_grad=True)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Hardsigmoid()
        self.fusion_conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
    def forward(self, x):
        B, C = x.size(0), x.size(1)
        y_pool = x.view(B, C)  # (B, C)
        A2_attention = self.multihead_attention(y_pool)  # (B, C)
        uv = self.delta_gen(y_pool)  # (B, 2*C)
        u, v = uv.chunk(2, dim=1)  # (B, C), (B, C)
        delta_A2 = torch.bmm(u.unsqueeze(2), v.unsqueeze(1))  # (B, C, C)
        A2_total = (self.weight_A2_base * self.A2.unsqueeze(0) + self.weight_A2_attention * A2_attention.unsqueeze(1) + self.weight_A2_delta * delta_A2)
        y = x.flatten(2).transpose(1, 2)  # [B, 1, C]
        A1 = self.softmax(self.conv2(y))  # (B, C, 1)
        A1 = A1.expand(B, C, C)  # (B, C, C)

        A = (self.A0 * A1) + A2_total  # (B, C, C)
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))

        y = y.transpose(1, 2).view(-1, C, 1, 1)
        return y
            
class MultiScaleCrackGraphAttention(nn.Module):
    def __init__(self, in_channel, ratio, crack_scales=[1, 3, 5, 7]):
        super(MultiScaleCrackGraphAttention, self).__init__()
        hide_channel = in_channel // ratio
        self.crack_scales = crack_scales

        self.crack_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False),
                nn.Conv2d(hide_channel, hide_channel, kernel_size=(scale, 1),
                          padding=(scale // 2, 0), bias=False), 
                nn.Conv2d(hide_channel, hide_channel, kernel_size=(1, scale),
                          padding=(0, scale // 2), bias=False), 
                nn.ReLU(inplace=True)
            ) for scale in crack_scales
        ])

        self.direction_graphs = nn.ModuleList([
            self._build_direction_graph(hide_channel) for _ in crack_scales
        ])

        self.scale_fusion = nn.Sequential(
            nn.Conv2d(hide_channel * len(crack_scales), hide_channel, 1, bias=False),
            nn.BatchNorm2d(hide_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hide_channel, in_channel, 1, bias=False)
        )
        self.scale_weights = nn.Parameter(torch.ones(len(crack_scales)))

        self.base_graph_attention = TDAGCA(hide_channel, 1)

    def _build_direction_graph(self, channels):
        return nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels * channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.size()

        scale_features = []
        for i, scale_conv in enumerate(self.crack_scale_convs):
            scale_feat = scale_conv(x) 
            scale_desc = F.adaptive_avg_pool2d(scale_feat, 1).view(B, -1)
            direction_graph = self.direction_graphs[i](scale_desc)
            direction_graph = direction_graph.view(B, scale_feat.size(1), scale_feat.size(1))
            scale_feat_pooled = F.adaptive_avg_pool2d(scale_feat, 1)
            graph_enhanced = self.base_graph_attention(scale_feat_pooled)
            scale_feat_enhanced = scale_feat * graph_enhanced
            scale_features.append(scale_feat_enhanced * self.scale_weights[i])
        fused_features = torch.cat(scale_features, dim=1)
        output = self.scale_fusion(fused_features)

        return output



class MGCA(nn.Module):
    def __init__(self, in_channel, ratio, use_multiscale=True, use_topology=True, use_frequency=True):
        super(MGCA, self).__init__()
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale_module = MultiScaleCrackGraphAttention(in_channel, ratio)
        num_modules = sum([use_multiscale, use_topology, use_frequency])
        self.fusion_weights = nn.Parameter(torch.ones(num_modules) / num_modules)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 4, in_channel, 1),
            nn.Sigmoid()
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        outputs = []

        if self.use_multiscale:
            multiscale_out = self.multiscale_module(x)
            outputs.append(multiscale_out)
        for i in range(1, len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=outputs[0].shape[2:],
                                      mode='bilinear', align_corners=False)
        if len(outputs) > 1:
            weighted_sum = sum(w * out for w, out in zip(self.fusion_weights, outputs))
        else:
            weighted_sum = outputs[0]
        attention = self.final_fusion(weighted_sum)
        enhanced_x = x * attention
        output = (self.residual_weight * enhanced_x) + (1 - self.residual_weight) * x
        return output

    
class SCFE(nn.Module):
    def __init__(self,inc,ratio=2,dilation=2,norm_type='GN'):
        super(SCFE,self).__init__()
        self.sa = CGSA(inc, ratio)
        self.ca = MGCA(inc, ratio,use_multiscale=True)
        self.gated = CAGU(dim=inc)
    def forward(self,x):
        out1 = self.ca(x)
        out2 = self.sa(x)
        gated_out = self.gated(out1, out2)
        return gated_out + x

