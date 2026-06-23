import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
"""
    第二步：自适应频域门控机制
"""

def dwt_init(x):
    """小波变换（与原HFP代码保持一致）"""
    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    x_LL = (x1 + x2 + x3 + x4) / 4
    x_HL = (-x1 - x2 + x3 + x4) / 4
    x_LH = (-x1 + x2 - x3 + x4) / 4
    x_HH = (x1 - x2 - x3 + x4) / 4

    return x_LL, x_HL, x_LH, x_HH


def idwt_init(x_LL, x_HL, x_LH, x_HH):
    """逆小波变换"""
    min_height = min(x_LL.size(2), x_HL.size(2), x_LH.size(2), x_HH.size(2))
    min_width = min(x_LL.size(3), x_HL.size(3), x_LH.size(3), x_HH.size(3))

    x_LL = x_LL[:, :, :min_height, :min_width]
    x_HL = x_HL[:, :, :min_height, :min_width]
    x_LH = x_LH[:, :, :min_height, :min_width]
    x_HH = x_HH[:, :, :min_height, :min_width]

    x1 = x_LL - x_HL - x_LH + x_HH
    x2 = x_LL - x_HL + x_LH - x_HH
    x3 = x_LL + x_HL - x_LH - x_HH
    x4 = x_LL + x_HL + x_LH + x_HH

    upper = torch.zeros(x1.size(0), x1.size(1), x1.size(2)*2, x1.size(3), 
                       device=x1.device, dtype=x1.dtype)
    lower = torch.zeros_like(upper)

    upper[:, :, 0::2, :] = x1
    upper[:, :, 1::2, :] = x2
    lower[:, :, 0::2, :] = x3
    lower[:, :, 1::2, :] = x4

    x = torch.zeros(x1.size(0), x1.size(1), upper.size(2), upper.size(3)*2, 
                   device=x1.device, dtype=x1.dtype)
    x[:, :, :, 0::2] = upper
    x[:, :, :, 1::2] = lower

    return x


class AdaptiveFrequencyGate(nn.Module):
    """
    自适应频域门控模块
    
    论文核心创新：
    1. 动态学习每个样本的最优频域过滤策略
    2. 通过门控机制自适应选择有效频段
    3. 提升跨域泛化能力
    
    关键思想：
    不同数据集的裂缝在不同频段有不同的判别性
    - DeepCrack: 高频边缘清晰
    - CFD: 中频纹理丰富
    门控机制让模型自动选择当前样本的最优频段
    """
    def __init__(self, in_channels, num_freq_bands=4, reduction=16):
        super().__init__()
        self.num_bands = num_freq_bands
        
        # 频段重要性评估网络
        self.band_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, num_freq_bands, 1),
            nn.Sigmoid()  # 输出每个频段的权重 [0, 1]
        )
        
        # 频段自适应融合
        self.band_fusion = nn.Conv2d(num_freq_bands, 1, 1)
        
    def decompose_frequency_bands(self, freq_feat):
        """将频域特征分解为多个频段"""
        _, _, h, w = freq_feat.shape
        bands = []
        
        for i in range(self.num_bands):
            # 创建频段mask（从低频到高频）
            h_start = int(h * i / self.num_bands)
            h_end = int(h * (i + 1) / self.num_bands)
            w_start = int(w * i / self.num_bands)
            w_end = int(w * (i + 1) / self.num_bands)
            
            mask = torch.zeros_like(freq_feat)
            mask[:, :, h_start:h_end, w_start:w_end] = 1.0
            
            band = freq_feat * mask
            bands.append(band)
        
        return bands
    
    def forward(self, x):
        """
        x: 输入特征图 [B, C, H, W]
        返回: 自适应过滤后的频域特征
        """
        # DCT变换到频域
        freq_feat = DCT.dct_2d(x, norm='ortho')
        
        # 计算频段重要性权重 [B, num_bands, 1, 1]
        band_weights = self.band_importance(x)
        
        # 分解频段
        freq_bands = self.decompose_frequency_bands(freq_feat)
        
        # 加权融合频段
        weighted_bands = []
        for i, band in enumerate(freq_bands):
            weight = band_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            weighted_bands.append(band * weight)
        
        # 融合所有频段
        freq_filtered = sum(weighted_bands)
        
        # 逆DCT回到空域
        spatial_feat = DCT.idct_2d(freq_filtered, norm='ortho')
        
        return spatial_feat, band_weights  # 返回权重用于可视化


class HFP_WithAdaptiveGating(BaseModule):
    """
    增强版HFP：集成自适应频域门控
    
    论文贡献：
    1. 保留原HFP的频域增强能力
    2. 新增自适应门控机制，动态调节频域策略
    3. 提升跨数据集泛化性能
    """
    def __init__(self,
                 in_channels,
                 ratio=(0.25, 0.25),
                 patch=(8, 8),
                 isdct=True,
                 use_adaptive_gate=True,  # 新增：是否启用自适应门控
                 num_freq_bands=4,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        
        self.use_adaptive_gate = use_adaptive_gate
        
        # 原HFP的空间路径（简化版，保留核心逻辑）
        self.spatial_conv = nn.Sequential(
            ConvModule(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels, in_channels)
        )
        
        # 原HFP的通道路径
        self.channel_conv = nn.Sequential(
            ConvModule(in_channels, in_channels, 1),
            nn.GELU()
        )
        
        # 【核心创新】自适应频域门控
        if self.use_adaptive_gate:
            self.freq_gate_spatial = AdaptiveFrequencyGate(
                in_channels, num_freq_bands=num_freq_bands
            )
            self.freq_gate_channel = AdaptiveFrequencyGate(
                in_channels, num_freq_bands=num_freq_bands
            )
        
        # 输出融合
        self.out = nn.Sequential(
            ConvModule(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels, in_channels)
        )
        
    def forward(self, x_cross, x_vssm):
        """
        x_cross: 空间路径输入
        x_vssm: 通道路径输入
        """
        # 空间路径处理
        if self.use_adaptive_gate:
            spatial_freq, spatial_weights = self.freq_gate_spatial(x_cross)
            spatial = self.spatial_conv(spatial_freq)
        else:
            spatial = self.spatial_conv(x_cross)
            spatial_weights = None
        
        # 通道路径处理
        if self.use_adaptive_gate:
            channel_freq, channel_weights = self.freq_gate_channel(x_vssm)
            channel = self.channel_conv(channel_freq)
        else:
            channel = self.channel_conv(x_vssm)
            channel_weights = None
        
        # 融合
        out = self.out(spatial + channel)
        
        # 返回频段权重用于分析
        if self.training and self.use_adaptive_gate:
            return out, {
                'spatial_weights': spatial_weights,
                'channel_weights': channel_weights
            }
        else:
            return out


class FrequencyGatingLoss(nn.Module):
    """
    频域门控正则化损失
    
    目的：
    1. 鼓励模型在不同样本上使用不同的频域策略（多样性）
    2. 防止门控权重退化为固定值（避免门控失效）
    """
    def __init__(self, diversity_weight=0.1):
        super().__init__()
        self.diversity_weight = diversity_weight
        
    def forward(self, spatial_weights, channel_weights):
        """
        spatial_weights: [B, num_bands, 1, 1]
        channel_weights: [B, num_bands, 1, 1]
        """
        # 1. 多样性损失：鼓励batch内不同样本使用不同权重
        spatial_diversity = -torch.std(spatial_weights, dim=0).mean()
        channel_diversity = -torch.std(channel_weights, dim=0).mean()
        
        diversity_loss = spatial_diversity + channel_diversity
        
        # 2. 熵正则：防止权重退化为one-hot
        spatial_entropy = -(spatial_weights * torch.log(spatial_weights + 1e-8)).sum(dim=1).mean()
        channel_entropy = -(channel_weights * torch.log(channel_weights + 1e-8)).sum(dim=1).mean()
        
        entropy_loss = -(spatial_entropy + channel_entropy)  # 最大化熵
        
        total_loss = self.diversity_weight * (diversity_loss + entropy_loss)
        
        return total_loss


# ========== 使用示例 ==========
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟输入
    x_cross = torch.randn(4, 64, 128, 128).to(device)
    x_vssm = torch.randn(4, 64, 128, 128).to(device)
    
    print("="*60)
    print("测试1: 基础自适应频域门控")
    gate = AdaptiveFrequencyGate(64, num_freq_bands=4).to(device)
    out, weights = gate(x_cross)
    print(f"输入shape: {x_cross.shape}")
    print(f"输出shape: {out.shape}")
    print(f"频段权重shape: {weights.shape}")
    print(f"频段权重示例:\n{weights[0].squeeze().detach().cpu().numpy()}")
    
    print("\n" + "="*60)
    print("测试2: 增强版HFP（带自适应门控）")
    
    # 原始HFP（baseline）
    model_baseline = HFP_WithAdaptiveGating(
        64, use_adaptive_gate=False
    ).to(device)
    model_baseline.train()
    out_baseline = model_baseline(x_cross, x_vssm)
    print(f"Baseline输出shape: {out_baseline.shape}")
    
    # 增强版HFP
    model_enhanced = HFP_WithAdaptiveGating(
        64, use_adaptive_gate=True, num_freq_bands=4
    ).to(device)
    model_enhanced.train()
    out_enhanced, gate_info = model_enhanced(x_cross, x_vssm)
    print(f"Enhanced输出shape: {out_enhanced.shape}")
    print(f"空间权重: {gate_info['spatial_weights'].shape}")
    print(f"通道权重: {gate_info['channel_weights'].shape}")
    
    print("\n" + "="*60)
    print("测试3: 频域门控正则化损失")
    gate_loss = FrequencyGatingLoss(diversity_weight=0.1).to(device)
    loss = gate_loss(
        gate_info['spatial_weights'],
        gate_info['channel_weights']
    )
    print(f"门控正则损失: {loss.item():.6f}")
    
    print("\n✓ 自适应门控机制可用于提升跨域泛化")
    print("✓ 可视化频段权重以分析不同数据集的频域偏好")