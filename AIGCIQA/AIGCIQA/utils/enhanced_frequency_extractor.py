import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia.color as K
import kornia.geometry.transform as KT
from torch.fft import fft2, fftshift, ifft2

class EnhancedFrequencyFeatureExtractor(nn.Module):
    """
    增强的频域特征提取器，结合多尺度频域分析和学习型滤波器
    """
    def __init__(self, output_dim=768, num_scales=4, num_orientations=8):
        super(EnhancedFrequencyFeatureExtractor, self).__init__()
        self.output_dim = output_dim
        self.num_scales = num_scales
        self.num_orientations = num_orientations

        # Y通道专用频域注意力模块（强化高频成分）
        self.y_freq_attention = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Cb/Cr通道的轻量级注意力模块
        self.chroma_freq_attention = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # 高频成分增强器（专门针对Y通道）
        self.high_freq_enhancer = nn.Sequential(
            nn.Conv2d(1, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
            nn.Tanh()  # 允许负值来抑制低频
        )
        
        # 多尺度特征提取
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2**i, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, 1),
                nn.BatchNorm2d(32)
            ) for i in range(num_scales)
        ])
        
        # 频域统计特征提取
        self.stat_extractor = nn.Sequential(
            nn.Linear(12, 64),  # 12个统计特征
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 特征融合
        total_features = 32 * num_scales + 64
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

        # 固定坐标缓存：只依赖(H, W, device)，避免每个batch重建meshgrid/mask/分桶
        self._high_freq_mask_cache = {}
        self._radial_bin_cache = {}

    def extract_frequency_stats(self, freq_magnitude):
        """提取频域统计特征"""
        B, C, H, W = freq_magnitude.shape
        
        # 将频谱分为中心和边缘区域
        center_h, center_w = H // 4, W // 4
        center = freq_magnitude[:, :, H//2-center_h:H//2+center_h, W//2-center_w:W//2+center_w]
        
        # 计算统计特征
        stats = []
        
        # 1. 全局能量 [B, C]
        total_energy = freq_magnitude.sum(dim=(2, 3))
        stats.append(total_energy)
        
        # 2. 中心能量比例 [B, C]
        center_energy = center.sum(dim=(2, 3))
        center_ratio = center_energy / (total_energy + 1e-8)
        stats.append(center_ratio)
        
        # 3. 频谱熵 [B, C]
        freq_prob = freq_magnitude / (freq_magnitude.sum(dim=(2, 3), keepdim=True) + 1e-8)
        entropy = -(freq_prob * torch.log(freq_prob + 1e-8)).sum(dim=(2, 3))
        stats.append(entropy)
        
        # 4. 径向分布特征 [B, C]
        radial_profile = self._compute_radial_profile(freq_magnitude)
        if radial_profile.dim() == 3:  # [B, C, num_profiles]
            radial_mean = radial_profile.mean(dim=2)  # [B, C]
        else:  # [B, C]
            radial_mean = radial_profile
        stats.append(radial_mean)
        
        return torch.cat(stats, dim=1)
    
    def _cache_key(self, H, W, device):
        return (H, W, device.type, device.index)

    def _get_high_freq_mask(self, H, W, device):
        """获取固定尺寸/设备的高频区域mask，避免每个batch重复meshgrid。"""
        key = self._cache_key(H, W, device)
        mask = self._high_freq_mask_cache.get(key)
        if mask is None:
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            center_y, center_x = H // 2, W // 2
            dist_from_center = torch.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
            max_dist = torch.sqrt(torch.tensor(center_x ** 2 + center_y ** 2, device=device))
            mask = (dist_from_center > 0.4 * max_dist).float()
            self._high_freq_mask_cache[key] = mask
        return mask

    def _get_radial_bins(self, H, W, device):
        """获取径向分桶索引，替代forward中的Python循环逐mask求和。"""
        key = self._cache_key(H, W, device)
        cached = self._radial_bin_cache.get(key)
        if cached is None:
            y, x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            center_y, center_x = H // 2, W // 2
            r = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_r = min(H, W) // 2
            valid = r < max_r
            bin_index = (r[valid] // 4).long()
            bin_count = torch.bincount(bin_index, minlength=max_r // 4).clamp_min(1).float()
            cached = (valid.flatten(), bin_index, bin_count)
            self._radial_bin_cache[key] = cached
        return cached

    def _compute_radial_profile(self, freq_magnitude):
        """计算径向频谱分布"""
        B, C, H, W = freq_magnitude.shape
        device = freq_magnitude.device
        valid_flat, bin_index, bin_count = self._get_radial_bins(H, W, device)

        values = freq_magnitude.flatten(2)[:, :, valid_flat]
        num_bins = bin_count.numel()
        sums = freq_magnitude.new_zeros(B, C, num_bins)
        index = bin_index.view(1, 1, -1).expand(B, C, -1)
        sums.scatter_add_(2, index, values)
        radial_profile = sums / bin_count.view(1, 1, -1)
        return radial_profile.mean(dim=2)
    
    def forward(self, x):
        # [AMP稳定性] FFT / torch.log(abs) / torch.angle 在半精度(fp16)下极易
        # 产生 inf/NaN，强制本模块在 fp32 下计算（模块较轻量，开销可忽略）。
        # 其余大模块(ConvNeXt主干/CLIP文本/MFB/MLP)仍享受 AMP 收益。
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward_impl(x.float())

    def _forward_impl(self, x):
        """
        前向传播

        参数:
            x: 输入图像 [B, 3, H, W]

        返回:
            频域特征向量 [B, output_dim]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # RGB转YCbCr
        ycbcr = K.rgb_to_ycbcr(x)
        
        # 1. 频域变换和差异化注意力处理
        fft_features = []
        for i in range(3):
            channel = ycbcr[:, i:i+1, :, :]
            
            # FFT变换
            freq = fft2(channel)
            freq_shift = fftshift(freq)
            magnitude = torch.log(torch.abs(freq_shift) + 1e-10)
            phase = torch.angle(freq_shift)
            
            if i == 0:  # Y通道 - 专用处理
                # 应用Y通道专用注意力
                y_attention = self.y_freq_attention(magnitude)
                
                # 使用缓存的高频掩码强化高频成分
                h, w = magnitude.shape[2], magnitude.shape[3]
                high_freq_mask = self._get_high_freq_mask(h, w, device).unsqueeze(0).unsqueeze(0)

                # 应用高频增强
                freq_enhancement = self.high_freq_enhancer(magnitude)
                enhanced_magnitude = magnitude + freq_enhancement * high_freq_mask * 0.5
                
                # 最终加权
                weighted_mag = enhanced_magnitude * y_attention
                
            else:  # Cb/Cr通道 - 轻量级处理
                # 应用色度通道的轻量级注意力
                chroma_attention = self.chroma_freq_attention(magnitude)
                weighted_mag = magnitude * chroma_attention
            
            fft_features.append(weighted_mag)
        
        # 合并通道
        fft_combined = torch.cat(fft_features, dim=1)  # [B, 3, H, W]
        
        # 2. 多尺度卷积特征提取
        multi_scale_features = []
        for i, conv in enumerate(self.multi_scale_convs):
            # 对FFT特征应用多尺度卷积
            scale_feature = conv(fft_combined)
            multi_scale_features.append(F.adaptive_avg_pool2d(scale_feature, (1, 1)).squeeze(-1).squeeze(-1))
        
        # 3. 提取统计特征
        freq_stats = self.extract_frequency_stats(torch.abs(fft_combined))
        stat_features = self.stat_extractor(freq_stats)
        
        # 4. 特征融合
        all_features = torch.cat(multi_scale_features + [stat_features], dim=1)
        output = self.fusion(all_features)
        
        return output


class AdaptiveFrequencyFilter(nn.Module):
    """自适应频域滤波器，用于图像质量相关特征提取"""
    def __init__(self, channels=3):
        super(AdaptiveFrequencyFilter, self).__init__()
        
        # 可学习的频域掩码生成器
        self.mask_generator = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        应用自适应频域滤波
        
        参数:
            x: 输入图像 [B, C, H, W]
            
        返回:
            滤波后的特征和滤波掩码
        """
        # 转换到频域
        freq = fft2(x)
        freq_shift = fftshift(freq)
        magnitude = torch.abs(freq_shift)
        phase = torch.angle(freq_shift)
        
        # 生成自适应掩码
        mask = self.mask_generator(torch.log(magnitude + 1e-10))
        
        # 应用掩码
        filtered_mag = magnitude * mask
        
        # 重建信号
        filtered_freq = filtered_mag * torch.exp(1j * phase)
        filtered_freq_shift = fftshift(filtered_freq)
        filtered_spatial = torch.real(ifft2(filtered_freq_shift))
        
        return filtered_spatial, mask