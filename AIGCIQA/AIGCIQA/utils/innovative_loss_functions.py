import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RankingContrastiveLoss(nn.Module):
    """基于排序的对比学习损失，更适合质量评估任务"""
    def __init__(self, margin=0.5, temperature=0.1):
        super(RankingContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, text_feat, img_feat, labels):
        """
        参数:
            text_feat: 文本特征 [B, D]
            img_feat: 图像特征 [B, D]
            labels: 质量分数 [B]
        """
        B = text_feat.size(0)
        
        # 归一化特征
        text_feat = F.normalize(text_feat, dim=1)
        img_feat = F.normalize(img_feat, dim=1)
        
        # 计算特征相似度
        similarity = torch.matmul(text_feat, img_feat.T) / self.temperature  # [B, B]
        
        # 创建质量差异矩阵
        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # [B, B]
        
        # 归一化标签差异到[-1, 1]范围
        if torch.abs(label_diff).max() > 0:
            label_diff = label_diff / torch.abs(label_diff).max()
        
        # 创建mask：只考虑质量差异显著的样本对，但限制计算量
        significant_diff = torch.abs(label_diff) > 0.1  # 降低阈值
        
        # 简化的排序损失，避免双重循环
        if significant_diff.sum() > 0:
            # 对角线元素（自相似度）
            diag_sim = torch.diag(similarity)
            
            # 计算排序损失
            loss_matrix = torch.zeros_like(similarity)
            for i in range(B):
                for j in range(B):
                    if i != j and significant_diff[i, j]:
                        expected_order = torch.sign(label_diff[i, j])
                        actual_order = diag_sim[i] - diag_sim[j]
                        loss_matrix[i, j] = torch.relu(self.margin - expected_order * actual_order)
            
            loss = loss_matrix[significant_diff].mean()
            loss = torch.clamp(loss, 0.0, 5.0)  # 限制损失范围
        else:
            loss = torch.tensor(0.0, device=text_feat.device)
        
        return loss


class MultiScaleQualityLoss(nn.Module):
    """多尺度质量感知损失"""
    def __init__(self, scales=[1.0, 0.5, 0.25], weights=[0.5, 0.3, 0.2]):
        super(MultiScaleQualityLoss, self).__init__()
        self.scales = scales
        self.weights = weights
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, features=None):
        """
        参数:
            pred: 预测分数 [B]
            target: 目标分数 [B]
            features: 可选的中间特征，用于多尺度分析
        """
        total_loss = 0
        
        # 基础MSE损失
        base_loss = self.mse(pred, target)
        total_loss += self.weights[0] * base_loss
        
        # 如果提供了特征，计算多尺度一致性
        if features is not None and len(features) > 1:
            for i, (scale, weight) in enumerate(zip(self.scales[1:], self.weights[1:])):
                if i < len(features) - 1:
                    # 确保两个特征维度相同，使用投影层统一维度
                    feat1 = features[i]
                    feat2 = features[i+1]
                    
                    # 获取最小维度作为目标维度
                    min_dim = min(feat1.size(-1), feat2.size(-1))
                    
                    # 如果维度不同，进行投影
                    if feat1.size(-1) != min_dim:
                        feat1 = feat1[..., :min_dim]
                    if feat2.size(-1) != min_dim:
                        feat2 = feat2[..., :min_dim]
                    
                    # 应用尺度变换
                    target_size = int(min_dim * scale)
                    if target_size > 0:
                        feat1_scaled = F.adaptive_avg_pool1d(feat1.unsqueeze(1), target_size).squeeze(1)
                        feat2_scaled = F.adaptive_avg_pool1d(feat2.unsqueeze(1), target_size).squeeze(1)
                        
                        consistency_loss = F.mse_loss(feat1_scaled, feat2_scaled)
                        total_loss += weight * consistency_loss
        
        return total_loss


class PerceptualQualityLoss(nn.Module):
    """感知质量损失，结合人类视觉系统特性"""
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(PerceptualQualityLoss, self).__init__()
        self.alpha = alpha  # 权重：绝对误差
        self.beta = beta    # 权重：相对误差
        self.gamma = gamma  # 权重：排序一致性
        
    def forward(self, pred, target):
        """
        参数:
            pred: 预测分数 [B]
            target: 目标分数 [B]
        """
        B = pred.size(0)
        
        # 1. 绝对误差（L1损失）
        abs_loss = F.l1_loss(pred, target)
        
        # 2. 相对误差（考虑人眼对不同质量范围的敏感度）
        # 归一化target到[0,1]范围以保持数值稳定性
        target_normalized = (target - target.min()) / (target.max() - target.min() + 1e-8)
        pred_normalized = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        
        quality_weight = 1.0 / (target_normalized + 0.1)  # 避免除零，减小权重影响
        weighted_error = quality_weight * torch.abs(pred_normalized - target_normalized)
        rel_loss = weighted_error.mean()
        
        # 3. 排序一致性损失
        if B > 1:
            # 计算预测和真实的排序
            pred_rank = pred.argsort().argsort().float()
            target_rank = target.argsort().argsort().float()
            
            # 使用更稳定的相关系数计算
            pred_rank_norm = F.normalize(pred_rank.unsqueeze(0), p=2, dim=1)
            target_rank_norm = F.normalize(target_rank.unsqueeze(0), p=2, dim=1)
            rank_correlation = F.cosine_similarity(pred_rank_norm, target_rank_norm)
            rank_loss = 1.0 - torch.clamp(rank_correlation, -1.0, 1.0)
        else:
            rank_loss = torch.tensor(0.0, device=pred.device)
        
        # 组合损失，添加数值裁剪
        total_loss = self.alpha * abs_loss + self.beta * rel_loss + self.gamma * rank_loss
        total_loss = torch.clamp(total_loss, 0.0, 10.0)  # 防止损失过大
        
        return total_loss


class DistributionMatchingLoss(nn.Module):
    """分布匹配损失，确保预测分数的分布与真实分布相似"""
    def __init__(self, n_bins=50):
        super(DistributionMatchingLoss, self).__init__()
        self.n_bins = n_bins
        
    def forward(self, pred, target):
        """
        使用Wasserstein距离匹配预测和真实分布
        """
        # 动态确定分数范围
        min_val = min(pred.min().item(), target.min().item())
        max_val = max(pred.max().item(), target.max().item())
        
        # 如果范围太小，使用默认范围
        if max_val - min_val < 1e-6:
            min_val = 0
            max_val = 5
        
        # 计算直方图
        pred_hist = torch.histc(pred, bins=self.n_bins, min=min_val, max=max_val)
        target_hist = torch.histc(target, bins=self.n_bins, min=min_val, max=max_val)
        
        # 归一化（避免除零）
        pred_sum = pred_hist.sum()
        target_sum = target_hist.sum()
        
        if pred_sum > 0:
            pred_hist = pred_hist / pred_sum
        else:
            pred_hist = torch.ones_like(pred_hist) / self.n_bins
            
        if target_sum > 0:
            target_hist = target_hist / target_sum
        else:
            target_hist = torch.ones_like(target_hist) / self.n_bins
        
        # 计算累积分布
        pred_cdf = torch.cumsum(pred_hist, dim=0)
        target_cdf = torch.cumsum(target_hist, dim=0)
        
        # Wasserstein距离
        wasserstein_dist = torch.abs(pred_cdf - target_cdf).sum() / self.n_bins
        
        return wasserstein_dist


class UncertaintyAwareLoss(nn.Module):
    """不确定性感知损失，对难以预测的样本给予不同权重"""
    def __init__(self, reduction='mean'):
        super(UncertaintyAwareLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred_mean, pred_var, target):
        """
        参数:
            pred_mean: 预测均值 [B]
            pred_var: 预测方差 [B]（表示不确定性）
            target: 目标值 [B]
        """
        # 负对数似然损失
        nll_loss = 0.5 * (torch.log(pred_var + 1e-6) + 
                         (target - pred_mean) ** 2 / (pred_var + 1e-6))
        
        if self.reduction == 'mean':
            return nll_loss.mean()
        elif self.reduction == 'sum':
            return nll_loss.sum()
        else:
            return nll_loss


class CompositeLoss(nn.Module):
    """组合损失函数，整合多种损失"""
    def __init__(self, loss_weights=None):
        super(CompositeLoss, self).__init__()
        
        # 调整默认权重，减少数值不稳定的损失函数权重
        if loss_weights is None:
            loss_weights = {
                'mse': 0.8,           # 增加MSE权重，因为它最稳定
                'perceptual': 0.2,   # 减少感知损失权重
                'ranking': 0,       # 减少排序损失权重
                'distribution': 0,  # 保持分布损失权重
                'multi_scale': 0   # 减少多尺度损失权重
            }
            #         if loss_weights is None:
            # loss_weights = {
            #     'mse': 0.6,           # 增加MSE权重，因为它最稳定
            #     'perceptual': 0.15,   # 减少感知损失权重
            #     'ranking': 0.1,       # 减少排序损失权重
            #     'distribution': 0.1,  # 保持分布损失权重
            #     'multi_scale': 0.05   # 减少多尺度损失权重
            # }

        
        self.weights = loss_weights
        
        # 初始化各个损失函数
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualQualityLoss()
        self.ranking_loss = RankingContrastiveLoss()
        self.distribution_loss = DistributionMatchingLoss()
        self.multi_scale_loss = MultiScaleQualityLoss()
        
    def forward(self, pred, target, text_feat=None, img_feat=None, features=None):
        """
        参数:
            pred: 预测分数
            target: 目标分数
            text_feat: 文本特征（可选）
            img_feat: 图像特征（可选）
            features: 多尺度特征（可选）
        """
        losses = {}
        
        # MSE损失（最稳定的基准）
        losses['mse'] = self.mse_loss(pred, target)
        
        # 感知质量损失
        try:
            losses['perceptual'] = self.perceptual_loss(pred, target)
            # 检查是否为异常值
            if torch.isnan(losses['perceptual']) or torch.isinf(losses['perceptual']):
                losses['perceptual'] = losses['mse']  # 回退到MSE
        except:
            losses['perceptual'] = losses['mse']  # 异常时回退
        
        # 排序对比损失（如果提供了特征）
        if text_feat is not None and img_feat is not None:
            try:
                losses['ranking'] = self.ranking_loss(text_feat, img_feat, target)
                if torch.isnan(losses['ranking']) or torch.isinf(losses['ranking']):
                    losses['ranking'] = torch.tensor(0.0, device=pred.device)
            except:
                losses['ranking'] = torch.tensor(0.0, device=pred.device)
        else:
            losses['ranking'] = torch.tensor(0.0, device=pred.device)
        
        # 分布匹配损失
        try:
            losses['distribution'] = self.distribution_loss(pred, target)
            if torch.isnan(losses['distribution']) or torch.isinf(losses['distribution']):
                losses['distribution'] = torch.tensor(0.0, device=pred.device)
        except:
            losses['distribution'] = torch.tensor(0.0, device=pred.device)
        
        # 多尺度损失（如果提供了特征）
        if features is not None:
            try:
                losses['multi_scale'] = self.multi_scale_loss(pred, target, features)
                if torch.isnan(losses['multi_scale']) or torch.isinf(losses['multi_scale']):
                    losses['multi_scale'] = torch.tensor(0.0, device=pred.device)
            except:
                losses['multi_scale'] = torch.tensor(0.0, device=pred.device)
        else:
            losses['multi_scale'] = torch.tensor(0.0, device=pred.device)
        
        # 加权组合
        total_loss = sum(self.weights[k] * v for k, v in losses.items())
        
        # 最终的数值检查和裁剪
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = losses['mse']  # 回退到最稳定的MSE损失
        
        # 现在所有数据集都归一化到[0,5]范围，使用统一的损失裁剪
        total_loss = torch.clamp(total_loss, 0.0, 20.0)
        
        return total_loss, losses