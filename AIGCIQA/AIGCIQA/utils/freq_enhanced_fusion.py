import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreqEnhancedSpatialFusion(nn.Module):
    """
    频域增强的空间特征融合模块
    
    核心思想：
    - 空间特征作为主干
    - 频域特征作为增强信息，Y通道感知
    - 使用残差连接：enhanced_features = spatial + α * freq
    - 基于AGIQA-3K分析：Y通道高频(+0.395)优先权重
    """
    
    def __init__(self, spatial_dim, freq_dim, hidden_dim=768, 
                 enhancement_weight=0.1, use_adaptive_weight=True):
        super(FreqEnhancedSpatialFusion, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_dim
        self.use_adaptive_weight = use_adaptive_weight
        
        # 特征维度适配层
        if spatial_dim != hidden_dim:
            self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        else:
            self.spatial_proj = nn.Identity()
            
        if freq_dim != hidden_dim:
            self.freq_proj = nn.Linear(freq_dim, hidden_dim)
        else:
            self.freq_proj = nn.Identity()
        
        # Y通道质量感知权重调节器（简化版）
        self.y_quality_enhancer = nn.Linear(1, 1)  # 将Y通道质量指示器转为权重
        
        print(f"Y通道感知融合: 频域特征{freq_dim}d，Y通道质量从频域提取器获取")
        
        # 自适应增强权重网络
        if use_adaptive_weight:
            self.weight_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            print(f"使用自适应频域增强权重网络")
        else:
            # 固定权重
            self.register_parameter('enhancement_weight', 
                                   nn.Parameter(torch.tensor(enhancement_weight)))
            print(f"使用固定频域增强权重: {enhancement_weight}")
        
        # 门控机制，控制频域特征的贡献
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        print(f"FreqEnhancedSpatialFusion初始化: spatial_dim={spatial_dim}, "
              f"freq_dim={freq_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, spatial_features, freq_features, return_weights=False):
        """
        前向传播
        
        Args:
            spatial_features: 空间特征 [B, spatial_dim]
            freq_features: 频域特征 [B, freq_dim]
            return_weights: 是否返回增强权重
            
        Returns:
            enhanced_features: 增强后的图像特征 [B, hidden_dim]
            enhancement_weight: 增强权重 (如果return_weights=True)
        """
        batch_size = spatial_features.size(0)
        
        # 🔍 调试信息：输入特征统计
        debug_mode = hasattr(self, '_debug_fusion') and self._debug_fusion
        if not hasattr(self, '_fusion_call_count'):
            self._fusion_call_count = 0
        self._fusion_call_count += 1
        
        # 只在前5次调用时输出详细调试信息
        show_debug = self._fusion_call_count <= 5 or debug_mode
        
        if show_debug:
            print(f"\n🎯 [第一次融合] FreqEnhancedSpatialFusion 调用 #{self._fusion_call_count}")
            print(f"   输入空间特征: {spatial_features.shape}, 范围[{spatial_features.min():.4f}, {spatial_features.max():.4f}]")
            print(f"   输入频域特征: {freq_features.shape}, 范围[{freq_features.min():.4f}, {freq_features.max():.4f}]")
        
        # 特征投影到统一维度
        spatial_proj = self.spatial_proj(spatial_features)  # [B, hidden_dim]
        freq_proj = self.freq_proj(freq_features)          # [B, hidden_dim]
        
        if show_debug:
            print(f"   投影后空间特征: {spatial_proj.shape}, 范围[{spatial_proj.min():.4f}, {spatial_proj.max():.4f}]")
            print(f"   投影后频域特征: {freq_proj.shape}, 范围[{freq_proj.min():.4f}, {freq_proj.max():.4f}]")
        
        # 获取Y通道质量指示器（如果可用）
        # 尝试从当前模块或父模块获取Y通道质量缓存
        y_quality_indicator = None
        
        # 检查是否有父模型可以提供Y通道质量信息
        # 这里我们使用一个简化的方法：基于频域特征的范数来估计Y通道质量
        freq_norm = torch.norm(freq_proj, dim=1, keepdim=True)  # [B, 1]
        freq_norm_normalized = torch.sigmoid(freq_norm / 10.0)  # 归一化到0-1
        y_quality_indicator = freq_norm_normalized
        
        # Y通道质量感知权重增强
        # 基于AGIQA-3K分析：Y通道高频比例越高，频域特征权重越大
        y_quality_weight = torch.sigmoid(self.y_quality_enhancer(y_quality_indicator))  # [B, 1]
        
        if show_debug:
            print(f"   Y通道质量指示器: 平均={y_quality_indicator.mean():.4f}, 范围[{y_quality_indicator.min():.4f}, {y_quality_indicator.max():.4f}]")
            print(f"   Y通道质量权重: 平均={y_quality_weight.mean():.4f}, 范围[{y_quality_weight.min():.4f}, {y_quality_weight.max():.4f}]")
        
        # 计算增强权重
        if self.use_adaptive_weight:
            # 自适应权重：基于空间和频域特征的相关性
            combined = torch.cat([spatial_proj, freq_proj], dim=1)  # [B, 2*hidden_dim]
            base_enhancement_weight = self.weight_net(combined)      # [B, 1]
            if show_debug:
                print(f"   🔄 使用自适应权重: 平均={base_enhancement_weight.mean():.4f}, 范围[{base_enhancement_weight.min():.4f}, {base_enhancement_weight.max():.4f}]")
        else:
            # 固定权重
            base_enhancement_weight = self.enhancement_weight.expand(batch_size, 1)
            if show_debug:
                print(f"   🔒 使用固定权重: {base_enhancement_weight[0].item():.4f}")
        
        # Y通道感知权重调节：结合基础权重和Y通道质量
        # 当Y通道高频能量高时，增加频域特征权重（最大2倍）
        enhancement_weight = base_enhancement_weight * (1 + y_quality_weight)  # [B, 1]
        
        # 频域特征门控
        freq_gate = self.gate(freq_proj)  # [B, hidden_dim]
        gated_freq = freq_proj * freq_gate
        
        if show_debug:
            print(f"   ⚖️  最终增强权重: 平均={enhancement_weight.mean():.4f}, 范围[{enhancement_weight.min():.4f}, {enhancement_weight.max():.4f}]")
            print(f"   🚪 频域门控值: 平均={freq_gate.mean():.4f}, 范围[{freq_gate.min():.4f}, {freq_gate.max():.4f}]")
            print(f"   🎛️  门控后频域: 平均={gated_freq.mean():.4f}, 范围[{gated_freq.min():.4f}, {gated_freq.max():.4f}]")
        
        # 残差连接：空间特征 + α * 门控频域特征（α由Y通道质量调节）
        enhanced_features = spatial_proj + enhancement_weight * gated_freq
        
        # 层归一化
        enhanced_features = self.layer_norm(enhanced_features)
        
        if show_debug:
            print(f"   ✨ 增强后特征: {enhanced_features.shape}, 范围[{enhanced_features.min():.4f}, {enhanced_features.max():.4f}]")
            print(f"   📊 特征统计: 均值={enhanced_features.mean():.4f}, 标准差={enhanced_features.std():.4f}")
        
        if return_weights:
            return enhanced_features, enhancement_weight.squeeze(-1)
        else:
            return enhanced_features


class ContrastiveFusion(nn.Module):
    """
    图像-文本对比学习融合模块
    
    核心思想：
    - 使用CLIP风格的对比学习机制
    - L2归一化确保特征在单位球面上
    - 可学习的温度参数控制对比学习强度
    """
    
    def __init__(self, image_dim, text_dim, hidden_dim=768, 
                 temperature=0.07, use_learnable_temp=True):
        super(ContrastiveFusion, self).__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.use_learnable_temp = use_learnable_temp
        
        # 特征投影层
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 温度参数
        if use_learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(np.log(1.0 / temperature)))
            print(f"使用可学习温度参数，初始值: {temperature}")
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
            print(f"使用固定温度参数: {temperature}")
        
        # 多模态融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        print(f"ContrastiveFusion初始化: image_dim={image_dim}, "
              f"text_dim={text_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, image_features, text_features, return_similarity=False):
        """
        前向传播
        
        Args:
            image_features: 图像特征 [B, image_dim]
            text_features: 文本特征 [B, text_dim]
            return_similarity: 是否返回相似度矩阵
            
        Returns:
            fused_features: 融合后的特征 [B, hidden_dim]
            similarity_matrix: 相似度矩阵 (如果return_similarity=True)
        """
        batch_size = image_features.size(0)
        
        # 🔍 调试信息：第二次融合
        if not hasattr(self, '_fusion_call_count'):
            self._fusion_call_count = 0
        self._fusion_call_count += 1
        
        # 只在前5次调用时输出详细调试信息
        show_debug = self._fusion_call_count <= 5
        
        if show_debug:
            print(f"\n🎯 [第二次融合] ContrastiveFusion 调用 #{self._fusion_call_count}")
            print(f"   输入图像特征: {image_features.shape}, 范围[{image_features.min():.4f}, {image_features.max():.4f}]")
            print(f"   输入文本特征: {text_features.shape}, 范围[{text_features.min():.4f}, {text_features.max():.4f}]")
        
        # 特征投影
        image_proj = self.image_proj(image_features)  # [B, hidden_dim]
        text_proj = self.text_proj(text_features)     # [B, hidden_dim]
        
        if show_debug:
            print(f"   投影后图像特征: {image_proj.shape}, 范围[{image_proj.min():.4f}, {image_proj.max():.4f}]")
            print(f"   投影后文本特征: {text_proj.shape}, 范围[{text_proj.min():.4f}, {text_proj.max():.4f}]")
        
        # L2归一化，投影到单位球面
        image_norm = F.normalize(image_proj, p=2, dim=1)
        text_norm = F.normalize(text_proj, p=2, dim=1)
        
        if show_debug:
            print(f"   归一化图像特征: 范围[{image_norm.min():.4f}, {image_norm.max():.4f}], 范数={torch.norm(image_norm, dim=1).mean():.4f}")
            print(f"   归一化文本特征: 范围[{text_norm.min():.4f}, {text_norm.max():.4f}], 范数={torch.norm(text_norm, dim=1).mean():.4f}")
        
        # 计算相似度矩阵（用于对比学习）
        if self.use_learnable_temp:
            temp = torch.exp(self.temperature)
        else:
            temp = self.temperature
        
        if show_debug:
            print(f"   🌡️  温度参数: {temp.item():.6f} ({'可学习' if self.use_learnable_temp else '固定'})")
        
        # 图像-文本相似度矩阵
        similarity_matrix = torch.matmul(image_norm, text_norm.t()) / temp  # [B, B]
        
        # 对角线元素表示匹配的图像-文本对的相似度
        matching_similarity = torch.diag(similarity_matrix).unsqueeze(1)  # [B, 1]
        
        if show_debug:
            print(f"   🎯 相似度矩阵: {similarity_matrix.shape}, 范围[{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
            print(f"   💫 匹配相似度: 平均={matching_similarity.mean():.4f}, 范围[{matching_similarity.min():.4f}, {matching_similarity.max():.4f}]")
            
            # 显示相似度矩阵的对角线vs非对角线统计
            diag_values = torch.diag(similarity_matrix)
            off_diag_mask = ~torch.eye(batch_size, dtype=bool, device=similarity_matrix.device)
            off_diag_values = similarity_matrix[off_diag_mask]
            print(f"   📊 对角线相似度(正样本): 平均={diag_values.mean():.4f}, 标准差={diag_values.std():.4f}")
            if len(off_diag_values) > 0:
                print(f"   📊 非对角线相似度(负样本): 平均={off_diag_values.mean():.4f}, 标准差={off_diag_values.std():.4f}")
        
        # 使用相似度作为权重进行特征融合
        # 相似度越高，说明图像-文本对齐越好，给予更高权重
        fusion_weight = torch.sigmoid(matching_similarity)  # [B, 1]
        
        if show_debug:
            print(f"   ⚖️  融合权重: 平均={fusion_weight.mean():.4f}, 范围[{fusion_weight.min():.4f}, {fusion_weight.max():.4f}]")
        
        # 加权融合：结合原始投影特征和归一化特征
        weighted_image = fusion_weight * image_norm + (1 - fusion_weight) * image_proj
        weighted_text = fusion_weight * text_norm + (1 - fusion_weight) * text_proj
        
        if show_debug:
            print(f"   🖼️  加权图像特征: 范围[{weighted_image.min():.4f}, {weighted_image.max():.4f}]")
            print(f"   📝 加权文本特征: 范围[{weighted_text.min():.4f}, {weighted_text.max():.4f}]")
        
        # 最终融合
        combined = torch.cat([weighted_image, weighted_text], dim=1)  # [B, 2*hidden_dim]
        fused_features = self.fusion_net(combined)  # [B, hidden_dim]
        
        if show_debug:
            print(f"   🔗 拼接特征: {combined.shape}, 范围[{combined.min():.4f}, {combined.max():.4f}]")
            print(f"   ✨ 最终融合特征: {fused_features.shape}, 范围[{fused_features.min():.4f}, {fused_features.max():.4f}]")
            print(f"   📊 最终统计: 均值={fused_features.mean():.4f}, 标准差={fused_features.std():.4f}")
        
        if return_similarity:
            return fused_features, similarity_matrix
        else:
            return fused_features
    
    def compute_contrastive_loss(self, image_features, text_features):
        """
        计算对比学习损失
        
        Args:
            image_features: 图像特征 [B, image_dim]
            text_features: 文本特征 [B, text_dim]
            
        Returns:
            contrastive_loss: 对比学习损失
        """
        _, similarity_matrix = self.forward(image_features, text_features, return_similarity=True)
        batch_size = similarity_matrix.size(0)
        
        # 创建标签：对角线为正样本
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # 图像到文本的交叉熵损失
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        
        # 文本到图像的交叉熵损失
        loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
        
        # 总对比学习损失
        contrastive_loss = (loss_i2t + loss_t2i) / 2
        
        return contrastive_loss


class TwoStageMultiModalFusion(nn.Module):
    """
    两阶段多模态融合的完整模块
    
    整合FreqEnhancedSpatialFusion和ContrastiveFusion
    """
    
    def __init__(self, spatial_dim, freq_dim, text_dim, hidden_dim=768,
                 enhancement_weight=0.1, temperature=0.07,
                 use_adaptive_weight=True, use_learnable_temp=True):
        super(TwoStageMultiModalFusion, self).__init__()
        
        # 第一阶段：频域增强空间特征融合
        self.stage1_fusion = FreqEnhancedSpatialFusion(
            spatial_dim=spatial_dim,
            freq_dim=freq_dim,
            hidden_dim=hidden_dim,
            enhancement_weight=enhancement_weight,
            use_adaptive_weight=use_adaptive_weight
        )
        
        # 第二阶段：图像-文本对比学习融合
        self.stage2_fusion = ContrastiveFusion(
            image_dim=hidden_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            temperature=temperature,
            use_learnable_temp=use_learnable_temp
        )
        
        print(f"TwoStageMultiModalFusion初始化完成")
        print(f"第一阶段: 空间({spatial_dim}) + 频域({freq_dim}) -> 图像({hidden_dim})")
        print(f"第二阶段: 图像({hidden_dim}) + 文本({text_dim}) -> 融合({hidden_dim})")
    
    def forward(self, spatial_features, freq_features, text_features, 
                return_intermediate=False, return_loss=False):
        """
        两阶段前向传播
        
        Args:
            spatial_features: 空间特征 [B, spatial_dim]
            freq_features: 频域特征 [B, freq_dim]
            text_features: 文本特征 [B, text_dim]
            return_intermediate: 是否返回中间结果
            return_loss: 是否计算对比学习损失
            
        Returns:
            final_features: 最终融合特征 [B, hidden_dim]
            intermediate_results: 中间结果字典 (可选)
            contrastive_loss: 对比学习损失 (可选)
        """
        # 🔍 调试信息：两阶段融合总览
        if not hasattr(self, '_fusion_call_count'):
            self._fusion_call_count = 0
        self._fusion_call_count += 1
        
        show_debug = self._fusion_call_count <= 3  # 只在前3次调用时显示详细信息
        
        if show_debug:
            print(f"\n🎯 [两阶段融合] TwoStageMultiModalFusion 调用 #{self._fusion_call_count}")
            print(f"   📥 输入空间特征: {spatial_features.shape}")
            print(f"   📥 输入频域特征: {freq_features.shape}")  
            print(f"   📥 输入文本特征: {text_features.shape}")
            print(f"   🔄 return_intermediate: {return_intermediate}, return_loss: {return_loss}")
        
        # 第一阶段：频域增强空间特征融合
        if show_debug:
            print(f"   🚀 开始第一阶段: FreqEnhancedSpatialFusion")
            
        enhanced_image_features, enhancement_weight = self.stage1_fusion(
            spatial_features, freq_features, return_weights=True
        )
        
        if show_debug:
            print(f"   ✅ 第一阶段完成: {enhanced_image_features.shape}")
            print(f"   📊 增强权重统计: 平均={enhancement_weight.mean():.4f}, 范围[{enhancement_weight.min():.4f}, {enhancement_weight.max():.4f}]")
        
        # 第二阶段：图像-文本对比学习融合
        if show_debug:
            print(f"   🚀 开始第二阶段: ContrastiveFusion")
            
        if return_loss:
            final_features, similarity_matrix = self.stage2_fusion(
                enhanced_image_features, text_features, return_similarity=True
            )
            contrastive_loss = self.stage2_fusion.compute_contrastive_loss(
                enhanced_image_features, text_features
            )
            if show_debug:
                print(f"   📊 对比学习损失: {contrastive_loss.item():.6f}")
        else:
            final_features = self.stage2_fusion(enhanced_image_features, text_features)
            contrastive_loss = None
            similarity_matrix = None
        
        if show_debug:
            print(f"   ✅ 第二阶段完成: {final_features.shape}")
            print(f"   🎉 两阶段融合完成! 最终特征: 均值={final_features.mean():.4f}, 标准差={final_features.std():.4f}")
        
        # 准备返回结果
        results = [final_features]
        
        if return_intermediate:
            intermediate_results = {
                'enhanced_image_features': enhanced_image_features,
                'enhancement_weight': enhancement_weight,
                'similarity_matrix': similarity_matrix
            }
            results.append(intermediate_results)
        
        if return_loss and contrastive_loss is not None:
            results.append(contrastive_loss)
        
        return results[0] if len(results) == 1 else tuple(results) 