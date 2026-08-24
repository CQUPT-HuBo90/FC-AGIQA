import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from utils.enhanced_frequency_extractor import EnhancedFrequencyFeatureExtractor
import math
from typing import Optional, Tuple

# 导入CLIP-ConvNeXt编码器
try:
    from backbone.clip_convnext_encoder import get_clip_convnext_encoder
    CLIPCONVNEXT_AVAILABLE = True
except ImportError:
    CLIPCONVNEXT_AVAILABLE = False
    print("警告: 未找到CLIP-ConvNeXt编码器，请确保已安装open_clip_torch")

# 导入CLIP-ConvNeXt文本编码器
try:
    from backbone.clip_convnext_text_encoder import get_clip_convnext_text_encoder
    CLIPCONVNEXT_TEXT_AVAILABLE = True
except ImportError:
    CLIPCONVNEXT_TEXT_AVAILABLE = False
    print("警告: 未找到CLIP-ConvNeXt文本编码器，请确保已安装open_clip_torch")

# 导入高级多模态融合模块
try:
    from utils.advanced_multimodal_fusion import (
        CrossModalTransformer, 
        DynamicFeatureFusion, 
        GatedMultiModalFusion
    )
    ENHANCED_FUSION_AVAILABLE = True
except ImportError:
    ENHANCED_FUSION_AVAILABLE = False
    print("警告: 未找到高级多模态融合模块，将使用原始版本")

# 导入频域增强融合模块
try:
    from utils.freq_enhanced_fusion import TwoStageMultiModalFusion
    FREQ_ENHANCED_FUSION_AVAILABLE = True
except ImportError:
    FREQ_ENHANCED_FUSION_AVAILABLE = False
    print("警告: 未找到频域增强融合模块，将使用原始版本")

# 改进的平均池化模块
class EnhancedMeanPooling(nn.Module):
    def __init__(self):
        super(EnhancedMeanPooling, self).__init__()
    
    def forward(self, last_hidden_state, attention_mask):
        """
        改进的平均池化层，处理维度不匹配问题
        
        参数:
            last_hidden_state: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        返回:
            pooled_output: [batch_size, hidden_size]
        """
        # 确保输入维度正确
        if last_hidden_state.dim() == 4 and last_hidden_state.size(1) == 1:
            # 处理某些模型可能会返回的额外维度: [batch, 1, seq_len, dim]
            last_hidden_state = last_hidden_state.squeeze(1)
        
        if attention_mask.dim() > 2:
            # 确保注意力掩码是2D [batch_size, seq_len]
            attention_mask = attention_mask.squeeze(1)
        
        # 确保维度匹配
        if attention_mask.size(0) != last_hidden_state.size(0):
            raise ValueError(f"批次大小不匹配: hidden_state={last_hidden_state.size(0)}, mask={attention_mask.size(0)}")
        
        # 根据注意力掩码进行加权平均
        attention_mask = attention_mask.float().unsqueeze(-1)
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        
        # 返回加权平均后的特征
        return sum_embeddings / sum_mask


class FrequencyEnhancedCLIPImageEncoder(nn.Module):
    """
    频域增强的CLIP图像编码器
    核心思想：将频域特征融合到CLIP图像特征中，而不是作为独立模态
    利用CLIP图像和文本编码器天生在同一空间的优势

    消融实验支持:
    - disable_freq_fusion=True: 禁用频域融合，直接返回CLIP特征（用于消融实验）
    """

    def __init__(
        self,
        clip_image_encoder,
        freq_feature_dim: int = 768,
        enhancement_method: str = "adaptive_fusion",  # "residual", "adaptive_fusion", "cross_attention"
        dropout_rate: float = 0.1,
        disable_freq_fusion: bool = False  # [消融] 禁用频域-CLIP融合
    ):
        super().__init__()

        self.clip_encoder = clip_image_encoder
        self.freq_feature_dim = freq_feature_dim
        self.enhancement_method = enhancement_method
        self.disable_freq_fusion = disable_freq_fusion  # 消融标志

        # 获取CLIP图像特征维度
        self.clip_dim = self.clip_encoder.get_feature_dim()

        # [消融] 如果禁用频域融合，跳过频域相关模块的初始化
        if self.disable_freq_fusion:
            print("[消融] FrequencyEnhancedCLIPImageEncoder: 禁用频域融合，直接使用CLIP特征")
            self.frequency_extractor = None
            self.fusion = None
            return

        # 频域特征提取器
        self.frequency_extractor = EnhancedFrequencyFeatureExtractor(
            output_dim=freq_feature_dim,
            num_scales=4,
            num_orientations=8
        )

        # 频域特征融合模块
        if enhancement_method == "residual":
            # 简单残差连接
            self.freq_adapter = nn.Sequential(
                nn.Linear(freq_feature_dim, self.clip_dim),
                nn.LayerNorm(self.clip_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.clip_dim, self.clip_dim)
            )
        elif enhancement_method == "adaptive_fusion":
            # 自适应加权融合
            self.fusion = AdaptiveFrequencyFusion(
                clip_dim=self.clip_dim,
                freq_dim=freq_feature_dim,
                hidden_dim=self.clip_dim // 2
            )
        elif enhancement_method == "cross_attention":
            # 交叉注意力融合
            self.fusion = FrequencyCrossAttention(
                clip_dim=self.clip_dim,
                freq_dim=freq_feature_dim,
                num_heads=8,
                dropout=dropout_rate
            )
        else:
            raise ValueError(f"Unknown enhancement method: {enhancement_method}")

        print(f"FrequencyEnhancedCLIPImageEncoder初始化:")
        print(f"  CLIP维度: {self.clip_dim}")
        print(f"  频域维度: {freq_feature_dim}")
        print(f"  增强方法: {enhancement_method}")
        print(f"  频域融合: {'启用' if not disable_freq_fusion else '禁用(消融)'}")
    
    def get_feature_dim(self):
        """返回输出特征维度"""
        return self.clip_dim
    
    def forward(self, image, raw_image=None):
        """
        前向传播：用频域特征增强CLIP图像特征

        Args:
            image: 输入图像 [batch_size, 3, H, W]，用于CLIP主干
            raw_image: 反归一化后的原始RGB图像 [batch_size, 3, H, W]，用于频域分支

        Returns:
            enhanced_features: 频域增强的CLIP图像特征 [batch_size, clip_dim]
        """
        # 1. 提取CLIP图像特征
        clip_features = self.clip_encoder(image)  # [B, clip_dim]

        # [消融] 如果禁用频域融合，直接返回CLIP特征
        if self.disable_freq_fusion:
            return clip_features

        # 2. 提取频域特征
        frequency_image = raw_image if raw_image is not None else image
        freq_features = self.frequency_extractor(frequency_image)  # [B, freq_dim]

        # 3. 融合频域和CLIP特征
        if self.enhancement_method == "residual":
            # 残差连接方式
            freq_adapted = self.freq_adapter(freq_features)
            enhanced_features = clip_features + freq_adapted
        elif self.enhancement_method == "adaptive_fusion":
            # 自适应融合
            enhanced_features = self.fusion(clip_features, freq_features)
        elif self.enhancement_method == "cross_attention":
            # 交叉注意力融合
            enhanced_features = self.fusion(clip_features, freq_features)

        return enhanced_features




class AdaptiveFrequencyFusion(nn.Module):
    def __init__(
        self,
        clip_dim: int,
        freq_dim: int,
        hidden_dim: int,
        num_freq_tokens: int = 4,
        attn_dropout: float = 0.05,   # 轻微 Dropout
        tau_init: float = 10.0        # 可学习温度初值
    ):
        super().__init__()
        self.num_freq_tokens = num_freq_tokens
        self.clip_dim = clip_dim

        # 投影
        self.freq_token_proj = nn.Linear(freq_dim, clip_dim * num_freq_tokens)

        # 注意力相关（仅做你要的两点）
        self.q_ln  = nn.LayerNorm(clip_dim)  # Q 预层归一
        self.k_ln  = nn.LayerNorm(clip_dim)  # K 预层归一
        self.q_proj = nn.Linear(clip_dim, clip_dim)
        self.k_proj = nn.Linear(clip_dim, clip_dim)
        self.v_proj = nn.Linear(clip_dim, clip_dim)
        self.ca_out_ln = nn.LayerNorm(clip_dim)

        self.tau = nn.Parameter(torch.tensor(tau_init))   # 可学习温度
        self.attn_dropout = nn.Dropout(attn_dropout)      # 轻微 Dropout

        # 向量门控（最后一层置零 → Sigmoid 后起步 0.5）
        self.weight_net = nn.Sequential(
            nn.Linear(2 * clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, clip_dim),
            nn.Sigmoid()
        )
        nn.init.zeros_(self.weight_net[-2].weight)
        nn.init.zeros_(self.weight_net[-2].bias)

        # 增强网络（保持不变）
        self.enhancement_net = nn.Sequential(
            nn.Linear(2 * clip_dim, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, clip_dim)
        )

    # --- 仅封装注意力 ---
    def _cross_attend(self, clip_features: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        """
        clip←freq 单向注意力：
          - Q/K 预层归一（LayerNorm）
          - 余弦相似度（L2 normalize）
          - 可学习温度 tau
          - 轻微 Dropout
        """
        B, T, C = freq_tokens.shape

        Q = self.q_proj(self.q_ln(clip_features)).unsqueeze(1)  # [B,1,C]
        K = self.k_proj(self.k_ln(freq_tokens))                  # [B,T,C]
        V = self.v_proj(freq_tokens)                             # [B,T,C]（V 不必归一）

        Qn = F.normalize(Q, dim=-1)
        Kn = F.normalize(K, dim=-1)

        scores = torch.matmul(Qn, Kn.transpose(1, 2)) * (self.tau / math.sqrt(C))  # [B,1,T]
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, V).squeeze(1)  # [B,C]
        return self.ca_out_ln(context)

    # --- 主流程：只调用函数 + 简单加减乘除 ---
    def forward(self, clip_features, freq_features):
        B, C = clip_features.shape
        freq_tokens = self.freq_token_proj(freq_features).view(B, self.num_freq_tokens, C)   # [B,T,C]
        freq_projected = self._cross_attend(clip_features, freq_tokens)                      # [B,C]

        fusion_weight = self.weight_net(torch.cat([clip_features, freq_projected], dim=1))   # [B,C]
        fused = fusion_weight * clip_features + (1 - fusion_weight) * freq_projected         # [B,C]

        enhanced = self.enhancement_net(torch.cat([clip_features, fused], dim=1))            # [B,C]
        return enhanced

class FrequencyCrossAttention(nn.Module):
    """
    频域-CLIP交叉注意力融合模块
    让频域特征作为query，CLIP特征作为key/value进行注意力计算
    """
    
    def __init__(self, clip_dim: int, freq_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.clip_dim = clip_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.head_dim = clip_dim // num_heads
        
        assert clip_dim % num_heads == 0, "clip_dim must be divisible by num_heads"
        
        # 频域特征投影为query
        self.freq_to_q = nn.Linear(freq_dim, clip_dim)
        
        # CLIP特征作为key和value
        self.clip_to_k = nn.Linear(clip_dim, clip_dim)
        self.clip_to_v = nn.Linear(clip_dim, clip_dim)
        
        # 输出投影
        self.output_proj = nn.Linear(clip_dim, clip_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(clip_dim)
        
    def forward(self, clip_features, freq_features):
        """
        Args:
            clip_features: CLIP图像特征 [B, clip_dim] 
            freq_features: 频域特征 [B, freq_dim]
            
        Returns:
            enhanced_features: 增强后的图像特征 [B, clip_dim]
        """
        batch_size = clip_features.size(0)
        
        # 1. 计算query, key, value
        q = self.freq_to_q(freq_features)  # [B, clip_dim]
        k = self.clip_to_k(clip_features)   # [B, clip_dim]
        v = self.clip_to_v(clip_features)   # [B, clip_dim]
        
        # 2. reshape为多头注意力格式
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
        
        # 3. 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, 1, 1]
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, 1, D]
        
        # 4. 合并多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.clip_dim)
        
        # 5. 输出投影和残差连接
        enhanced_features = self.output_proj(attn_output)
        enhanced_features = self.layer_norm(clip_features + enhanced_features)
        
        return enhanced_features


# 多头注意力融合机制
class MultiHeadAttentionFusion(nn.Module):
    """多头注意力机制，使用PyTorch原生实现，增强文本特征对图像特征的引导作用"""
    def __init__(self, text_dim, img_dim, num_heads=16, proj_dim=768):
        super(MultiHeadAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        
        # 确保投影维度可被头数整除
        assert self.proj_dim % num_heads == 0, "proj_dim必须能被num_heads整除"
        
        # 文本和图像特征投影层
        self.text_proj = nn.Linear(text_dim, proj_dim)  # 文本特征投影
        self.img_proj = nn.Linear(img_dim, proj_dim)    # 图像特征投影
        
        # 使用PyTorch原生的多头注意力
        self.mha = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(proj_dim)
        
        print(f"初始化原生MultiHeadAttentionFusion: text_dim={text_dim}, img_dim={img_dim}, proj_dim={proj_dim}, num_heads={num_heads}")
        
    def forward(self, text_features, img_features, batch_count=None, epoch=None, dataset_name=None):
        # 只在第一个epoch的第一个批次输出调试信息
        is_first_batch = batch_count is not None and batch_count == 0 and (epoch is not None and epoch == 0)
        
        # 打印输入维度信息，帮助调试
        batch_size_text = text_features.size(0)
        batch_size_img = img_features.size(0)
        
        if is_first_batch:
            print(f"MultiHeadAttention输入: text={text_features.shape}, img={img_features.shape}")
        
        # 确保批次大小一致
        if batch_size_text != batch_size_img:
            if is_first_batch:
                print(f"批次大小不匹配: text={batch_size_text}, img={batch_size_img}")
            
            # 处理批次大小不匹配的情况
            if batch_size_text == 1 and batch_size_img > 1:
                # 单个文本特征，多个图像特征 
                text_features = text_features.expand(batch_size_img, -1)
                if is_first_batch:
                    print(f"扩展文本特征至: {text_features.shape}")
            elif batch_size_img == 1 and batch_size_text > 1:
                # 单个图像特征，多个文本特征
                img_features = img_features.expand(batch_size_text, -1)
                if is_first_batch:
                    print(f"扩展图像特征至: {img_features.shape}")
        
        # 特征投影
        text_proj = self.text_proj(text_features)  # [batch_size, proj_dim]
        img_proj = self.img_proj(img_features)     # [batch_size, proj_dim]
        
        if is_first_batch:
            print(f"投影后: text_proj={text_proj.shape}, img_proj={img_proj.shape}")
        
        # 统一使用标准注意力机制：每个样本作为1个token，避免2D输入被当作跨batch序列
        fusion_features, _ = self.mha(
            text_proj.unsqueeze(1),
            img_proj.unsqueeze(1),
            img_proj.unsqueeze(1)
        )
        fusion_features = fusion_features.squeeze(1)

        # 残差连接和层归一化
        fusion_features = self.layer_norm(fusion_features + text_proj)  # 简单残差连接

        return fusion_features


class Encoder(nn.Module):
    """改进的编码器模块，整合空间和频域特征，使用分层特征融合与门控机制

    消融实验支持:
    - ablation_mode='baseline': CLIP + 简单注意力融合 + MLP（无频域，无MFB）
    - ablation_mode='no_freq_encoder': 移除频域编码器，保留MFB融合
    - ablation_mode='no_freq_clip_fusion': 保留频域编码器，禁用频域-CLIP融合
    - ablation_mode='no_mfb': 保留频域编码器，禁用MFB双线性池化
    """
    def __init__(self, image_encoder, model_path="./bert-base-uncased", freq_feature_dim=768,
                 use_frequency_features=True, num_heads=16, use_enhanced_fusion=False,
                 fusion_type='standard', dataset_name=None, use_freq_enhanced_fusion=False,
                 freq_enhancement_weight=0.2, contrastive_temperature=0.07,
                 ablation_mode='full', disable_freq_clip_fusion=False, disable_mfb_fusion=False):
        super(Encoder, self).__init__()

        # 保存dataset_name用于任务特定的处理
        self.dataset_name = dataset_name

        # ========== 消融实验配置（控制变量法设计） ==========
        # full为基准，每次只移除一个模块
        self.ablation_mode = ablation_mode
        self.disable_mfb = disable_mfb_fusion

        # 根据消融模式调整参数
        if ablation_mode == 'baseline':
            # Baseline: 纯CLIP + 简单注意力融合
            use_frequency_features = False
            use_freq_enhanced_fusion = False
            disable_freq_clip_fusion = True
            self.disable_mfb = True
            print("[消融] Baseline: 纯CLIP + 直接特征相加")
        elif ablation_mode == 'freq_encoder_only':
            # +频域编码器: CLIP + 频域(独立特征) + 简单注意力融合
            use_frequency_features = True
            use_freq_enhanced_fusion = True  # 使用传统三模态融合路径
            disable_freq_clip_fusion = True  # 禁用Freq-CLIP融合，频域作为独立特征
            self.disable_mfb = True
            print("[消融] +FreqEncoder: CLIP + 频域编码器(独立) + 直接特征相加")
        elif ablation_mode == 'freq_clip_fusion':
            # +Freq-CLIP融合: CLIP + 频域融合到CLIP + 简单注意力融合
            use_frequency_features = True
            use_freq_enhanced_fusion = True
            disable_freq_clip_fusion = False  # 启用Freq-CLIP融合
            self.disable_mfb = True  # 还没有MFB
            print("[消融] +FreqCLIPFusion: CLIP(融合频域) + 直接特征相加")
        elif ablation_mode == 'full':
            # Full: CLIP + 频域融合 + MFB
            use_frequency_features = True
            use_freq_enhanced_fusion = True
            disable_freq_clip_fusion = False
            self.disable_mfb = False
            print("[完整模型] Full: CLIP + 频域融合 + MFB双线性池化")
        # ========== 控制变量法消融模式 ==========
        elif ablation_mode == 'no_freq_encoder':
            # 移除频域编码器：保留其他所有模块
            use_frequency_features = False  # 禁用频域特征
            use_freq_enhanced_fusion = False  # 禁用频域增强融合（避免引入替代融合）
            disable_freq_clip_fusion = True  # 没有频域特征就无法融合
            self.disable_mfb = False  # 保留MFB
            print("[消融-控制变量] no_freq_encoder: 移除频域编码器")
        elif ablation_mode == 'no_freq_clip_fusion':
            # 移除AFF：频域作为独立特征concat，保留MFB图文融合
            use_frequency_features = True
            use_freq_enhanced_fusion = False  # 不需要两阶段融合
            disable_freq_clip_fusion = True  # 禁用AFF
            self.disable_mfb = False  # 保留MFB
            print("[消融-控制变量] no_freq_clip_fusion: 移除AFF，频域独立concat，保留MFB")
        elif ablation_mode == 'no_mfb':
            # 移除MFB：保留频域编码器和融合
            use_frequency_features = True
            use_freq_enhanced_fusion = True
            disable_freq_clip_fusion = False
            self.disable_mfb = True  # 禁用MFB
            print("[消融-控制变量] no_mfb: 移除MFB双线性池化")
        else:
            print(f"[警告] 未知的ablation_mode={ablation_mode}，使用完整模型")

        self.disable_freq_clip_fusion = disable_freq_clip_fusion
        # --- MFB (双线性池化) 所需的层 ---
        # MFB 参数 (k_factor=5, output_dim=768)
        self.mfb_k_factor = 5
        self.mfb_output_dim = 768 # 保持与原特征维度一致
        k_dim = self.mfb_output_dim * self.mfb_k_factor # 768 * 5 = 3840
        
        # 1. 扩展层 (Expand) [1]
        self.mfb_img_expand = nn.Linear(768, k_dim)
        self.mfb_txt_expand = nn.Linear(768, k_dim)
        
        self.mfb_dropout = nn.Dropout(0.1)
        
        # 2. 压缩层 (Squeeze) [1]
        self.mfb_pool = nn.AvgPool1d(kernel_size=self.mfb_k_factor)

        # 初始化文本编码器 - 支持CLIPConvNeXt文本编码器
        if model_path == "clip_convnext" and CLIPCONVNEXT_TEXT_AVAILABLE:
            print("使用CLIP-ConvNeXt文本编码器")
            self.text_encoder = get_clip_convnext_text_encoder(
                model_name='convnext_large_d_320',
                pretrained='laion2b_s29b_b131k_ft',
                output_dim=768,
                freeze_encoder=False,
                use_wrapper=True,
                max_length=77
            )
            # 直接使用CLIPConvNeXt的配置
            self.config = type('Config', (), {'hidden_size': 768})()
            self.is_clip_model = True
        else:
            # 使用传统的transformer文本编码器
            self.text_encoder = AutoModel.from_pretrained(model_path)
            # 获取配置
            self.config = AutoConfig.from_pretrained(model_path)
            self.is_clip_model = False
        
        # 优化的图像编码器初始化 - 基于CLIP优势的频域增强
        # [消融] 根据消融模式决定是否使用频域增强
        if self.is_clip_model and use_frequency_features and hasattr(image_encoder, 'get_feature_dim'):
            # 当使用CLIP文本编码器时，将频域特征直接融合到CLIP图像编码器中
            print("🚀 检测到CLIP模型，使用频域增强的CLIP图像编码器")
            self.image_encoder = FrequencyEnhancedCLIPImageEncoder(
                clip_image_encoder=image_encoder,
                freq_feature_dim=freq_feature_dim,
                enhancement_method="adaptive_fusion",  # 使用自适应融合方法
                dropout_rate=0.1,
                disable_freq_fusion=disable_freq_clip_fusion  # [消融] 传递消融参数
            )
            if disable_freq_clip_fusion:
                print("[消融] 频域-CLIP融合已禁用，直接使用CLIP特征")
            else:
                print("✅ 频域特征将直接增强CLIP图像表示，充分利用CLIP语义空间")
        else:
            # 传统方式：图像编码器和频域特征分别处理
            self.image_encoder = image_encoder
            print("📊 使用传统的独立图像编码器和频域特征融合方式")
        
        # 使用改进的池化层
        self.pooler = EnhancedMeanPooling()
        
        # 获取特征维度
        self.txt_dim = self.config.hidden_size
        self.img_dim = self._get_image_dim()
        # 使用原始特征维度，取消投影操作
        self.proj_dim = self.txt_dim  # 统一使用文本编码器的维度(768)
        
        # 移除投影层 - 直接使用原始特征维度进行融合
        # self.text_projection = nn.Linear(self.txt_dim, self.proj_dim)  # 已移除
        # self.image_projection = nn.Linear(self.img_dim, self.proj_dim)  # 已移除
        
        # 添加图像特征维度适配层（仅在图像特征维度与文本不同时使用）
        if self.img_dim != self.txt_dim:
            print(f"图像特征维度({self.img_dim})与文本特征维度({self.txt_dim})不同，添加适配层")
            self.image_adapter = nn.Linear(self.img_dim, self.txt_dim)
        else:
            print(f"图像和文本特征维度均为{self.txt_dim}，无需适配层")
            self.image_adapter = None
        
        # 是否使用频域特征
        self.use_frequency_features = use_frequency_features
        self.use_enhanced_fusion = use_enhanced_fusion and ENHANCED_FUSION_AVAILABLE
        self.use_freq_enhanced_fusion = use_freq_enhanced_fusion and FREQ_ENHANCED_FUSION_AVAILABLE
        self.fusion_type = fusion_type
        self.freq_feature_dim = freq_feature_dim
        
        # 初始化增强版频域特征提取器
        if self.use_frequency_features:
            print("使用增强版频域特征提取器")
            self.frequency_extractor = EnhancedFrequencyFeatureExtractor(
                output_dim=freq_feature_dim,
                num_scales=4,
                num_orientations=8
            )
        
        print(f"文本特征维度: {self.txt_dim}, 图像特征维度: {self.img_dim}, 投影维度: {self.proj_dim}, 频域特征维度: {self.freq_feature_dim if self.use_frequency_features else 0}")
        
        # 选择融合方法
        if self.use_freq_enhanced_fusion and self.use_frequency_features:
            # 使用两阶段频域增强融合
            print("使用两阶段频域增强融合方案")
            self.two_stage_fusion = TwoStageMultiModalFusion(
                spatial_dim=self.proj_dim,      # 空间特征维度
                freq_dim=freq_feature_dim,      # 频域特征维度
                text_dim=self.proj_dim,         # 文本特征维度
                hidden_dim=self.proj_dim,       # 统一隐藏维度
                enhancement_weight=freq_enhancement_weight,
                temperature=contrastive_temperature,
                use_adaptive_weight=True,
                use_learnable_temp=True
            )

        elif self.use_freq_enhanced_fusion and not self.use_frequency_features:
            # 频域特征关闭但仍启用频域增强融合时，退化为图文对比融合
            print("初始化独立的ContrastiveFusion模块 (无频域特征模式)")
            from utils.freq_enhanced_fusion import ContrastiveFusion
            self.contrastive_fusion = ContrastiveFusion(
                image_dim=self.proj_dim,
                text_dim=self.proj_dim,
                hidden_dim=self.proj_dim
            )
        elif self.use_enhanced_fusion:
            print(f"使用高级多模态融合: {self.fusion_type}")
            if self.fusion_type == 'transformer':
                # 使用跨模态Transformer融合
                self.enhanced_fusion = CrossModalTransformer(
                    d_model=self.proj_dim,
                    n_heads=8,
                    n_layers=3,
                    dropout=0.1
                )
            elif self.fusion_type == 'dynamic':
                # 使用动态特征融合
                self.enhanced_fusion = DynamicFeatureFusion(
                    text_dim=self.proj_dim,
                    img_dim=self.proj_dim,
                    freq_dim=self.freq_feature_dim,
                    hidden_dim=self.proj_dim
                )
            elif self.fusion_type == 'gated':
                # 使用门控多模态融合
                modal_dims = [self.proj_dim, self.proj_dim]
                if self.use_frequency_features:
                    modal_dims.append(self.freq_feature_dim)
                self.enhanced_fusion = GatedMultiModalFusion(
                    modal_dims=modal_dims,
                    output_dim=self.proj_dim
                )
            else:
                print(f"未知的融合类型: {self.fusion_type}, 使用标准融合")
                self.use_enhanced_fusion = False
        
        if not self.use_enhanced_fusion:
            # 使用标准的多头注意力机制
            print("使用标准多头注意力融合")
            self.attention = MultiHeadAttentionFusion(self.proj_dim, self.proj_dim, num_heads=num_heads, proj_dim=self.proj_dim)
        
        # 分层特征融合和门控机制
        if self.use_frequency_features:
            # 文本-图像门控机制 - 使用统一后的768维特征
            self.text_image_gate = nn.Sequential(
                nn.Linear(self.proj_dim * 2, 256),  # 768*2 -> 256
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
            
            # 频域门控机制 - 频域特征现在也是768维
            self.freq_gate = nn.Sequential(
                nn.Linear(self.proj_dim + freq_feature_dim, 256),  # 768+768 -> 256
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
            # 可视化提取器（可选，用于分析不同模态的贡献）
            self.register_buffer('gate_values', torch.zeros(2))
            self.register_buffer('gate_counts', torch.zeros(2))
        
        # 全局池化层，用于处理Swin Transformer等模型的输出
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _get_image_dim(self):
        """根据图像编码器类型确定输出维度"""
        backbone_name = self.image_encoder.__class__.__name__.lower()
        
        # 首先检查是否有get_feature_dim方法（新的CLIP-ConvNeXt编码器）
        if hasattr(self.image_encoder, 'get_feature_dim'):
            return self.image_encoder.get_feature_dim()
        
        # 根据backbone类型返回合适的维度
        if 'clipconvnext' in backbone_name:
            # CLIP-ConvNeXt编码器 - 实际输出是768维
            return 768  # 默认值，实际会通过get_feature_dim获取
        elif 'resnet18' in backbone_name or 'resnet34' in backbone_name:
            return 512
        elif 'resnet' in backbone_name:  # ResNet50/101/152
            return 2048
        elif 'inception' in backbone_name:
            return 1536
        elif 'swin' in backbone_name:
            return 1024
        elif 'vit' in backbone_name:
            return 1024
        else:
            # 默认值
            return 512

    def _get_frequency_image(self, image):
        """将CLIP/ImageNet归一化后的输入还原到[0,1] RGB，供YCbCr/FFT频域分支使用。"""
        if self.is_clip_model:
            mean = image.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std = image.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        else:
            mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (image * std + mean).clamp(0.0, 1.0)

    def forward(self, image, text_ids, attention_mask, batch_count=None, epoch=None):
        """
        前向传播
        
        参数:
            image: 图像输入 [batch_size, channels, height, width]
            text_ids: 文本token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            batch_count: 当前epoch中的批次计数，用于控制调试输出
            epoch: 当前epoch计数，用于控制调试输出
            
        返回:
            fusion_features: 融合后的特征
            text_features: 文本特征 [batch_size, txt_dim]
            image_features: 图像特征 [batch_size, img_dim]
        """
        # 只在第一个epoch的第一个批次输出调试信息
        is_first_batch = batch_count is not None and batch_count == 0 and (epoch is not None and epoch == 0)
        
        # 确保输入维度正确
        if text_ids.dim() > 2:
            text_ids = text_ids.squeeze(1)
        if attention_mask.dim() > 2:
            attention_mask = attention_mask.squeeze(1)
        
        # 文本编码 - 支持不同类型的文本编码器
        if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'preprocess_text'):
            # CLIPConvNeXt文本编码器
            # 直接使用来自数据加载器的token
            text_outputs = self.text_encoder(text_ids, attention_mask)
            
            # CLIPConvNeXt文本编码器返回的是包装后的输出对象
            text_features_orig = text_outputs.last_hidden_state.squeeze(1)  # 移除序列维度
        else:
            # 传统的transformer文本编码器
            text_outputs = self.text_encoder(
                input_ids=text_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            # 提取文本特征
            text_features_orig = self.pooler(text_outputs.last_hidden_state, attention_mask)
        
        # 频域分支需要[0,1] RGB图像；CLIP/主干仍使用归一化后的输入
        frequency_image = self._get_frequency_image(image) if self.use_frequency_features else None

        # 图像编码（空间域特征）
        if isinstance(self.image_encoder, FrequencyEnhancedCLIPImageEncoder):
            image_features_orig = self.image_encoder(image, raw_image=frequency_image)
        else:
            image_features_orig = self.image_encoder(image)

        # 处理不同维度的图像特征输出
        if image_features_orig.dim() > 2:
            if image_features_orig.dim() == 4:  # [B, C, H, W]
                image_features_orig = self.global_pool(image_features_orig).flatten(1)
            else:  # [B, L, C]
                image_features_orig = image_features_orig.mean(dim=1)
        
        # 确保批次大小一致
        batch_size = text_features_orig.size(0)
        if image_features_orig.size(0) != batch_size:
            raise ValueError(f"批次大小不匹配: text={batch_size}, image={image_features_orig.size(0)}")
        
        # 直接使用原始特征，不进行投影操作
        text_features = text_features_orig  # 直接使用768维文本特征
        
        # 如果图像特征维度与文本不同，使用适配层统一维度
        if self.image_adapter is not None:
            image_features = self.image_adapter(image_features_orig)
        else:
            image_features = image_features_orig  # 直接使用原始图像特征
        
        # 打印特征维度，辅助调试 - 只在第一个epoch的第一个批次
        if is_first_batch:
            print(f"原始特征维度: text={text_features_orig.shape}, image={image_features_orig.shape}")
            print(f"投影后特征维度: text={text_features.shape}, image={image_features.shape}")
            
        # 🚀 优化的CLIP空间融合流程
        # [修复] freq_encoder_only模式需要提取独立的频域特征
        if self.ablation_mode == 'freq_encoder_only':
            # 强制提取独立频域特征（即使使用CLIP模型）
            if is_first_batch:
                print("[消融-修复] freq_encoder_only: 提取独立频域特征")
            freq_features = self.frequency_extractor(frequency_image)

            # 使用简单融合（不使用MFB）
            if self.disable_mfb:
                fusion_features = image_features + text_features
                fusion_features = F.normalize(fusion_features, p=2, dim=1)
                if is_first_batch:
                    print("[消融] freq_encoder_only: 直接特征相加融合")
            else:
                fusion_features = self.attention(text_features, image_features, batch_count, epoch, self.dataset_name)
        elif self.ablation_mode == 'no_freq_encoder':
            # 仅移除频域分支，保持MFB图文融合
            if is_first_batch:
                print("[消融-控制变量] no_freq_encoder: 移除频域分支，保留MFB图文融合")
            fusion_features = self._clip_space_fusion(
                image_features=image_features,
                text_features=text_features,
                temperature=0.07,
                is_first_batch=is_first_batch
            )
        elif self.ablation_mode == 'no_freq_clip_fusion':
            # 方案A: 频域独立提取，不经过AFF，MFB保留用于图文融合
            freq_features = self.frequency_extractor(frequency_image)
            fusion_features = self._clip_space_fusion(
                image_features=image_features,
                text_features=text_features,
                temperature=0.07,
                is_first_batch=is_first_batch
            )
            if is_first_batch:
                print("[消融-控制变量] no_freq_clip_fusion: 频域独立提取，MFB图文融合")
        elif self.is_clip_model and isinstance(self.image_encoder, FrequencyEnhancedCLIPImageEncoder):
            # CLIP模型优化路径：频域特征已经融合到图像特征中
            if is_first_batch:
                print("🎯 使用优化的CLIP空间融合：频域特征已增强CLIP图像特征")

            # 图像特征已经是频域增强的CLIP特征，直接与文本特征融合
            if self.use_freq_enhanced_fusion:
                # 使用CLIP优化的对比学习融合
                if is_first_batch:
                    print("🔥 CLIP空间对比学习融合：利用语义对齐优势")
                
                # 简化的CLIP空间融合
                fusion_features = self._clip_space_fusion(
                    image_features=image_features,
                    text_features=text_features,
                    temperature=0.07,
                    is_first_batch=is_first_batch
                )
            else:
                # [消融] 禁用MFB时使用直接相加
                if self.disable_mfb:
                    fusion_features = image_features + text_features
                    fusion_features = F.normalize(fusion_features, p=2, dim=1)
                    if is_first_batch:
                        print("[消融] 直接特征相加融合")
                else:
                    # 使用注意力机制融合
                    attention_fusion = self.attention(text_features, image_features, batch_count, epoch, self.dataset_name)
                    fusion_features = attention_fusion
                    if is_first_batch:
                        print("🎯 CLIP空间注意力融合完成")
                    
        elif self.use_frequency_features:
            # 传统路径：频域特征作为独立模态处理
            if is_first_batch:
                print("📊 使用传统三模态融合：图像+文本+频域")
                
            # 提取频域特征
            freq_features = self.frequency_extractor(frequency_image)
            
            if is_first_batch:
                print(f"频域特征维度: {freq_features.shape}")
            
            # 确保频域特征批次大小与其他特征一致
            if freq_features.size(0) != batch_size:
                if is_first_batch:
                    print(f"警告: 频域特征批次大小不匹配: freq={freq_features.size(0)}, expected={batch_size}")
                # 如果批次大小为1，扩展到正确批次大小
                if freq_features.size(0) == 1:
                    freq_features = freq_features.expand(batch_size, -1)
            
            # 选择融合策略
            if self.use_freq_enhanced_fusion:
                # 使用两阶段频域增强融合
                if is_first_batch:
                    print("使用两阶段频域增强融合")
                
                # 执行两阶段融合
                fusion_features = self.two_stage_fusion(
                    spatial_features=image_features,
                    freq_features=freq_features,
                    text_features=text_features,
                    return_intermediate=False,
                    return_loss=False
                )
                
                if is_first_batch:
                    print(f"两阶段融合后特征维度: {fusion_features.shape}")
                    
            elif self.use_enhanced_fusion:
                if is_first_batch:
                    print(f"使用高级多模态融合: {self.fusion_type}")
                
                if self.fusion_type == 'transformer':
                    # 跨模态Transformer融合
                    fusion_features = self.enhanced_fusion(text_features, image_features, freq_features)
                elif self.fusion_type == 'dynamic':
                    # 动态特征融合
                    fusion_features, fusion_weights = self.enhanced_fusion(text_features, image_features, freq_features)
                    if is_first_batch:
                        print(f"动态融合权重: {fusion_weights.mean(dim=0)}")
                elif self.fusion_type == 'gated':
                    # 门控多模态融合
                    fusion_features = self.enhanced_fusion(text_features, image_features, freq_features)
            else:
                # 使用标准的门控和注意力融合
                # 1. 首先融合文本和图像特征（使用门控机制）
                text_image_gate_input = torch.cat([text_features, image_features], dim=1)
                text_image_gate_value = self.text_image_gate(text_image_gate_input)
                
                # 文本特征和图像特征的初步融合
                text_image_fusion = text_image_gate_value * text_features + (1 - text_image_gate_value) * image_features
                
                # 2. 使用注意力机制进一步融合
                attention_fusion = self.attention(text_features, image_features, batch_count, epoch, self.dataset_name)
                
                # 3. 频域特征与前面融合结果的融合（使用门控机制）
                freq_gate_input = torch.cat([attention_fusion, freq_features], dim=1)
                freq_gate_value = self.freq_gate(freq_gate_input)
                
                # 最终融合
                fusion_features = freq_gate_value * attention_fusion + (1 - freq_gate_value) * freq_features
                
                # 更新门控值统计（用于分析）
                if self.training:
                    self.gate_values[0] += text_image_gate_value.mean().item()
                    self.gate_values[1] += freq_gate_value.mean().item()
                    self.gate_counts[0] += 1
                    self.gate_counts[1] += 1
                
                if is_first_batch:
                    print(f"文本-图像门控值平均: {text_image_gate_value.mean().item()}")
                    print(f"频域门控值平均: {freq_gate_value.mean().item()}")
                    
                # 如果是训练的最后一个epoch，输出平均门控值
                if self.training and epoch is not None and epoch > 0 and batch_count == 0:
                    avg_text_image_gate = self.gate_values[0] / max(1, self.gate_counts[0])
                    avg_freq_gate = self.gate_values[1] / max(1, self.gate_counts[1])
                    print(f"平均门控值: 文本-图像={avg_text_image_gate:.4f}, 频域={avg_freq_gate:.4f}")
        else:
            # 不使用频域特征时
            if self.use_freq_enhanced_fusion:
                # 两阶段融合在没有频域特征时，退化为图像-文本对比学习融合
                if is_first_batch:
                    print("频域增强融合退化为图像-文本对比学习融合")
                
                # 使用对比学习融合（跳过第一阶段）
                # 使用对比学习融合（跳过第一阶段）
                # [修复] 使用在__init__中初始化的self.contrastive_fusion，避免设备不匹配问题
                if hasattr(self, 'contrastive_fusion'):
                    fusion_features = self.contrastive_fusion(image_features, text_features)
                else:
                    # 获取设备信息
                    device = image_features.device
                    
                    if is_first_batch:
                        print("警告: contrastive_fusion未在__init__中初始化，正在尝试动态初始化并移动到设备...")
                    
                    from utils.freq_enhanced_fusion import ContrastiveFusion
                    self.contrastive_fusion = ContrastiveFusion(
                        image_dim=self.proj_dim,
                        text_dim=self.proj_dim,
                        hidden_dim=self.proj_dim
                    ).to(device)
                    fusion_features = self.contrastive_fusion(image_features, text_features)
                
            elif self.use_enhanced_fusion and self.fusion_type == 'gated':
                # 只使用文本和图像特征的门控融合
                fusion_features = self.enhanced_fusion(text_features, image_features)
            elif not self.use_enhanced_fusion:
                # [消融] 禁用MFB时只用图像特征
                if self.disable_mfb:
                    fusion_features = image_features
                    if is_first_batch:
                        print("[消融] 只用图像特征回归")
                else:
                    # 使用标准的注意力融合
                    fusion_features = self.attention(text_features, image_features, batch_count, epoch, self.dataset_name)
            else:
                # 其他高级融合方法在没有频域特征时回退到标准融合
                fusion_features = self.attention(text_features, image_features, batch_count, epoch, self.dataset_name)
        
        if is_first_batch:
            print(f"最终融合特征维度: {fusion_features.shape}")

        # ==================== 任务自适应特征拼接 ====================
        
        # [消融实验] 统一使用 Concatenation (拼接)，不再额外拼接图像特征
        if self.ablation_mode == 'baseline':
            # Baseline: Image + Text
            # [修复] 对图像特征进行归一化 (Norm~12 -> Norm=1)，匹配文本特征量级
            image_norm = F.normalize(image_features_orig, p=2, dim=1)
            output_features = torch.cat([image_norm, text_features_orig], dim=1)
            if is_first_batch:
                print(f"[消融] Baseline: Image_Norm(768)+Text(768) = {output_features.shape[1]}维")
            return output_features, text_features_orig, image_features_orig
            
        elif self.ablation_mode == 'freq_encoder_only':
            # FreqOnly: Image + Text + Freq
            # 确保freq_features已计算
            if not self.use_frequency_features: 
                # 理论上不应该发生，但作为防守
                freq_features = torch.zeros_like(image_features_orig)
            
            # [修复] 对图像特征进行归一化
            image_norm = F.normalize(image_features_orig, p=2, dim=1)
            output_features = torch.cat([image_norm, text_features_orig, freq_features], dim=1)
            if is_first_batch:
                print(f"[消融] FreqOnly: Image_Norm(768)+Text(768)+Freq({freq_features.shape[1]}) = {output_features.shape[1]}维")
            return output_features, text_features_orig, image_features_orig

        elif self.ablation_mode == 'freq_clip_fusion':
            # FreqClipFusion: EnhancedImage + Text
            # image_features 已经是 EnhancedImage。如果它包含原始图像成分，可能也很大。为了安全，统一归一化。
            image_features_norm = F.normalize(image_features, p=2, dim=1)
            output_features = torch.cat([image_features_norm, text_features_orig], dim=1)
            if is_first_batch:
                print(f"[消融] FreqClipFusion: EnhancedImage_Norm(768)+Text(768) = {output_features.shape[1]}维")
            return output_features, text_features_orig, image_features_orig

        # ========== 控制变量法消融模式 ==========
        elif self.ablation_mode == 'no_freq_encoder':
            # 移除频域编码器：无 freq 增强的纯 CLIP 特征 + MFB
            # 公平对比必须保留 MFB
            image_features_norm = F.normalize(image_features_orig, p=2, dim=1)
            output_features = torch.cat([fusion_features, image_features_norm], dim=1)
            if is_first_batch:
                print(f"[消融-控制变量-公平修正] no_freq_encoder: Fusion(768)+Image_Norm(768) = {output_features.shape[1]}维")
            return output_features, text_features_orig, image_features_orig

        elif self.ablation_mode == 'no_freq_clip_fusion':
            # 方案A: MFB图文融合 + 独立频域特征 concat
            image_features_norm = F.normalize(image_features_orig, p=2, dim=1)
            output_features = torch.cat([fusion_features, image_features_norm, freq_features], dim=1)
            if is_first_batch:
                print(f"[消融-控制变量] no_freq_clip_fusion: Fusion(768)+Image_Norm(768)+Freq(768) = {output_features.shape[1]}维")
            return output_features, text_features_orig, image_features_orig

        elif self.ablation_mode == 'no_mfb':
            # 移除MFB：图文特征直接拼接
            image_features_norm = F.normalize(image_features_orig, p=2, dim=1)
            output_features = torch.cat([text_features_orig, image_features_norm], dim=1)
            if is_first_batch:
                print(f"[消融-控制变量] no_mfb: Text(768)+Image_Norm(768) = {output_features.shape[1]}维")
            return output_features, text_features_orig, image_features_orig

        # [完整模型] (和原本的 disable_mfb 逻辑分离)
        # 消融模式（禁用MFB但非上述模式，或者为了兼容旧逻辑）：只用图像特征
        if self.disable_mfb and self.ablation_mode not in ['baseline', 'freq_encoder_only', 'freq_clip_fusion', 'no_freq_encoder', 'no_freq_clip_fusion', 'no_mfb']:
            output_features = fusion_features  # [B, 768] 图像特征
            if is_first_batch:
                print(f"[消融-旧] 只用图像特征 = 768维")
            return output_features, text_features_orig, image_features_orig

        # 统一使用融合+图像特征（一致性任务追加文本特征与相似度标量）
        # [修复] 对图像特征进行归一化，使其与MFB特征(norm=1)量级一致
        image_features_norm = F.normalize(image_features_orig, p=2, dim=1)

        is_correspondence_task = self.dataset_name in ['AIGCIQA2023c', 'AGIQA3Kc', 'I2IQAc']
        if is_correspondence_task:
            text_norm = F.normalize(text_features_orig, p=2, dim=1)
            correspondence_sim = (text_norm * image_features_norm).sum(dim=1, keepdim=True)  # [B, 1]
            output_features = torch.cat([
                fusion_features,      # [B, 768] (已归一化)
                text_features_orig,   # [B, 768]
                correspondence_sim    # [B, 1]
            ], dim=1)  # [B, 1537]
        else:
            output_features = torch.cat([
                fusion_features,      # [B, 768] (已归一化)
                image_features_norm   # [B, 768] (新归一化)
            ], dim=1)  # [B, 1536]

        if is_first_batch:
            if is_correspondence_task:
                print(f"📦 特征拼接: fusion(768) + text(768) + corr(1) = 1537维")
            else:
                print(f"📦 特征拼接: fusion(768) + image_norm(768) = 1536维")
            print(f"   [Deepmind修复] 已对图像特征进行L2归一化，解决模长不匹配问题")

        return output_features, text_features_orig, image_features_orig

    def _clip_space_fusion(self, image_features, text_features, temperature=0.07, is_first_batch=False):
            """
            优化的CLIP空间融合方法

            消融实验支持:
            - disable_mfb=True: 禁用MFB，直接返回图像特征
            """

            # [消融] 如果禁用MFB，只用图像特征
            if self.disable_mfb:
                if is_first_batch:
                    print("[消融] 只用图像特征回归")
                return image_features

            # --- MFB逻辑 ---
            # 融合的唯一来源是 MFB 的二阶交互

            B = image_features.size(0)

            # 1. 扩展 (Expand)
            img_exp = self.mfb_img_expand(image_features)
            txt_exp = self.mfb_txt_expand(text_features)

            # 2. 交互 (Interact) - 在高维空间中进行Hadamard积
            fused = img_exp * txt_exp
            fused = self.mfb_dropout(fused)

            # 3. 压缩 (Squeeze)
            fused_rs = fused.view(B, self.mfb_output_dim, self.mfb_k_factor)

            # Sum pooling (通过 AvgPool * k_factor 实现)
            fused_pool = self.mfb_pool(fused_rs).squeeze(2) * self.mfb_k_factor

            # Power Norm (Signed Sqrt)
            fused_pow = torch.sqrt(F.relu(fused_pool)) - torch.sqrt(F.relu(-fused_pool))

            # L2 Norm
            fusion_features = F.normalize(fused_pow, p=2, dim=1)

            # --- MFB逻辑结束 ---

            if is_first_batch:
                print(">>> 使用_clip_space_fusion (模式: MFB二阶交互)")

            return fusion_features

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from transformers import AutoConfig
except ImportError:
    print("Warning: transformers package not found. 'AutoConfig' will not be available.")
    # 定义一个最小的替代品
    class AutoConfig:
        @staticmethod
        def from_pretrained(model_path):
            print(f"Warning: Attempting to load config from {model_path} without transformers pkg.")
            return type('Config', (), {'hidden_size': 768})()

class MLP(nn.Module):
    """
    【已修复并简化】的回归器模型。
    此架构接收 MFB融合特征 和 原始图像特征 的拼接。
    根据性能反馈，简化为单层隐藏层 (1536 -> 512 -> 1)，以避免信息瓶颈。
    """
    def __init__(
        self,
        image_dim,  # 原始图像特征维度 (例如 768)
        model_path="./bert-base-uncased",
        dataset_name=None,
        freq_feature_dim=768, # MFB融合特征维度 (现在由txt_dim控制)
        normalize_to_five=False,
        txt_dim=None,
        ablation_mode='full'
    ):
        super(MLP, self).__init__()

        self.ablation_mode = ablation_mode

        # --- 兼容您脚本的配置加载 ---
        if txt_dim is not None:
            self.config = type('Config', (), {'hidden_size': txt_dim})()
        elif model_path == "clip_convnext":
            self.config = type('Config', (), {'hidden_size': 768})()
        else:
            self.config = AutoConfig.from_pretrained(model_path)

        self.dataset_name = dataset_name
        self.normalize_to_five = normalize_to_five

        # ==================== 消融模式：Standardized Concatenation ====================
        if ablation_mode == 'baseline':
            # Baseline: Image + Text (Concat)
            # 768 + 768 = 1536
            fusion_dim = image_dim + self.config.hidden_size
            hidden_dim = 512
            dropout_rate = 0.2
            print(f"[消融] Baseline (Concat) MLP配置:")
            print(f"   输入维度: {fusion_dim} (Image+Text)")
            
        elif ablation_mode == 'freq_encoder_only':
            # FreqOnly: Image + Text + Freq (Concat)
            # 768 + 768 + 768 = 2304
            fusion_dim = image_dim + self.config.hidden_size + freq_feature_dim
            hidden_dim = 512
            dropout_rate = 0.2
            print(f"[消融] FreqEncoderOnly (Concat) MLP配置:")
            print(f"   输入维度: {fusion_dim} (Image+Text+Freq)")
            
        elif ablation_mode == 'freq_clip_fusion':
            # FreqClipFusion: EnhancedImage + Text (Concat)
            # 768 + 768 = 1536
            fusion_dim = image_dim + self.config.hidden_size
            hidden_dim = 512
            dropout_rate = 0.2
            print(f"[消融] FreqClipFusion (Concat) MLP配置:")
            print(f"   输入维度: {fusion_dim} (EnhancedImage+Text)")

        elif ablation_mode == 'no_freq_clip_fusion':
            # 方案A: MFB(image,text) + image_norm + freq = 2304
            fusion_dim = self.config.hidden_size + image_dim + freq_feature_dim  # 768+768+768=2304
            hidden_dim = 512
            dropout_rate = 0.3
            print(f"[消融-控制变量] no_freq_clip_fusion MLP配置:")
            print(f"   输入维度: {fusion_dim} (Fusion+Image+Freq)")

        elif ablation_mode in ['no_freq_encoder', 'no_mfb']:
            # 控制变量法消融模式:
            # 这些模式保持了 Fusion(768) + Image(768) = 1536 的输出结构
            fusion_dim = self.config.hidden_size + image_dim # 768 + 768 = 1536
            hidden_dim = 512
            dropout_rate = 0.3
            print(f"[消融-控制变量] {ablation_mode} MLP配置:")
            print(f"   输入维度: {fusion_dim} (Fusion+Image)")
            
        else:
            # ==================== 完整模型 / 任务自适应MLP架构 ====================
            mfb_output_dim = self.config.hidden_size  # 768
            
            is_correspondence_task = dataset_name in ['AIGCIQA2023c', 'AGIQA3Kc', 'I2IQAc'] if dataset_name else False
            if is_correspondence_task:
                # 一致性任务：fusion + text + corr_sim
                fusion_dim = mfb_output_dim + self.config.hidden_size + 1  # 768 + 768 + 1 = 1537
            else:
                # 其他任务：fusion + image
                fusion_dim = mfb_output_dim + image_dim  # 768 + 768 = 1536
            hidden_dim = 512
            dropout_rate = 0.3
            print(f"📦 任务MLP配置:")
            if is_correspondence_task:
                print(f"   输入维度: {fusion_dim} (融合{mfb_output_dim} + 文本{self.config.hidden_size} + corr(1))")
            else:
                print(f"   输入维度: {fusion_dim} (融合{mfb_output_dim} + 图像{image_dim})")

        # ==================== 结束任务自适应配置 ====================

        # 新结构: [Input] -> FC1 -> Norm1 -> ReLU -> Drop1 -> FC2 -> [Output]
        self.fc1 = nn.Linear(fusion_dim, hidden_dim)    # (1536 -> 512)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)        # (0.3)
        
        self.fc2 = nn.Linear(hidden_dim, 1)             # (512 -> 1)
        
        self.label_smoothing = 0.0
        self.use_sigmoid = False
        
        print(f"MLP (回归头) 初始化: 数据集={dataset_name}, normalize_to_five={normalize_to_five}")
        
        # 初始化权重
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        
        print(f"MLP (回归头) 配置 [已修复-单层]: fusion_dim={fusion_dim} (MFB+图像), hidden_dim={hidden_dim}, dropout={dropout_rate}")

    def forward(self, x):
        """
        前向传播
        参数:
            x (Tensor): 形状为，
                       应包含 MFB融合特征 和 原始图像特征 的拼接
        """
        
        # 应用第一层
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 应用输出层 (移除了原有的第二层)
        x = self.fc2(x)
        
        # 根据数据集类型调整输出范围
        output = x.squeeze(-1)
        if self.normalize_to_five:
            output = torch.sigmoid(output) * 5.0
        
        return output
