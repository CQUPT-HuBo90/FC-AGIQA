import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Union, List

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("open_clip not available. Install with: pip install open_clip_torch")


class CLIPConvNeXtTextEncoder(nn.Module):
    """
    CLIP ConvNeXt文本编码器，基于open_clip实现
    
    这个类使用open_clip库访问CLIP ConvNeXt模型的文本编码器部分，
    提供与项目中其他文本编码器一致的接口。
    """
    
    def __init__(
        self, 
        model_name: str = 'convnext_large_d_320',
        pretrained: str = 'laion2b_s29b_b131k_ft',
        freeze_encoder: bool = False,
        output_dim: Optional[int] = None,
        use_projection: bool = True,
        dropout_rate: float = 0.1,
        max_length: int = 77
    ):
        """
        初始化CLIP ConvNeXt文本编码器
        
        Args:
            model_name: ConvNeXt模型名称，支持:
                - convnext_base_w_320: 基础版本，320x320输入
                - convnext_large_d_320: 大型版本，320x320输入（推荐）
                - convnext_base: 基础版本，256x256输入
                - convnext_large_d: 大型版本，256x256输入
            pretrained: 预训练权重名称
            freeze_encoder: 是否冻结编码器参数
            output_dim: 输出特征维度，如果为None则使用模型默认维度
            use_projection: 是否使用额外的投影层
            dropout_rate: dropout比率
            max_length: 最大文本长度
        """
        super(CLIPConvNeXtTextEncoder, self).__init__()
        
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("open_clip is required for CLIPConvNeXtTextEncoder. Install with: pip install open_clip_torch")
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.use_projection = use_projection
        self.max_length = max_length
        
        # 支持的ConvNeXt模型配置
        # 根据open_clip文档，这些模型的文本编码器都输出768维特征
        self.supported_models = {
            'convnext_base_w_320': {'text_dim': 768},
            'convnext_large_d_320': {'text_dim': 768},  # 推荐使用
            'convnext_base': {'text_dim': 768},
            'convnext_large_d': {'text_dim': 768},
        }
        
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(self.supported_models.keys())}")
        
        # 获取模型配置
        self.model_config = self.supported_models[model_name]
        self.text_dim = self.model_config['text_dim']
        
        # 如果未指定输出维度，使用模型默认维度
        if self.output_dim is None:
            self.output_dim = self.text_dim
        
        # 初始化CLIP模型
        self._init_clip_model()
        
        # 设置冻结参数
        if freeze_encoder:
            self._freeze_encoder()
        
        # 可选的投影层
        self.projection = None
        if use_projection and self.output_dim != self.text_dim:
            self.projection = self._create_projection_head(dropout_rate)
        
        print(f"初始化CLIPConvNeXtTextEncoder: {model_name}")
        print(f"预训练权重: {pretrained}")
        print(f"文本特征维度: {self.text_dim} -> {self.output_dim}")
        print(f"参数冻结: {freeze_encoder}")
        print(f"最大文本长度: {max_length}")
        
    def _init_clip_model(self):
        """初始化CLIP模型"""
        try:
            # 使用open_clip加载CLIP模型
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained
            )
            
            # 获取tokenizer
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # 只使用文本编码器部分
            self.text_encoder = self.clip_model
            
            print(f"成功加载CLIP ConvNeXt模型: {self.model_name}")
            print(f"预训练权重: {self.pretrained}")
            
        except Exception as e:
            print(f"加载CLIP ConvNeXt模型失败: {e}")
            print("尝试加载不带预训练权重的模型...")
            
            # 如果预训练权重加载失败，尝试不使用预训练权重
            try:
                self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_name, 
                    pretrained=None
                )
                self.tokenizer = open_clip.get_tokenizer(self.model_name)
                self.text_encoder = self.clip_model
                print("成功加载模型架构（无预训练权重）")
            except Exception as e2:
                raise RuntimeError(f"无法加载CLIP ConvNeXt模型: {e2}")
    
    def _freeze_encoder(self):
        """冻结编码器参数"""
        # 冻结文本编码器的参数
        if hasattr(self.clip_model, 'text'):
            for param in self.clip_model.text.parameters():
                param.requires_grad = False
        elif hasattr(self.clip_model, 'token_embedding'):
            # 如果模型结构不同，尝试冻结token embedding和transformer部分
            for param in self.clip_model.token_embedding.parameters():
                param.requires_grad = False
            if hasattr(self.clip_model, 'transformer'):
                for param in self.clip_model.transformer.parameters():
                    param.requires_grad = False
            if hasattr(self.clip_model, 'ln_final'):
                for param in self.clip_model.ln_final.parameters():
                    param.requires_grad = False
        else:
            # 最后的备选方案：冻结整个模型
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        print("CLIP ConvNeXt文本编码器参数已冻结")
    
    def _create_projection_head(self, dropout_rate: float):
        """创建投影头"""
        return nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )
    
    def get_feature_dim(self):
        """返回输出特征维度"""
        return self.output_dim
    
    def get_hidden_size(self):
        """返回隐藏层维度，兼容现有代码"""
        return self.output_dim
    
    def preprocess_text(self, texts):
        """
        预处理文本数据，确保符合CLIP输入要求
        
        Args:
            texts: 输入文本，字符串列表
            
        Returns:
            tokenized: 预处理后的文本token
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用open_clip的tokenizer进行文本预处理
        tokenized = self.tokenizer(texts)
        
        return tokenized
    
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 文本token ID，shape为[batch_size, seq_len]，来自CLIP tokenizer
            attention_mask: 注意力掩码（兼容性参数，CLIP不使用）
            
        Returns:
            features: 文本特征，shape为[batch_size, output_dim]
        """
        # 直接使用输入的token ID，这些已经由CLIP tokenizer处理过
        text_tokens = input_ids
        
        # 确保token在正确的设备上
        device = None
        try:
            # 安全地获取模型设备
            for param in self.clip_model.parameters():
                device = param.device
                break
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if device is not None and device.type == 'cuda':
            text_tokens = text_tokens.to(device)
        
        # 通过CLIP文本编码器
        with torch.set_grad_enabled(self.training):
            # 使用CLIP模型的encode_text方法
            if hasattr(self.clip_model, 'encode_text'):
                features = self.clip_model.encode_text(text_tokens)
            else:
                # 备选方案：直接调用模型
                features = self.clip_model(text_tokens)
                if hasattr(features, 'text_embeds'):
                    features = features.text_embeds
                elif isinstance(features, tuple):
                    features = features[0]  # 通常第一个是文本特征
        
        # 应用投影层（如果有）
        if self.projection is not None:
            features = self.projection(features)
        
        return features


class CLIPConvNeXtTextEncoderWrapper(nn.Module):
    """
    CLIP ConvNeXt文本编码器包装器，提供与transformers AutoModel兼容的接口
    
    这个类确保与现有的FMAC代码完全兼容，特别是与model.py中的Encoder类。
    """
    
    def __init__(
        self, 
        model_name: str = 'convnext_large_d_320',
        pretrained: str = 'laion2b_s29b_b131k_ft',
        freeze_encoder: bool = False,
        output_dim: int = 768,
        max_length: int = 77
    ):
        super(CLIPConvNeXtTextEncoderWrapper, self).__init__()
        
        # 创建底层编码器
        self.encoder = CLIPConvNeXtTextEncoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            output_dim=output_dim,
            use_projection=True,
            max_length=max_length
        )
        
        # 创建兼容的config对象
        self.config = type('Config', (), {
            'hidden_size': output_dim,
            'vocab_size': 50000,  # 默认值
            'max_position_embeddings': max_length,
        })()
        
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        """
        兼容transformers接口的前向传播
        
        Args:
            input_ids: 文本token ID
            attention_mask: 注意力掩码（兼容性参数）
            output_hidden_states: 是否输出隐藏状态（兼容性参数）
            
        Returns:
            outputs: 包含last_hidden_state的对象
        """
        # 获取文本特征
        features = self.encoder(input_ids, attention_mask)
        
        # 创建兼容的输出对象
        class OutputObject:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
                
        # 扩展维度以匹配预期的序列格式 [batch_size, 1, hidden_size]
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        return OutputObject(features)


def get_clip_convnext_text_encoder(
    model_name: str = 'convnext_large_d_320',
    pretrained: str = 'laion2b_s29b_b131k_ft',
    output_dim: int = 768,
    freeze_encoder: bool = False,
    use_wrapper: bool = True,
    max_length: int = 512
):
    """
    工厂函数，创建CLIP ConvNeXt文本编码器
    
    Args:
        model_name: ConvNeXt模型名称
        pretrained: 预训练权重名称
        output_dim: 输出维度
        freeze_encoder: 是否冻结编码器
        use_wrapper: 是否使用包装器以兼容transformers接口
        max_length: 最大文本长度
        
    Returns:
        CLIP ConvNeXt文本编码器实例
    """
    if use_wrapper:
        return CLIPConvNeXtTextEncoderWrapper(
            model_name=model_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            output_dim=output_dim,
            max_length=max_length
        )
    else:
        return CLIPConvNeXtTextEncoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            output_dim=output_dim,
            use_projection=True,
            max_length=max_length
        )


# 示例用法和测试
if __name__ == "__main__":
    if OPEN_CLIP_AVAILABLE:
        print("测试CLIP ConvNeXt文本编码器...")
        
        # 测试基础编码器
        encoder = get_clip_convnext_text_encoder(
            model_name='convnext_large_d_320',
            pretrained='laion2b_s29b_b131k_ft',
            output_dim=768,
            use_wrapper=True
        )
        
        # 创建测试输入
        test_texts = ["这是一个测试文本", "another test text"]
        
        with torch.no_grad():
            # 测试tokenization
            tokens = encoder.encoder.preprocess_text(test_texts)
            print(f"Tokenized shape: {tokens.shape}")
            
            # 测试forward pass
            features = encoder.encoder(tokens)
            print(f"输出特征形状: {features.shape}")
            print(f"输出特征维度: {encoder.encoder.get_feature_dim()}")
        
        print("测试完成！")
    else:
        print("open_clip不可用，跳过测试") 