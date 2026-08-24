import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import Optional, Union, List
import os

class CLIPConvNeXtEncoder(nn.Module):
    """
    基于open_clip的CLIP-ConvNeXt图像编码器
    
    支持多种ConvNeXt变体：
    - convnext_base_w_320: 基础版本，320x320输入
    - convnext_large_d_320: 大型版本，320x320输入（推荐）
    - convnext_base: 基础版本，256x256输入
    """
    
    def __init__(
        self,
        model_name: str = 'convnext_large_d_320',
        pretrained: str = 'laion2b_s29b_b131k_ft',
        freeze_encoder: bool = False,
        output_dim: Optional[int] = None,
        use_projection: bool = True,
        dropout_rate: float = 0.1,
        use_trunk_features: bool = False
    ):
        """
        初始化CLIP-ConvNeXt编码器

        Args:
            model_name: ConvNeXt模型名称
            pretrained: 预训练权重名称
            freeze_encoder: 是否冻结编码器参数
            output_dim: 输出特征维度，如果为None则使用模型默认维度
            use_projection: 是否使用额外的投影层
            dropout_rate: dropout比率
            use_trunk_features: 若为True，使用ConvNeXt trunk的倒数空间特征图
                （投影头之前）做全局平均池化，保留模糊/噪声/伪影等低层质量信息，
                更适合图像质量评估（IQA）任务。默认False（沿用CLIP语义投影向量）。
        """
        super(CLIPConvNeXtEncoder, self).__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.use_projection = use_projection
        self.use_trunk_features = use_trunk_features
        
        # 支持的模型配置
        # 注意：CLIP-ConvNeXt的实际输出维度是768，不是1024
        self.supported_models = {
            'convnext_base_w_320': {'input_size': 320, 'feature_dim': 640},
            'convnext_large_d_320': {'input_size': 320, 'feature_dim': 768},  # 修正：实际输出是768
            'convnext_base': {'input_size': 256, 'feature_dim': 640},
            'convnext_large_d': {'input_size': 256, 'feature_dim': 768},  # 修正：实际输出是768
        }
        
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(self.supported_models.keys())}")
        
        # 获取模型配置
        self.model_config = self.supported_models[model_name]
        self.input_size = self.model_config['input_size']
        self.feature_dim = self.model_config['feature_dim']
        
        # 如果未指定输出维度，使用模型默认维度
        if self.output_dim is None:
            self.output_dim = self.feature_dim
        
        # 初始化CLIP模型
        self._init_clip_model()

        # trunk特征模式下，实际输出维度来自ConvNeXt trunk最后一层通道数
        self.trunk_feature_dim = self._infer_trunk_feature_dim() if self.use_trunk_features else self.feature_dim
        if self.use_trunk_features and self.output_dim == self.feature_dim:
            self.output_dim = self.trunk_feature_dim

        # 设置冻结参数
        if freeze_encoder:
            self._freeze_encoder()

        # 可选的投影层
        self.projection = None
        input_feature_dim = self.trunk_feature_dim if self.use_trunk_features else self.feature_dim
        if use_projection and self.output_dim != input_feature_dim:
            self.projection = self._create_projection_head(dropout_rate, input_feature_dim)

        print(f"初始化CLIPConvNeXtEncoder: {model_name}")
        print(f"预训练权重: {pretrained}")
        print(f"输入尺寸: {self.input_size}x{self.input_size}")
        print(f"特征模式: {'trunk倒数空间特征' if self.use_trunk_features else 'CLIP投影语义向量'}")
        print(f"特征维度: {input_feature_dim} -> {self.output_dim}")
        print(f"参数冻结: {freeze_encoder}")
        
    def _init_clip_model(self):
        """初始化CLIP模型"""
        try:
            # 尝试加载预训练模型
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained
            )
            
            # 只保留图像编码器，避免未使用的文本塔进入state_dict/optimizer
            self.image_encoder = self.clip_model.visual
            self.clip_model = None

            print(f"成功加载预训练权重: {self.pretrained}")
            
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("尝试加载不带预训练权重的模型...")
            
            # 如果预训练权重加载失败，尝试不使用预训练权重
            try:
                self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_name, 
                    pretrained=None
                )
                self.image_encoder = self.clip_model.visual
                self.clip_model = None
                print("成功加载模型架构（无预训练权重）")
            except Exception as e2:
                raise RuntimeError(f"无法加载模型: {e2}")
    
    def _freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        print("CLIP-ConvNeXt编码器参数已冻结")
    
    def _create_projection_head(self, dropout_rate: float, input_dim: int = None):
        """创建投影头"""
        in_dim = input_dim if input_dim is not None else self.feature_dim
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

    def _extract_trunk_features(self, images):
        """
        提取ConvNeXt trunk的倒数空间特征图（投影头之前），并做全局平均池化。

        open_clip的视觉塔(self.image_encoder)是TimmModel，内部:
          - self.image_encoder.trunk: timm的ConvNeXt主干
          - self.image_encoder.head:  CLIP投影头(pool+proj到联合嵌入空间)
        我们只取trunk的空间特征，跳过会丢失低层质量信息的投影头。

        Returns:
            features: [B, C] 池化后的trunk特征
        """
        trunk = getattr(self.image_encoder, 'trunk', None)
        if trunk is None:
            # 退化：没有trunk属性时直接走完整视觉塔
            feats = self.image_encoder(images)
        elif hasattr(trunk, 'forward_features'):
            # timm标准接口：返回空间特征图 [B, C, H, W]
            feats = trunk.forward_features(images)
        else:
            feats = trunk(images)

        # 全局平均池化到 [B, C]
        if feats.dim() == 4:        # [B, C, H, W]
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        elif feats.dim() == 3:      # [B, L, C] (某些实现)
            feats = feats.mean(dim=1)
        return feats

    def _infer_trunk_feature_dim(self):
        """跑一个dummy样本，推断trunk池化特征的通道数"""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                device = next(self.image_encoder.parameters()).device
                test_input = torch.randn(1, 3, self.input_size, self.input_size, device=device)
                feats = self._extract_trunk_features(test_input)
                dim = feats.shape[1]
            print(f"[trunk特征] 推断ConvNeXt trunk池化维度: {dim}")
            return dim
        except Exception as e:
            print(f"[trunk特征] 维度推断失败，回退到配置维度{self.feature_dim}: {e}")
            return self.feature_dim
        finally:
            if was_training:
                self.train()
    
    def get_feature_dim(self):
        """返回输出特征维度"""
        # 实际跑一个样本来获取真实输出维度（走完整forward路径，
        # 因此trunk特征模式/投影层都会被正确计入）
        if hasattr(self, 'image_encoder'):
            try:
                with torch.no_grad():
                    test_input = torch.randn(1, 3, self.input_size, self.input_size)
                    if next(self.image_encoder.parameters()).is_cuda:
                        test_input = test_input.cuda()

                    test_output = self.forward(test_input)
                    return test_output.shape[1]
            except Exception as e:
                print(f"[get_feature_dim] 动态探测失败，回退到配置维度: {e}")

        # 如果使用了自定义投影层，返回投影后的维度
        if self.projection is not None:
            return self.output_dim

        # trunk特征模式：返回trunk通道维度
        if self.use_trunk_features:
            return self.trunk_feature_dim

        # 否则返回配置的输出维度
        return self.output_dim
    
    def get_input_size(self):
        """返回期望的输入尺寸"""
        return self.input_size
    
    def preprocess_image(self, images):
        """
        预处理图像数据
        
        Args:
            images: 输入图像，可以是PIL图像列表或tensor
            
        Returns:
            preprocessed_images: 预处理后的图像tensor
        """
        if isinstance(images, list):
            # 如果是PIL图像列表，使用CLIP的预处理
            return torch.stack([self.preprocess(img) for img in images])
        elif torch.is_tensor(images):
            # 如果已经是tensor，检查尺寸和格式
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # 确保图像在正确的范围内[0,1]
            if images.max() > 1.0:
                images = images / 255.0
            
            # 调整到期望的尺寸
            if images.shape[-1] != self.input_size or images.shape[-2] != self.input_size:
                images = F.interpolate(
                    images, 
                    size=(self.input_size, self.input_size), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # ConvNeXt的标准化参数（基于CLIP预训练）
            # 这些值来自open_clip的默认配置
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            
            if images.is_cuda:
                mean = mean.cuda()
                std = std.cuda()
            
            images = (images - mean) / std
            return images
        else:
            raise ValueError(f"不支持的图像格式: {type(images)}")
    
    def forward(self, images):
        """
        前向传播
        
        Args:
            images: 输入图像，shape为[batch_size, channels, height, width]
            
        Returns:
            features: 图像特征，shape为[batch_size, output_dim]
        """
        # 预处理图像（如果需要）
        if not (torch.is_tensor(images) and images.dim() == 4):
            images = self.preprocess_image(images)
        
        # 确保输入尺寸正确
        if images.shape[-1] != self.input_size or images.shape[-2] != self.input_size:
            images = F.interpolate(
                images, 
                size=(self.input_size, self.input_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 通过ConvNeXt图像编码器
        with torch.set_grad_enabled(self.training):
            if self.use_trunk_features:
                features = self._extract_trunk_features(images)
            else:
                features = self.image_encoder(images)

                # 如果输出是多维的，进行全局平均池化
                if features.dim() > 2:
                    features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        
        # 应用投影层（如果有）
        if self.projection is not None:
            features = self.projection(features)
        
        return features


class CLIPConvNeXtEncoderWithFeatureExtraction(CLIPConvNeXtEncoder):
    """
    增强版CLIP-ConvNeXt编码器，支持多层特征提取
    """
    
    def __init__(self, *args, extract_layers: List[str] = None, **kwargs):
        """
        初始化增强版编码器
        
        Args:
            extract_layers: 要提取特征的层名称列表
        """
        super().__init__(*args, **kwargs)
        self.extract_layers = extract_layers or ['final']
        self.feature_hooks = {}
        
        if self.extract_layers and 'final' not in self.extract_layers:
            self._register_hooks()
    
    def _register_hooks(self):
        """注册特征提取钩子"""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_hooks[name] = output
            return hook
        
        # 为指定层注册钩子
        for name, module in self.image_encoder.named_modules():
            if any(layer in name for layer in self.extract_layers):
                module.register_forward_hook(get_activation(name))
    
    def forward(self, images, return_all_features=False):
        """
        前向传播，支持多层特征提取
        
        Args:
            images: 输入图像
            return_all_features: 是否返回所有层的特征
            
        Returns:
            features: 最终特征或特征字典
        """
        # 清空之前的特征
        self.feature_hooks.clear()
        
        # 获取最终特征
        final_features = super().forward(images)
        
        if return_all_features and self.feature_hooks:
            # 返回所有特征
            all_features = {'final': final_features}
            all_features.update(self.feature_hooks)
            return all_features
        else:
            return final_features


def get_clip_convnext_encoder(
    model_name: str = 'convnext_large_d_320',
    pretrained: str = 'laion2b_s29b_b131k_ft',
    output_dim: Optional[int] = None,
    freeze_encoder: bool = False,
    use_enhanced: bool = False,
    use_trunk_features: bool = False,
    **kwargs
):
    """
    工厂函数，创建CLIP-ConvNeXt图像编码器
    
    Args:
        model_name: ConvNeXt模型名称
        pretrained: 预训练权重名称
        output_dim: 输出维度
        freeze_encoder: 是否冻结编码器
        use_enhanced: 是否使用增强版本
        **kwargs: 其他参数
        
    Returns:
        CLIP-ConvNeXt编码器实例
    """
    if use_enhanced:
        return CLIPConvNeXtEncoderWithFeatureExtraction(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            freeze_encoder=freeze_encoder,
            use_trunk_features=use_trunk_features,
            **kwargs
        )
    else:
        return CLIPConvNeXtEncoder(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            freeze_encoder=freeze_encoder,
            use_trunk_features=use_trunk_features,
            **kwargs
        )


# 示例用法和测试
if __name__ == "__main__":
    # 测试基础编码器
    print("测试CLIP-ConvNeXt编码器...")
    
    encoder = get_clip_convnext_encoder(
        model_name='convnext_large_d_320',
        pretrained='laion2b_s29b_b131k_ft',
        output_dim=1024
    )
    
    # 创建测试输入
    test_input = torch.randn(2, 3, 320, 320)
    
    with torch.no_grad():
        features = encoder(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {features.shape}")
        print(f"输出特征维度: {encoder.get_feature_dim()}")
    
    print("测试完成！") 