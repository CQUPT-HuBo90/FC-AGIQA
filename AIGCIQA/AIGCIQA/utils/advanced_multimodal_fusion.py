import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalTransformer(nn.Module):
    """跨模态Transformer融合模块"""
    def __init__(self, d_model=768, n_heads=8, n_layers=3, dropout=0.1):
        super(CrossModalTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # 跨模态Transformer层
        self.layers = nn.ModuleList([
            CrossModalTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(d_model * 3, d_model)
        
    def forward(self, text_feat, img_feat, freq_feat):
        """
        参数:
            text_feat: [B, D]
            img_feat: [B, D]
            freq_feat: [B, D]
        返回:
            fused_feat: [B, D]
        """
        B = text_feat.size(0)
        
        # 添加模态标识并重塑为序列 [B, 3, D]
        features = torch.stack([text_feat, img_feat, freq_feat], dim=1)
        
        # 添加位置编码
        features = self.pos_encoding(features)
        
        # 通过Transformer层
        for layer in self.layers:
            features = layer(features)
        
        # 展平并投影
        features_flat = features.reshape(B, -1)
        output = self.output_proj(features_flat)
        
        return output


class CrossModalTransformerLayer(nn.Module):
    """单个跨模态Transformer层"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossModalTransformerLayer, self).__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: [B, 3, D]"""
        # 自注意力
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 跨模态注意力（每个模态关注其他模态）
        B, S, D = x.shape
        cross_out = []
        for i in range(S):
            # 当前模态作为query，其他模态作为key/value
            query = x[:, i:i+1, :]
            kv_indices = [j for j in range(S) if j != i]
            key_value = x[:, kv_indices, :]
            
            out, _ = self.cross_attn(query, key_value, key_value)
            cross_out.append(out)
        
        cross_out = torch.cat(cross_out, dim=1)
        x = self.norm2(x + self.dropout(cross_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 修复：确保div_term的大小匹配实际的维度切片
        # 正弦分量使用 0::2 (偶数索引)，余弦分量使用 1::2 (奇数索引)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # 正弦分量：使用完整的div_term
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 余弦分量：只使用与奇数索引数量匹配的div_term部分
        if d_model > 1:  # 确保有奇数索引位置
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """x: [B, S, D]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DynamicFeatureFusion(nn.Module):
    """动态特征融合模块，根据输入内容自适应调整融合权重"""
    def __init__(self, text_dim, img_dim, freq_dim, hidden_dim=768):
        super(DynamicFeatureFusion, self).__init__()
        
        # 特征投影
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)
        
        # 动态权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softmax(dim=1)
        )
        
        # 特征交互网络
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, text_feat, img_feat, freq_feat):
        """
        动态融合三种模态特征
        
        参数:
            text_feat: 文本特征 [B, text_dim]
            img_feat: 图像特征 [B, img_dim]
            freq_feat: 频域特征 [B, freq_dim]
            
        返回:
            融合特征 [B, hidden_dim]
        """
        # 特征投影
        text_h = self.text_proj(text_feat)
        img_h = self.img_proj(img_feat)
        freq_h = self.freq_proj(freq_feat)
        
        # 拼接特征
        concat_feat = torch.cat([text_h, img_h, freq_h], dim=1)
        
        # 生成动态权重
        weights = self.weight_generator(concat_feat)  # [B, 3]
        
        # 加权融合
        weighted_feat = (weights[:, 0:1] * text_h + 
                        weights[:, 1:2] * img_h + 
                        weights[:, 2:3] * freq_h)
        
        # 特征交互
        interaction_feat = self.interaction(concat_feat)
        
        # 组合加权特征和交互特征
        combined = torch.cat([weighted_feat, interaction_feat], dim=1)
        output = self.output(combined)
        
        return output, weights


class GatedMultiModalFusion(nn.Module):
    """门控多模态融合，支持任意数量的模态"""
    def __init__(self, modal_dims, output_dim=768):
        super(GatedMultiModalFusion, self).__init__()
        
        self.n_modals = len(modal_dims)
        
        # 每个模态的投影层
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in modal_dims
        ])
        
        # 门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 1),
                nn.Sigmoid()
            ) for _ in range(self.n_modals)
        ])
        
        # 全局融合
        self.global_fusion = nn.Sequential(
            nn.Linear(output_dim * self.n_modals, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, *modal_features):
        """
        参数:
            modal_features: 各模态特征的列表
            
        返回:
            融合后的特征
        """
        # 投影所有模态
        projected = [proj(feat) for proj, feat in zip(self.projections, modal_features)]
        
        # 计算平均特征作为参考
        avg_feat = torch.stack(projected).mean(dim=0)
        
        # 门控融合
        gated_features = []
        for i, (feat, gate) in enumerate(zip(projected, self.gates)):
            gate_input = torch.cat([feat, avg_feat], dim=1)
            gate_value = gate(gate_input)
            gated_feat = feat * gate_value
            gated_features.append(gated_feat)
        
        # 全局融合
        concat_feat = torch.cat(gated_features, dim=1)
        output = self.global_fusion(concat_feat)
        
        return output