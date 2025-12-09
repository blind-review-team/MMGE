import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils.modal_fusion import DenoisingModule


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块，允许不同模态之间的双向注意力机制
    
    Args:
        embed_dim: 嵌入维度
        num_heads: 注意力头数
        dropout: Dropout比例
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须能被头数整除"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
        # 初始化参数
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len_q, embed_dim]
            key: 键张量 [batch_size, seq_len_k, embed_dim]
            value: 值张量 [batch_size, seq_len_v, embed_dim]
            mask: 可选掩码张量
            
        Returns:
            output: 注意力输出 [batch_size, seq_len_q, embed_dim]
            attention_weights: 注意力权重
        """
        batch_size = query.shape[0]
        
        # 线性投影
        q = self.q_proj(query)  # [batch_size, seq_len_q, embed_dim]
        k = self.k_proj(key)    # [batch_size, seq_len_k, embed_dim]
        v = self.v_proj(value)  # [batch_size, seq_len_v, embed_dim]
        
        # 重塑为多头形式
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len_q, head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len_k, head_dim]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len_v, head_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len_q, seq_len_k]
        
        # 应用掩码（如果提供）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_weights = self.dropout(attention_weights)
        
        # 计算注意力输出
        output = torch.matmul(attention_weights, v)  # [batch_size, num_heads, seq_len_q, head_dim]
        
        # 重塑回原始形状
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)  # [batch_size, seq_len_q, embed_dim]
        
        # 最终线性投影
        output = self.o_proj(output)
        
        return output, attention_weights


class GatedFusion(nn.Module):
    """
    门控融合模块，使用门控机制融合不同模态
    
    Args:
        embed_dim: 嵌入维度
    """
    def __init__(self, embed_dim, dropout=0.1):
        super(GatedFusion, self).__init__()
        self.gate_v = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, visual_embed, text_embed):
        """
        前向传播
        
        Args:
            visual_embed: 视觉嵌入 [batch_size, embed_dim]
            text_embed: 文本嵌入 [batch_size, embed_dim]
            
        Returns:
            fused_embed: 融合后的嵌入 [batch_size, embed_dim]
        """
        # 连接嵌入用于门控计算
        concat_embed = torch.cat([visual_embed, text_embed], dim=-1)
        
        # 计算视觉和文本门控值
        gate_v_value = self.gate_v(concat_embed)
        gate_t_value = self.gate_t(concat_embed)
        
        # 应用门控
        gated_v = visual_embed * gate_v_value
        gated_t = text_embed * gate_t_value
        
        # 连接门控后的特征
        gated_concat = torch.cat([gated_v, gated_t], dim=-1)
        
        # 通过MLP融合
        fusion_embed = self.fusion_mlp(gated_concat)
        
        # 残差连接和层归一化
        fusion_embed = self.layer_norm(fusion_embed + (visual_embed + text_embed) / 2)
        
        return fusion_embed


class BilinearFusion(nn.Module):
    """
    双线性融合模块，使用双线性变换捕捉模态间的高阶交互
    
    Args:
        embed_dim: 嵌入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: Dropout比例
    """
    def __init__(self, embed_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super(BilinearFusion, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // 2
        if output_dim is None:
            output_dim = embed_dim
            
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        self.t_proj = nn.Linear(embed_dim, hidden_dim)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 用于残差连接的线性映射
        self.residual_proj = nn.Linear(embed_dim * 2, output_dim)
        
    def forward(self, visual_embed, text_embed):
        """
        前向传播
        
        Args:
            visual_embed: 视觉嵌入 [batch_size, embed_dim]
            text_embed: 文本嵌入 [batch_size, embed_dim]
            
        Returns:
            fused_embed: 融合后的嵌入 [batch_size, output_dim]
        """
        # 投影到隐藏空间
        v_proj = self.v_proj(visual_embed)
        t_proj = self.t_proj(text_embed)
        
        # 双线性变换
        bilinear_output = self.bilinear(v_proj, t_proj)
        bilinear_output = self.dropout(bilinear_output)
        
        # 残差连接
        residual_input = torch.cat([visual_embed, text_embed], dim=-1)
        residual_output = self.residual_proj(residual_input)
        
        # 融合输出
        fused_embed = self.layer_norm(bilinear_output + residual_output)
        
        return fused_embed


class ModalFusion(nn.Module):
    """
    模态融合模块，提供多种融合方法的统一接口
    
    Args:
        embed_dim: 嵌入维度
        fusion_type: 融合类型，可选 'attention', 'gated', 'bilinear', 'concat'
        num_heads: 注意力头数（用于attention融合）
        dropout: Dropout比例
    """
    def __init__(self, embed_dim, fusion_type='attention', num_heads=4, dropout=0.1):
        super(ModalFusion, self).__init__()
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        
        if fusion_type == 'attention':
            self.fusion_module = CrossModalAttention(embed_dim, num_heads, dropout)
        elif fusion_type == 'gated':
            self.fusion_module = GatedFusion(embed_dim, dropout)
        elif fusion_type == 'bilinear':
            self.fusion_module = BilinearFusion(embed_dim, dropout=dropout)
        elif fusion_type == 'concat':
            self.fusion_module = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
            
        # 最终的投影层
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
            
    def forward(self, visual_embed, text_embed):
        """
        前向传播
        
        Args:
            visual_embed: 视觉嵌入 [batch_size, embed_dim]
            text_embed: 文本嵌入 [batch_size, embed_dim]
            
        Returns:
            fused_embed: 融合后的嵌入 [batch_size, embed_dim]
        """
        if self.fusion_type == 'attention':
            # 双向注意力融合
            v2t_output, _ = self.fusion_module(visual_embed.unsqueeze(1), 
                                             text_embed.unsqueeze(1), 
                                             text_embed.unsqueeze(1))
            t2v_output, _ = self.fusion_module(text_embed.unsqueeze(1), 
                                             visual_embed.unsqueeze(1), 
                                             visual_embed.unsqueeze(1))
            
            # 去除序列维度
            v2t_output = v2t_output.squeeze(1)
            t2v_output = t2v_output.squeeze(1)
            
            # 结合双向注意力的结果
            fused_embed = (v2t_output + t2v_output) / 2
            
        elif self.fusion_type == 'gated' or self.fusion_type == 'bilinear':
            fused_embed = self.fusion_module(visual_embed, text_embed)
            
        elif self.fusion_type == 'concat':
            concat_embed = torch.cat([visual_embed, text_embed], dim=-1)
            fused_embed = self.fusion_module(concat_embed)
            
        # 最终投影
        fused_embed = self.final_proj(fused_embed)
        
        return fused_embed


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块，动态调整不同模态的融合权重
    
    Args:
        embed_dim: 嵌入维度
        dropout: Dropout比例
    """
    def __init__(self, embed_dim, dropout=0.1):
        super(AdaptiveFusion, self).__init__()
        
        # 用于计算模态重要性权重的MLP
        self.weight_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # 用于融合的投影层
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, visual_embed, text_embed, item_id_embed=None):
        """
        前向传播
        
        Args:
            visual_embed: 视觉嵌入 [batch_size, embed_dim]
            text_embed: 文本嵌入 [batch_size, embed_dim]
            item_id_embed: 物品ID嵌入 [batch_size, embed_dim]（可选）
            
        Returns:
            fused_embed: 融合后的嵌入 [batch_size, embed_dim]
        """
        # 连接模态特征以计算权重
        concat_embed = torch.cat([visual_embed, text_embed], dim=-1)
        
        # 计算模态权重
        modal_weights = self.weight_mlp(concat_embed)  # [batch_size, 2]
        
        # 加权融合
        v_weight, t_weight = modal_weights.chunk(2, dim=-1)
        weighted_fusion = v_weight * visual_embed + t_weight * text_embed
        
        # 如果提供了ID嵌入，则将其考虑在内
        if item_id_embed is not None:
            # 加入与ID嵌入的乘积交互，增强特征表达
            v_enhanced = torch.mul(visual_embed, item_id_embed)
            t_enhanced = torch.mul(text_embed, item_id_embed)
            
            # 重新计算融合权重
            enhanced_concat = torch.cat([v_enhanced, t_enhanced], dim=-1)
            enhanced_weights = self.weight_mlp(enhanced_concat)
            v_enh_weight, t_enh_weight = enhanced_weights.chunk(2, dim=-1)
            
            # 混合原始融合和增强融合
            enhanced_fusion = v_enh_weight * v_enhanced + t_enh_weight * t_enhanced
            weighted_fusion = (weighted_fusion + enhanced_fusion) / 2
        
        # 通过投影层和层归一化
        fused_embed = self.fusion_proj(weighted_fusion)
        fused_embed = self.layer_norm(fused_embed + (visual_embed + text_embed) / 2)  # 残差连接
        
        return fused_embed


# 辅助函数，用于批量处理大规模嵌入
def batch_fusion(fusion_module, embed1, embed2, batch_size=1024):
    """
    批量处理大规模嵌入的融合，以避免内存溢出
    
    Args:
        fusion_module: 融合模块
        embed1: 第一个嵌入 [num_items, embed_dim]
        embed2: 第二个嵌入 [num_items, embed_dim]
        batch_size: 批次大小
        
    Returns:
        fused_embeds: 融合后的嵌入 [num_items, embed_dim]
    """
    num_items = embed1.size(0)
    embed_dim = embed1.size(1)
    device = embed1.device
    
    # 创建输出张量
    fused_embeds = torch.zeros(num_items, embed_dim, device=device)
    
    # 批量处理
    for i in range(0, num_items, batch_size):
        end_idx = min(i + batch_size, num_items)
        batch_embed1 = embed1[i:end_idx]
        batch_embed2 = embed2[i:end_idx]
        
        # 融合当前批次
        batch_fused = fusion_module(batch_embed1, batch_embed2)
        fused_embeds[i:end_idx] = batch_fused
        
    return fused_embeds