from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor
from .mln import MLN, nerf_positional_encoding
import torch.nn.functional as F

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.2,
        attn_drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
    ):
        super().__init__()
        self.post_norm = post_norm

        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2 = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward_post(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.attn(
            query=src,
            key=src,
            value=src,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        return src

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        if self.post_norm:
            return self.forward_post(
                src=src, mask=mask, key_padding_mask=key_padding_mask
            )

        return self.forward_pre(src=src, mask=mask, key_padding_mask=key_padding_mask)



class InteractionModule(nn.Module):
    def __init__(
        self,
        dim,
        pose_dim,
        num_heads,
        num_blocks=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.2,
        attn_drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.mln = MLN(pose_dim * 12)
        self.inter_blocks = nn.ModuleList(
            InterBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            for i in range(num_blocks)
        )

    def forward(
        self,
        cur_embed,
        memory_embed,
        cur_pose,
        memory_pose,
        cur_pos_embed=None,
        memory_pos_embed=None,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        cur_pose = nerf_positional_encoding(cur_pose).unsqueeze(1).repeat(1, cur_embed.size(1), 1)
        memory_pose = nerf_positional_encoding(memory_pose).unsqueeze(1).repeat(1, memory_embed.size(1), 1)
        cur_embed = self.mln(cur_embed, cur_pose)
        memory_embed = self.mln(memory_embed, memory_pose)
        if cur_pos_embed is not None:
            cur_embed += cur_pos_embed
        if memory_pos_embed is not None:
            memory_embed += memory_pos_embed
        for blk in self.inter_blocks:
            cur_embed = blk(src=cur_embed, src_kv=memory_embed, mask=mask, key_padding_mask=key_padding_mask)
        return cur_embed


class InterBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.2,
        attn_drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2_kv = self.norm1(src_kv)
        src2 = self.attn(
            query=src2,
            key=src2_kv,
            value=src2_kv,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# 高性能、向量化的专家混合 (MoE) 模块
class MoE(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([MLP(dim, mlp_ratio) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        x_flat = x.reshape(-1, D)
        num_tokens = x_flat.shape[0]

        gate_logits = self.gate(x_flat)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)

        y_flat = torch.zeros_like(x_flat)
        flat_indices = indices.view(-1)
        repeated_tokens = x_flat.repeat_interleave(self.top_k, dim=0)

        for i in range(self.num_experts):
            mask = (flat_indices == i)
            if mask.any():
                expert_inputs = repeated_tokens[mask]
                expert_outputs = self.experts[i](expert_inputs)
                weights_for_expert = weights.view(-1)[mask].unsqueeze(1)
                weighted_outputs = expert_outputs * weights_for_expert
                original_positions = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)[mask]
                y_flat.scatter_add_(0, original_positions.unsqueeze(1).expand(-1, D), weighted_outputs)

        return y_flat.view(B, N, D)

class DecoderLayer(nn.Module):
    def __init__(self,
                 dim=128,
                 num_heads=8,
                 mlp_ratio=4.0,
                 num_experts=8,
                 top_k=2,
                 **kwargs):
        super().__init__()
        
        # --- 模块1: 交叉注意力 ---
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # --- 模块2: 专家混合网络 (MoE) ---
        # 注意：这里我们使用 MoE 来代替标准的 FFN
        self.moe = MoE(dim, num_experts=num_experts, top_k=top_k, mlp_ratio=mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        
        # --- 模块3: 自注意力 ---
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        
        # --- 模块4: 最终的前馈网络 (FFN) ---
        self.ffn = MLP(dim, mlp_ratio=mlp_ratio)
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, queries, context, context_key_padding_mask: Optional[torch.Tensor] = None):
        # queries: (B, K, D) - 多模态查询
        # context: (B, N, D) - 编码器输出的场景信息

        # 1. 首先进行交叉注意力，让queries吸收场景信息
        cross_attn_output, _ = self.cross_attn(query=queries,
                                               key=context,
                                               value=context,
                                               key_padding_mask=context_key_padding_mask)
        # Add & Norm
        queries = self.norm1(queries + cross_attn_output)
        
        # 2. 通过MoE层进行高容量的特征变换
        moe_output = self.moe(queries)
        # Add & Norm
        queries = self.norm2(queries + moe_output)
        
        # 3. 然后进行自注意力，让不同模态的queries之间进行信息交互
        self_attn_output, _ = self.self_attn(query=queries,
                                             key=queries,
                                             value=queries)
        # Add & Norm
        queries = self.norm3(queries + self_attn_output)
        
        # 4. 最后通过一个标准的FFN进行特征提炼
        ffn_output = self.ffn(queries)
        # Add & Norm
        queries = self.norm4(queries + ffn_output)
        
        return queries