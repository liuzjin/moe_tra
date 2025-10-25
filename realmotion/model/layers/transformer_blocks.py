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

class SparseMoE_MLP(nn.Module):
    """
    一个稀疏混合专家MLP层，专门用于替换Transformer块中的标准FFN。
    """
    def __init__(self, in_features: int, hidden_features: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.in_features = in_features
        self.hidden_features = hidden_features

        # 1. 门控网络: 输入(D)，输出(N_experts)
        self.gate = nn.Linear(in_features, num_experts)
        
        # 2. 专家列表: 每个专家都是一个标准的MLP/FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.GELU(), # 或者其他激活函数
                nn.Linear(hidden_features, in_features)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, SeqLen, Dim)
        batch_size, seq_len, dim = x.shape
        
        # (B, N, D) -> (B*N, D)
        x_flat = x.reshape(-1, dim)
        
        # --- 门控 ---
        # (B*N, D) -> (B*N, N_experts)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        
        # --- Top-k 路由 ---
        # top_k_probs: (B*N, K), top_k_indices: (B*N, K)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # 归一化权重
        top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)

        # --- Top-k 专家加权 ---
        # 初始化一个临时张量来存储k个专家的输出
        # (B*N, K, D)
        temp_expert_outputs = torch.zeros(
            batch_size * seq_len, self.top_k, dim, device=x.device
        )

        for i in range(self.num_experts):
            # 找到哪些 token 的 top-k 选择中包含了当前专家 i
            expert_mask = (top_k_indices == i)
            row_indices, col_indices = expert_mask.nonzero(as_tuple=True)

            if row_indices.numel() > 0:
                # 提取需要由专家 i 处理的 token
                features_for_expert = x_flat[row_indices]
                # 运行专家网络
                expert_output = self.experts[i](features_for_expert)
                # 将结果填充到临时张量的正确位置
                temp_expert_outputs[row_indices, col_indices] = expert_output
        
        # 加权求和
        # (B*N, K, 1) * (B*N, K, D) -> (B*N, K, D)
        weighted_outputs = top_k_probs.unsqueeze(-1) * temp_expert_outputs
        # (B*N, K, D) -> (B*N, D)
        y_flat = torch.sum(weighted_outputs, dim=1)

        # Reshape回原始形状
        # (B*N, D) -> (B, N, D)
        y = y_flat.reshape(batch_size, seq_len, dim)
        
        # (可选) 返回负载均衡损失，用于训练
        # aux_loss = ... 
        
        return y

class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, use_moe_ffn=False, num_experts=8, top_k=2, **kwargs):
        super().__init__()
        # --- 交叉注意力 ---
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, **kwargs)
        self.cross_attn_dropout = nn.Dropout(0.1)

        # --- 自注意力 ---
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, **kwargs)
        self.self_attn_dropout = nn.Dropout(0.1)

        # --- FFN / MoE-FFN ---
        self.ffn_norm = nn.LayerNorm(dim)
        if use_moe_ffn:
            self.ffn = SparseMoE_MLP(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                num_experts=num_experts,
                top_k=top_k
            )
        else:
            # 标准的FFN
            self.ffn = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(int(dim * mlp_ratio), dim)
            )
        self.ffn_dropout = nn.Dropout(0.1)

    def forward(self, queries, context, context_key_padding_mask=None):
        # 1. 交叉注意力
        q = self.cross_attn_norm(queries)
        attn_out, _ = self.cross_attn(
            query=q, key=context, value=context, key_padding_mask=context_key_padding_mask
        )
        queries = queries + self.cross_attn_dropout(attn_out)

        # 2. 自注意力
        q = self.self_attn_norm(queries)
        attn_out, _ = self.self_attn(query=q, key=q, value=q)
        queries = queries + self.self_attn_dropout(attn_out)

        # 3. FFN / MoE-FFN
        x = self.ffn_norm(queries)
        ffn_out = self.ffn(x)
        queries = queries + self.ffn_dropout(ffn_out)
        
        return queries
