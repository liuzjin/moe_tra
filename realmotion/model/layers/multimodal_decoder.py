import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from realmotion.model.layers.transformer_blocks import Block, InterBlock

class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps, return_prob=True) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.return_prob = return_prob

        self.multimodal_proj = nn.Linear(embed_dim, 6 * embed_dim)

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, future_steps * 2),
        )
        if return_prob:
            self.pi = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
            )

    def forward(self, x):
        x = self.multimodal_proj(x).view(-1, 6, self.embed_dim)
        loc = self.loc(x).view(-1, 6, self.future_steps, 2)
        if self.return_prob:
            pi = self.pi(x).squeeze(-1)
            probs = F.softmax(pi, dim=-1)
        else:
            pi = None

        
        return {"predictions": loc, # (B, top_k, T, 2) -> Top-K的轨迹
                "logits": pi, 
                "probs": probs,
                 "mode": x }
    
class MLPExpert(nn.Module):
    def __init__(self, d_model: int, prediction_horizon: int):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        hidden_dim = d_model * 2 

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_horizon * 2) 
        )

    def forward(self, expert_feature):

        # (Batch, d_model) -> (Batch, prediction_horizon * 2)
        flat_trajectory = self.mlp(expert_feature)
        
        # (Batch, prediction_horizon * 2) -> (Batch, prediction_horizon, 2)
        predicted_trajectory = flat_trajectory.view(-1, self.prediction_horizon, 2)
        
        return predicted_trajectory

class MoeDecoder(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 future_steps: int, 
                 num_experts: int, 
                 top_k: int,
                 intent_label: bool = True) -> None: # <<< 核心改动 1: 增加 intent_label 参数
        super().__init__()
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_experts = num_experts
        self.top_k = top_k
        self.intent_label = intent_label # 保存训练模式
        
        # --- 模块定义 (保持不变) ---
        self.multimodal_feature_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * num_experts),
            nn.LayerNorm(embed_dim * num_experts)
        )
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_experts)
        )
        self.experts = nn.ModuleList([
            MLPExpert(embed_dim, future_steps)
            for _ in range(self.num_experts)
        ])

    def forward(self, encoder_out: torch.Tensor, training: bool, key_padding_mask=None) -> Dict:
        # 假设输入来自编码器，我们只取第一个[CLS] token的特征
        x = encoder_out[:, 0].unsqueeze(1) # 保持 (B, 1, D) 的形状
        x_squeezed = x.squeeze(1)
        
        expert_features = self.multimodal_feature_generator(x_squeezed).view(-1, self.num_experts, self.embed_dim)
        
        # --- 计算门控logits (保持不变) ---
        logits = self.gating_network(x_squeezed) # (B, num_experts)
        
        if training:
            predictions = []
            for i in range(self.num_experts):
                current_expert_feature = expert_features[:, i, :]
                pred = self.experts[i](current_expert_feature)
                predictions.append(pred)
            all_predictions = torch.stack(predictions, dim=1)
            
            # --- 核心改动 3: 根据 intent_label 准备输出字典 ---
            output = {
                "predictions": all_predictions,
                "logits": logits,  # 返回原始logits, 兼容两种损失函数
                "mode": expert_features
            }
            
            # 如果是无标签模式 (intent_label=False), 则计算并添加 aux_loss
            if not self.intent_label:
                probs = F.softmax(logits, dim=-1)
                # 计算负载均衡损失
                mean_probs = torch.mean(probs, dim=0)
                aux_loss = self.num_experts * torch.sum(mean_probs * mean_probs)
                output["aux_loss"] = aux_loss
                
            return output
        else:
            # 简化推理返回，直接返回字典
            probs = F.softmax(logits, dim=-1) # 推理时使用probs
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
            batch_size = x.shape[0]
            top_k_predictions = torch.zeros(batch_size, self.top_k, self.future_steps, 2, device=x.device)
            expert_features = self.multimodal_feature_generator(x_squeezed).view(-1, self.num_experts, self.embed_dim)

            # 这里的循环为了保持与您原始代码的逻辑一致性。
            # 推荐未来优化为torch.gather。
            for i in range(batch_size):
                for j in range(self.top_k):
                    expert_idx = top_k_indices[i, j]
                    feature = expert_features[i, expert_idx, :]
                    top_k_predictions[i, j] = self.experts[expert_idx](feature)
                    
            return {
                "predictions": top_k_predictions,
                "probs": top_k_probs,
                "top_k_indices": top_k_indices,
                "mode": expert_features
            }

class QueryBasedMoeDecoder(nn.Module):
    def __init__(self, 
                 dim=128, 
                 mlp_ratio = 4.0,
                 qkv_bias = False,
                 drop = 0.2,
                 attn_drop = 0.2,
                 drop_path= 0.2,
                 query_cross_layers=1,
                 query_self_atten=True,
                 query_self_layers=1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 future_steps=10, 
                 future_len=60,
                 num_experts=9, 
                 top_k=2,
                 num_heads=8,
                 intent_label=True): # 增加了注意力头的数量作为参数
        super().__init__()
        self.embed_dim = dim
        self.future_steps = future_steps
        self.num_experts = num_experts
        self.top_k = top_k
        self.intent_label = intent_label
        self.num_segments = future_len // future_steps
        self.num_modes = top_k

        # --- 1. 可学习的意图查询向量 ---
        # 这是新架构的核心。每个向量将学会代表一种特定的驾驶意图。
        # --- 1. 时序分段查询 (不变) ---
        self.segment_queries = nn.Parameter(torch.randn(1, self.num_segments, self.embed_dim))
        self.mode_queries = nn.Parameter(torch.randn(1, self.num_modes, self.embed_dim))

        # --- 2. 交叉注意力层 ---
        # 这个层将使用意图查询(Query)来从上下文(Key, Value)中提取信息。
        self.query_cross_blocks =nn.ModuleList(InterBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))
        self.query_self_atten = query_self_atten
        if query_self_atten:
            self.mode_self_attn_blocks = nn.ModuleList( Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path,
                ) for i in range(query_self_layers))
        
            
        # --- 3. Logit生成器 ---
        # 从交叉注意力的输出（即每个专家的专业化特征）中，计算出该专家的得分。
        self.mode_cross_attn_blocks = nn.ModuleList(InterBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))
        # --- 4. 门控网络头 (Gating Head) ---
        # 输入一个模态特征(D)，输出一个完整的专家序列(S*N)
        self.gating_head = nn.Linear(self.embed_dim, self.num_segments * self.num_experts)

        # --- 5. 全局模态概率头 (新增) ---
        # 输入K个模态特征，输出K个概率
        self.mode_prob_head = nn.Linear(self.embed_dim, 1)
        # --- 4. 专家网络列表 (与之前相同) ---
        self.experts = nn.ModuleList([
            MLPExpert(self.embed_dim, self.future_steps)
            for _ in range(self.num_experts)
        ])

    def forward(self, context, training= True, key_padding_mask=None, gt_intent_sequence=None):
        batch_size = context.shape[0]
        
        # --- 步骤 1: 生成时序分段特征 (和之前一样) ---
        segment_queries_batch = self.segment_queries.expand(batch_size, -1, -1)
        segment_features = segment_queries_batch
        for blk in self.query_cross_blocks:
            segment_features = blk(src=segment_features, src_kv=context, key_padding_mask=key_padding_mask)
        # segment_features shape: (B, S, D)

        # --- 步骤 2: 生成 K 个多模态特征 ---
        mode_queries_batch = self.mode_queries.expand(batch_size, -1, -1) # (B, K, D)
        
        # K个模态查询与S个分段特征进行交叉注意力，为每个模态规划专家路径
        mode_features = mode_queries_batch
        for blk in self.mode_cross_attn_blocks:
            mode_features = blk(src=mode_features, src_kv=segment_features) # Q=mode_queries, K/V=segment_features
        
        if self.query_self_atten:
            for blk in self.mode_self_attn_blocks:
                mode_features = blk(src=mode_features)
        # mode_features shape: (B, K, D)
        
        # --- 步骤 3: 计算 K 个模态的全局概率 ---
        # (B, K, D) -> (B, K, 1) -> (B, K)
        mode_logits = self.mode_prob_head(mode_features).squeeze(-1)
        mode_probs = F.softmax(mode_logits, dim=-1)

        # --- 步骤 4: 为 K 个模态分别生成专家选择序列 ---
        # (B, K, D) -> (B, K, S*N) -> (B, K, S, N)
        segment_logits_per_mode = self.gating_head(mode_features).view(
            batch_size, self.num_modes, self.num_segments, self.num_experts
        )
        
        # --- 步骤 5: 路由与预测 K 条轨迹 ---
        features_flat = segment_features.reshape(-1, self.embed_dim) # (B*S, D)
        
        all_modal_trajs = []
        for k in range(self.num_modes):
            # 获取当前模态 k 的分段logits: (B, S, N)
            segment_logits = segment_logits_per_mode[:, k, :, :]
            
            # (B, S, N) -> (B*S, N)
            logits_flat = segment_logits.reshape(-1, self.num_experts)

            # --- 根据 intent_label 标志选择路由策略 ---
            if self.intent_label:
                # --- 策略A: Top-1 硬路由 (适用于意图监督) ---
                expert_indices = None
                if training and gt_intent_sequence is not None:
                    # 训练时使用真值意图强制路由
                    expert_indices = gt_intent_sequence.reshape(-1)
                else:
                    # 推理时使用模型自己的Top-1预测
                    expert_indices = torch.argmax(logits_flat, dim=-1)

                segment_predictions = torch.zeros(
                    batch_size * self.num_segments, self.future_steps * 2, device=context.device
                )
                for i in range(self.num_experts):
                    mask = (expert_indices == i)
                    if mask.any():
                        features_for_expert = features_flat[mask]
                        expert_output = self.experts[i](features_for_expert)
                        segment_predictions.masked_scatter_(mask.unsqueeze(-1), expert_output)
            
            else:
                # --- 策略B: Top-k 软路由 (无监督模式) ---
                probs_flat = F.softmax(logits_flat, dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs_flat, self.top_k, dim=-1)
                top_k_probs = F.normalize(top_k_probs, p=1, dim=-1) # (B*S, top_k)

                # segment_predictions = torch.zeros(
                #     batch_size * self.num_segments, self.future_steps * 2, device=context.device
                # )
                temp_predictions = torch.zeros(
                batch_size * self.num_segments, self.top_k, self.future_steps, 2, device=context.device
                )
                for i in range(self.num_experts):
                    expert_mask = (top_k_indices == i)
                    row_indices, col_indices = expert_mask.nonzero(as_tuple=True)
                    if row_indices.numel() > 0:
                        features_for_expert = features_flat[row_indices]
                        expert_output = self.experts[i](features_for_expert)
                        # gates_for_expert = top_k_probs[row_indices, col_indices].unsqueeze(1)
                        # segment_predictions.index_add_(0, row_indices, expert_output * gates_for_expert)
                        temp_predictions[row_indices, col_indices] = expert_output
                weighted_predictions = top_k_probs.unsqueeze(-1).unsqueeze(-1) * temp_predictions
                # (B*S, top_k, steps*2) -> (B*S, steps*2)
                segment_predictions = torch.sum(weighted_predictions, dim=1)
            # --- 拼接成一条完整的轨迹 (两种策略通用) ---
            final_traj = segment_predictions.view(
                batch_size, self.num_segments, self.future_steps, 2
            ).reshape(batch_size, -1, 2)
            all_modal_trajs.append(final_traj)

        # --- 准备输出 (与之前相同) ---
        final_predictions = torch.stack(all_modal_trajs, dim=1)
        mode_probs = F.softmax(mode_logits, dim=-1)
        output = {
            "predictions": final_predictions,
            "probs": mode_probs,
            "logits": mode_logits,
            "segment_logits_per_mode": segment_logits_per_mode
        }
        
        return output