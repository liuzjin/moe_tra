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
    def __init__(self, d_model: int, prediction_horizon: int, drop: int):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        hidden_dim = d_model * 2 

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, prediction_horizon * 2) 
        )

    def forward(self, expert_feature):

        # (Batch, d_model) -> (Batch, prediction_horizon * 2)
        flat_trajectory = self.mlp(expert_feature)
        
        # (Batch, prediction_horizon * 2) -> (Batch, prediction_horizon, 2)
        predicted_trajectory = flat_trajectory.view(-1, self.prediction_horizon, 2)
        
        return predicted_trajectory

class SimpleSegmentalMoeDecoder(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 mlp_dim: int,
                 future_steps: int,
                 future_len: int = 60,
                 num_experts: int = 9, 
                 num_modes: int = 6, # K
                 top_k: int = 2,
                 intent_label: bool = True,
                 drop: int = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.steps_per_segment = future_steps
        self.num_experts = num_experts
        self.num_modes = num_modes # K
        self.num_segments = future_len // future_steps
        self.top_k = top_k
        self.intent_label = intent_label
        self.total_future_steps = self.num_segments * self.steps_per_segment

        # --- 1. 全局模态概率头 (MLP-based) ---
        # 输入全局场景编码(D)，输出K个模态的logits
        self.mode_prob_network = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, num_modes),
        )

        # --- 2. 分段门控网络 (MLP-based) ---
        # 输入全局场景编码(D)，一次性生成所有K个模态、所有S个分段的专家logits
        self.segment_gating_network = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim * 2),
            nn.LayerNorm(mlp_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim * 2, num_modes * self.num_segments * self.num_experts),
            
        )

        # --- 3. 专家网络列表 (保持不变) ---
        # 专家依然是简单的MLP，但它们的输入现在需要变一下
        # 我们需要为每个分段生成一个独特的特征
        self.segment_feature_generator = nn.Sequential(
            nn.Linear(embed_dim, self.num_segments * self.mlp_dim),
            nn.LayerNorm(self.num_segments * self.mlp_dim),
            nn.ReLU(),
            nn.Dropout(drop) # <<< 4. 在激活函数后添加Dropout
        )

        self.experts = nn.ModuleList([
            MLPExpert(self.mlp_dim, self.steps_per_segment, drop)
            for _ in range(self.num_experts)
        ])

    def forward(self, 
                encoder_out: torch.Tensor, 
                training: bool = True, 
                **kwargs) -> Dict:
        
        batch_size = encoder_out.shape[0]
        
        # --- 步骤 1: 提取全局场景编码 ---
        # 假设我们使用编码器输出的第一个 token ([CLS] token)
        # scene_context shape: (B, D)
        scene_context = encoder_out[:, 0]
        
        # --- 步骤 2: 计算 K 个模态的全局概率 ---
        # (B, D) -> (B, K)
        mode_logits = self.mode_prob_network(scene_context)
        mode_probs = F.softmax(mode_logits, dim=-1)

        # --- 步骤 3: 生成所有分段的专家选择序列 ---
        # (B, D) -> (B, K*S*N) -> (B, K, S, N)
        segment_logits_per_mode = self.segment_gating_network(scene_context).view(
            batch_size, self.num_modes, self.num_segments, self.num_experts
        )
        
        # --- 步骤 4: 为专家生成分段特征 ---
        # (B, D) -> (B, S*D) -> (B, S, D)
        segment_features = self.segment_feature_generator(scene_context).view(
            batch_size, self.num_segments, self.embed_dim
        )
        # (B, S, D) -> (B*S, D)
        features_flat = segment_features.reshape(-1, self.embed_dim)

        # --- 步骤 5: (统一的) Top-k 软路由与预测 K 条轨迹 ---
        all_modal_trajs = []
        for k in range(self.num_modes):
            # 获取当前模态 k 的分段logits: (B, S, N)
            segment_logits = segment_logits_per_mode[:, k, :, :]
            # (B, S, N) -> (B*S, N)
            logits_flat = segment_logits.reshape(-1, self.num_experts)
            
            # --- 始终执行 Top-k 软路由 ---
            probs_flat = F.softmax(logits_flat, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs_flat, self.top_k, dim=-1)
            top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)

            # 临时张量用于存储 top-k 个专家的原始预测
            temp_predictions = torch.zeros(
                batch_size * self.num_segments, self.top_k, self.steps_per_segment, 2, device=encoder_out.device
            )
            
            for i in range(self.num_experts):
                expert_mask = (top_k_indices == i)
                row_indices, col_indices = expert_mask.nonzero(as_tuple=True)

                if row_indices.numel() > 0:
                    features_for_expert = features_flat[row_indices]
                    expert_output = self.experts[i](features_for_expert).view(-1, self.steps_per_segment, 2)
                    temp_predictions[row_indices, col_indices] = expert_output
            
            # 使用 top-k 概率进行加权求和
            weighted_predictions = top_k_probs.unsqueeze(-1).unsqueeze(-1) * temp_predictions
            segment_predictions_flat = torch.sum(weighted_predictions, dim=1)

            # --- 拼接成一条完整的轨迹 ---
            final_traj = segment_predictions_flat.view(
                batch_size, self.num_segments, self.steps_per_segment, 2
            ).reshape(batch_size, self.total_future_steps, 2)
            all_modal_trajs.append(final_traj)

        # --- 步骤 6: 准备输出 ---
        final_predictions = torch.stack(all_modal_trajs, dim=1)
        
        output = {
            "predictions": final_predictions,
            "pi": mode_probs,
            "logits": mode_logits,
            "segment_logits_per_mode": segment_logits_per_mode
        }
        
        return output

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

    def forward(self, context, training= True, key_padding_mask=None):
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

            # --- 始终执行 Top-k 软路由 ---
            probs_flat = F.softmax(logits_flat, dim=-1)
            # (B*S, top_k)
            top_k_probs, top_k_indices = torch.topk(probs_flat, self.top_k, dim=-1)
            # 归一化权重
            top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)

            # 临时张量用于存储 top-k 个专家的原始预测
            # (B*S, top_k, steps, 2)
            temp_predictions = torch.zeros(
                batch_size * self.num_segments, self.top_k, self.future_steps, 2, device=context.device
            )
            
            for i in range(self.num_experts):
                # 找到哪些分段的 top-k 选择中包含了当前专家 i
                expert_mask = (top_k_indices == i) # (B*S, top_k)
                row_indices, col_indices = expert_mask.nonzero(as_tuple=True)

                if row_indices.numel() > 0:
                    features_for_expert = features_flat[row_indices]
                    # 专家输出 (num_non_zero, steps*2) -> (num_non_zero, steps, 2)
                    expert_output = self.experts[i](features_for_expert).view(-1, self.future_steps, 2)
                    
                    # 将预测结果填充到临时张量的正确位置
                    temp_predictions[row_indices, col_indices] = expert_output
            
            # 使用 top-k 概率进行加权求和
            # (B*S, top_k, 1, 1) * (B*S, top_k, steps, 2) -> (B*S, top_k, steps, 2)
            weighted_predictions = top_k_probs.unsqueeze(-1).unsqueeze(-1) * temp_predictions
            # (B*S, top_k, steps, 2) -> (B*S, steps, 2)
            segment_predictions_flat = torch.sum(weighted_predictions, dim=1)

            # --- 拼接成一条完整的轨迹 ---
            final_traj = segment_predictions_flat.view(
                batch_size, self.num_segments, self.future_steps, 2
            ).reshape(batch_size, -1, 2)
            all_modal_trajs.append(final_traj)

        # --- 步骤 6: 准备输出 ---
        final_predictions = torch.stack(all_modal_trajs, dim=1)
        mode_probs = F.softmax(mode_logits, dim=-1)
        
        output = {
            "predictions": final_predictions,   # (B, K, T, 2)
            "probs": mode_probs,                # (B, K)
            "logits": mode_logits,              # (B, K)
            # 始终返回分段logits，由损失函数决定如何使用
            "segment_logits_per_mode": segment_logits_per_mode # (B, K, S, N)
        }
        
        return output

class IndependentSubMoE(nn.Module):
    def __init__(self, embed_dim: int, future_steps: int, num_sub_experts: int, top_k_micro: int):
        super().__init__()
        self.num_sub_experts = num_sub_experts
        self.top_k_micro = top_k_micro
        self.future_steps = future_steps

        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.num_sub_experts)
        )
        self.experts = nn.ModuleList([
            MLPExpert(embed_dim, future_steps) for _ in range(num_sub_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, D)
        
        # 1. 内部微观门控
        sub_logits = self.gating_network(x) # (B, N_sub)
        sub_probs = F.softmax(sub_logits, dim=-1)
        
        # 2. 选择内部Top-k专家
        top_k_probs, top_k_indices = torch.topk(sub_probs, self.top_k_micro, dim=-1)
        top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)

        # 3. Top-k专家加权预测
        temp_predictions = torch.zeros(
            x.shape[0], self.top_k_micro, self.future_steps, 2, device=x.device
        )
        
        for i in range(self.num_sub_experts):
            mask = (top_k_indices == i)
            row_indices, col_indices = mask.nonzero(as_tuple=True)
            if row_indices.numel() > 0:
                features_for_expert = x[row_indices]
                expert_output = self.experts[i](features_for_expert).view(-1, self.future_steps, 2)
                temp_predictions[row_indices, col_indices] = expert_output
        
        weighted_predictions = top_k_probs.unsqueeze(-1).unsqueeze(-1) * temp_predictions
        final_trajectory = torch.sum(weighted_predictions, dim=1) # (B, T, 2)
        
        return final_trajectory

class HierarchicalGatingDecoder(nn.Module):
    def __init__(self, 
                 embed_dim: int, 
                 future_steps: int,
                 intents: int = 9, 
                 num_experts: int = 4,
                 top_k: int = 2,
                 modes: int = 6):  # K_macro, 推理时输出的模态数
        super().__init__()
        self.embed_dim = embed_dim
        self.intents = intents
        self.modes = modes # K
        self.future_steps = future_steps
        
        # --- 1. 顶层宏观意图门控 ---
        self.top_level_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, self.intents)
        )

        # --- 2. 独立的、专家不共享的子MoE系统 ---
        self.sub_systems = nn.ModuleList([
            IndependentSubMoE(
                    embed_dim=embed_dim,
                    future_steps=future_steps,
                    num_sub_experts=num_experts,
                    top_k_micro=top_k
                ) for i in range(intents)])


    def forward(self, 
                encoder_out: torch.Tensor, 
                training: bool = True,
                gt_intent_sequence = None,
                **kwargs) -> Dict:
        # gt_macro_intent shape: (B,)
        
        scene_context = encoder_out[:, 0]
        batch_size = scene_context.shape[0]

        # --- 步骤 1: 计算顶层宏观意图的Logits ---
        macro_logits = self.top_level_gate(scene_context) # (B, M)

        if training:
            # --- 训练逻辑 ---
            # 我们只训练真值意图对应的那个子系统，以提供最强的监督信号
            gt_intent_sequence = gt_intent_sequence.squeeze()
            if gt_intent_sequence is None:
                raise ValueError("gt_macro_intent must be provided during training.")
            
            final_predictions = torch.zeros(batch_size, self.future_steps, 2, device=scene_context.device)
            
            for i in range(self.intents):
                # 找到宏观意图为 i 的样本
                mask = (gt_intent_sequence == i)
                if mask.any():
                    # 只让第 i 个子系统对这些样本进行预测
                    sub_system_output = self.sub_systems[i](scene_context[mask])
                    # 将结果放回最终的预测张量
                    final_predictions[mask] = sub_system_output

            # 训练时，predictions 只有一个模态，即真值模态的预测结果
            # 这使得回归损失的计算非常直接
            output = {
                "predictions": final_predictions.unsqueeze(1), # (B, 1, T, 2)
                "logits": macro_logits # (B, M) - 用于计算顶层门控损失
            }
        else:
            # --- 推理逻辑 ---
            # 我们选择Top-K个最可能的宏观意图，并让对应的子系统生成轨迹
            macro_probs = F.softmax(macro_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(macro_probs, self.modes, dim=-1)
            
            all_modal_trajs = torch.zeros(
                batch_size, self.modes, self.future_steps, 2, device=scene_context.device
            )

            # 遍历 K 个 top 模态
            for k in range(self.modes):
                # 当前第k个最可能的意图索引 (B,)
                intent_indices = top_k_indices[:, k]
                
                # 遍历所有可能的 M 个意图
                for i in range(self.intents):
                    mask = (intent_indices == i)
                    if mask.any():
                        # 让第 i 个子系统对这些样本进行预测
                        sub_system_output = self.sub_systems[i](scene_context[mask])
                        # 将结果放入第 k 个模态的对应位置
                        all_modal_trajs[mask, k, :, :] = sub_system_output
            
            output = {
                "predictions": all_modal_trajs, # (B, K, T, 2)
                "pi": top_k_probs,           # (B, K)
                "logits": macro_logits          # (B, M) - 完整的logits
            }
            
        return output