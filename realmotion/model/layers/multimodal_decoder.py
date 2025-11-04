import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from realmotion.model.layers.transformer_blocks import Block, DecoderLayer, Inter_cross_self_Block, Inter_map_hist_Block, InterBlock

class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim, future_steps,num_segments=5, return_prob=True) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.return_prob = return_prob
        self.aggregation_layer = nn.Sequential(
            nn.Linear(num_segments * embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

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
        B, T, D = x.shape
        x_flat = x.view(B, -1) 
        aggregated_x = self.aggregation_layer(x_flat) # 输出形状: (B, D), 例如 (48, 128)
        x = self.multimodal_proj(aggregated_x).view(-1, 6, self.embed_dim)
        loc = self.loc(x).view(-1, 6, self.future_steps, 2)
        if self.return_prob:
            pi = self.pi(x).squeeze(-1)
            probs = F.softmax(pi, dim=-1)
        else:
            pi = None

        
        return {"predictions": loc, # (B, top_k, T, 2) -> Top-K的轨迹
                "logits": pi, 
                "pi": probs,
                 "mode": x }
    
class MLPExpert(nn.Module):
    def __init__(self, d_model: int, prediction_horizon: int, drop: int=0.0):
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
                 his_num_segments: int = 5,
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
        self.his_num_segments = his_num_segments
        self.num_segments = future_len // future_steps
        self.top_k = top_k
        self.intent_label = intent_label
        self.total_future_steps = self.num_segments * self.steps_per_segment

        # --- 1. 全局模态概率头 (MLP-based) ---
        # 输入全局场景编码(D)，输出K个模态的logits
        self.mode_prob_network = nn.Sequential(
            nn.Linear(his_num_segments * embed_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, num_modes),
        )

        # --- 2. 分段门控网络 (MLP-based) ---
        # 输入全局场景编码(D)，一次性生成所有K个模态、所有S个分段的专家logits
        self.segment_gating_network = nn.Sequential(
            nn.Linear(his_num_segments *embed_dim, mlp_dim * 2),
            nn.LayerNorm(mlp_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim * 2, num_modes * self.num_segments * self.num_experts),
            
        )

        # --- 3. 专家网络列表 (保持不变) ---
        # 专家依然是简单的MLP，但它们的输入现在需要变一下
        # 我们需要为每个分段生成一个独特的特征
        self.segment_feature_generator = nn.Sequential(
            nn.Linear(his_num_segments *embed_dim, self.num_segments * self.mlp_dim),
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
        scene_context = encoder_out[:, :self.his_num_segments]
        scene_context = scene_context.reshape(batch_size, -1)
        
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
            batch_size, self.num_segments, self.mlp_dim
        )
        # (B, S, D) -> (B*S, D)
        features_flat = segment_features.reshape(-1, self.mlp_dim)

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
                 mlp_drop =0.2,
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
            MLPExpert(self.embed_dim, self.future_steps, mlp_drop)
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
            "pi": mode_logits,              # (B, K)
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


class MoE_QueryDecoder(nn.Module):
    def __init__(self,
                 dim=128,
                 num_layers=3, # 解码器层数
                 num_heads=8,
                 mlp_ratio=2.0,
                 future_len=60,
                 future_steps=60,
                 num_modes=6,
                 use_moe_ffn: bool = True, # 控制是否在FFN中使用MoE
                 num_experts=8,
                 top_k=2,
                 **kwargs):
        super().__init__()
        self.num_modes = num_modes
        self.segment = future_len // future_steps
        self.all_modes = num_modes * self.segment
        self.future_steps = future_steps
        self.total_future_steps = future_len

        # --- 1. K个可学习的多模态查询向量 ---
        self.mode_queries = nn.Parameter(torch.randn(1, self.all_modes, dim))

        # --- 2. 堆叠的解码器层 ---
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                top_k=top_k,
                **kwargs
            ) for _ in range(num_layers)
        ])
        
        # --- 3. 最终的预测头 ---
        # a) 轨迹解码头 (MLP)
        self.loc_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, self.future_steps * 2)
        )
        
        # b) 概率解码头 (MLP)
        self.prob_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, context, training=True, key_padding_mask=None, **kwargs):
        batch_size = context.shape[0]

        # --- 1. 准备初始查询 ---
        # (1, K, D) -> (B, K, D)
        queries = self.mode_queries.expand(batch_size, -1, -1)
        
        # --- 2. 通过多层解码器进行特征提纯 ---
        for layer in self.decoder_layers:
            queries = layer(queries, context, context_key_padding_mask=key_padding_mask)
        # 经过处理后，queries 的形状仍然是 (B, K, D)
        # 现在的 queries 已经融合了场景信息，并且通过自注意力变得多样化
        
        # --- 3. 解码最终结果 ---
        # a) 轨迹
        # (B, K, D) -> (B, K, T*2) -> (B, K, T, 2)
        delta_predictions  = self.loc_head(queries).view(
            batch_size, self.all_modes, self.future_steps, 2
        )
        mode_logits_segmented = self.prob_head(queries).squeeze(-1) # (B, K)

        # --- 概率和轨迹的重组 ---
    

        # 1. 对logits进行求和
        # (B, K) -> (B, K_original, num_segments)
        mode_logits_reshaped = mode_logits_segmented.view(batch_size, self.num_modes, self.segment)
        # (B, K_original, num_segments) -> (B, K_original)
        final_mode_logits = torch.sum(mode_logits_reshaped, dim=2)

        delta_segments = delta_predictions.view(
            batch_size, self.num_modes, self.segment, self.future_steps, 2
        )

        # 初始化一个列表来存储计算出的绝对坐标分段
        absolute_segments = []
        
        # 初始化每个模态的起点，相对于车辆当前位置，起点为(0,0)
        # 形状为 (B, num_modes, 1, 2) 以便进行广播相加
        last_endpoint = torch.zeros(
            batch_size, self.num_modes, 1, 2, 
            device=delta_segments.device, 
            dtype=delta_segments.dtype
        )

        # 循环遍历每一个分段
        for i in range(self.segment):
            # 获取当前分段的位移预测
            # shape: (B, num_modes, T_seg, 2)
            current_delta_segment = delta_segments[:, :, i, :, :]
            
            # 将上一段的终点加到当前段的每一个位移点上，得到当前段的绝对坐标
            current_absolute_segment = current_delta_segment + last_endpoint
            
            # 将计算好的绝对坐标分段存入列表
            absolute_segments.append(current_absolute_segment)
            
            # 更新下一轮循环所需要的“上一段的终点”
            # 取当前绝对坐标分段的最后一个点作为新的终点
            # shape: (B, num_modes, 2) -> unsqueeze -> (B, num_modes, 1, 2)
            last_endpoint = current_absolute_segment[:, :, -1, :].unsqueeze(2)

        # 将列表中所有的绝对坐标分段在时间维度上拼接起来
        # List of (B, K, T_seg, 2) -> (B, K, T_total, 2)
        final_predictions = torch.cat(absolute_segments, dim=2)

        output = {
            "predictions": final_predictions, # (B, K, T, 2)
            "pi": final_mode_logits             # (B, K)
        }
        
        return output

class RegressionSegmentDecoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_modes: int,
                 future_len: int,
                 future_steps: int,
                 num_experts: int,
                 top_k: int,
                 num_heads: int = 8,
                 mlp_ratio = 4.0,
                 qkv_bias = False,
                 drop = 0.2,
                 attn_drop = 0.2,
                 mlp_drop =0.2,
                 drop_path= 0.2,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 query_cross_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_len = future_len
        self.future_steps = future_steps
        self.num_segments = future_len // future_steps
        self.num_experts = num_experts
        self.top_k = top_k

        # --- 1. 初始化模块 ---
        # K个可学习的模态查询，作为K种不同未来的“种子”
        self.mode_queries = nn.Parameter(torch.randn(1, self.num_modes, self.embed_dim))
        
        # 交叉注意力，用于融合历史意图和模态查询
        # self.history_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.history_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.query_init =nn.ModuleList(Inter_cross_self_Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))

        self.query_attn =nn.ModuleList(Inter_cross_self_Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(self.num_segments))
        # self.query_attn =Inter_cross_self_Block(
        #             dim=embed_dim,
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             drop=drop,
        #             attn_drop=attn_drop,
        #             drop_path=drop_path,
        #             act_layer=act_layer,
        #             norm_layer=norm_layer,
        #         )

        # --- 2. 自回归循环模块 ---
        # GRUCell用于在每个时间步更新“思考状态”
        self.gru_cell = nn.GRUCell(input_size=embed_dim, hidden_size=embed_dim)
        # self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads)
        # 门控网络(Planner)，根据思考状态决定调用哪个专家
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, 64), 
            nn.GELU(), 
            nn.Linear(64, num_experts),
        )
        
        # 专家列表(Executor)，每个专家生成一个运动原语
        self.experts = nn.ModuleList([
            MLPExpert(embed_dim, future_steps) for _ in range(num_experts)
        ])
        
        # 将生成的轨迹段重新编码为特征，用于更新GRU状态
        self.segment_embedder = nn.Sequential(
            nn.Linear(future_steps * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # --- 3. 最终输出模块 ---
        # 预测每个模态的概率
        self.prob_head = nn.Sequential(
            nn.Linear(embed_dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )


    def forward(self, history_intent_embeddings,mode,key_padding_mask=None,lane_mask=None):
        """
        Args:
            history_intent_embeddings (torch.Tensor): 编码器输出的历史意图序列。
                                                      形状: (B, T_hist_segments, D)，例如 (32, 5, 128)
        """
        B = history_intent_embeddings.shape[0]

        # --- 步骤 1: 初始化 K 个模态的“思考状态” ---
        queries = self.mode_queries.expand(B, -1, -1) # (B, K, D)
        
        for blk in self.query_init:
            initial_state = blk(src=queries, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
        
        initial_state = self.history_norm(initial_state) # (B, K, D)

        # --- 步骤 2: 预测全局模态概率 ---
        # 基于对历史的初始理解，直接预测每个模态的可能性
        mode_logits = self.prob_head(initial_state).squeeze(-1) # (B, K)
        
        # --- 步骤 3: 准备自回归生成 ---
        # 将所有模态展平到一个批次中，以进行高效的并行计算
        thought_state = initial_state.view(B * self.num_modes, self.embed_dim)
        gru_input = torch.zeros_like(thought_state)
        # gru_input = history_intent_embeddings[:, :7, :].mean(dim=1).unsqueeze(1).expand(-1, self.num_modes, -1)
        gru_input = gru_input.reshape(B * self.num_modes, self.embed_dim)
        future_segments = []
        states = []
        # --- 步骤 4: 自回归循环 ---
        for i in range(self.num_segments):
            # a) 更新思考状态
            thought_state = self.gru_cell(gru_input, thought_state)
            # thought_state = self.attn_layer(gru_input, thought_state, thought_state)[0]
            # b) 规划：决定下一步意图 (路由到专家)
            thought_state_q = thought_state.view(B, self.num_modes, self.embed_dim)
            context_output = self.query_attn[i](src=thought_state_q, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
            rich_thought_state = self.context_norm(thought_state_q + context_output).squeeze(1)
            
            rich_thought_state = rich_thought_state.view(B * self.num_modes, self.embed_dim)
            expert_logits = self.gating_network(rich_thought_state) # (B*K, num_experts)
            states.append(expert_logits)
            segment_output_flat = self._moe_execution(rich_thought_state, expert_logits)
            future_segments.append(segment_output_flat)
            
            # d) 更新：将生成的轨迹段编码，作为下一次GRU的输入
            gru_input = self.segment_embedder(segment_output_flat)

        # --- 步骤 5: 拼接和整理输出 ---
        # 将分段列表堆叠起来
        # List[(B*K, steps*2)] -> (S, B*K, steps*2)
        stacked_segments = torch.stack(future_segments, dim=0)
        
        # 调整形状为最终轨迹格式
        # (S, B*K, steps*2) -> (B*K, S, steps*2) -> (B*K, T, 2)
        trajectories_flat = stacked_segments.permute(1, 0, 2).reshape(
            B * self.num_modes, self.future_len, 2
        )
        
        # (B*K, T, 2) -> (B, K, T, 2)
        final_predictions = trajectories_flat.view(B, self.num_modes, self.future_len, 2)
        states = torch.stack(states, dim=1)
        # states = states.reshape(B, -1, self.embed_dim)
        return {
            "predictions": final_predictions, # (B, K, T, 2)
            "pi": mode_logits,            # (B, K)
            "segment_logits_per_mode": states,
            "states": states
        }
    def _moe_execution(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        高效、向量化的 MoE 执行函数。

        Args:
            state (torch.Tensor): 当前的思考状态, 形状 (N, D)，其中 N = B * K。
            logits (torch.Tensor): 门控网络输出的 logits, 形状 (N, num_experts)。

        Returns:
            torch.Tensor: 加权求和后的轨迹段输出, 形状 (N, future_steps * 2)。
        """
        # 1. 选择 Top-K 专家及其权重
        probs = F.softmax(logits, dim=-1)
        # top_k_probs: (N, top_k), top_k_indices: (N, top_k)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # 归一化权重
        top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)

        # 2. 准备分发 (Dispatch)
        # 创建一个扁平化的索引，指明每个 "token-expert" 对属于哪个专家
        # (N, top_k) -> (N * top_k)
        flat_indices = top_k_indices.view(-1)
        
        # 将输入状态重复 top_k 次，以匹配扁平化的索引
        # (N, D) -> (N * top_k, D)
        repeated_state = state.repeat_interleave(self.top_k, dim=0)

        # 3. 批量执行专家网络
        # 初始化一个空的输出张量
        y_flat = torch.zeros(
            state.shape[0] * self.top_k, 
            self.future_steps * 2, 
            device=state.device,
            dtype=state.dtype
        )
        
        # 这个循环遍历专家数量 (e.g., 9)，而不是批次大小
        for i in range(self.num_experts):
            # 找到所有分配给当前专家 i 的任务
            mask = (flat_indices == i)
            
            if mask.any():
                # 收集所有需要该专家处理的输入状态
                expert_inputs = repeated_state[mask]
                # 一次性地送入专家网络进行计算
                expert_outputs = self.experts[i](expert_inputs)
                # 将结果放回扁平化的输出张量中
                y_flat[mask] = expert_outputs.view(-1, self.future_steps*2)
        
        # 4. 加权聚合 (Combine)
        # (N * top_k, steps * 2) -> (N, top_k, steps * 2)
        y = y_flat.view(state.shape[0], self.top_k, -1)
        
        # 使用 top_k 的权重进行加权求和
        # (N, top_k, 1) * (N, top_k, steps * 2) -> (N, top_k, steps * 2)
        weighted_y = top_k_probs.unsqueeze(-1) * y
        
        # (N, top_k, steps * 2) -> (N, steps * 2)
        final_output = torch.sum(weighted_y, dim=1)
        
        return final_output
    


class Regress_refine(RegressionSegmentDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.refine_model = nn.ModuleList(Inter_cross_self_Block(
                    dim=kwargs.get("embed_dim"),
                    num_heads=kwargs.get("num_heads"),
                    mlp_ratio=kwargs.get("mlp_ratio"),
                    qkv_bias=kwargs.get("qkv_bias"),
                    drop=0.2,
                    attn_drop=0.2,
                    drop_path=0.2,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                ) for i in range(kwargs.get("query_cross_layers")))
        self.refine_embed  = nn.Sequential(
            nn.Linear(kwargs.get("future_len") * 2, kwargs.get("embed_dim")),
            nn.ReLU(),
            nn.Linear(kwargs.get("embed_dim"), kwargs.get("embed_dim"))
        )
        self.refine_head = nn.Sequential(
            nn.Linear(kwargs.get("embed_dim"), kwargs.get("future_len") * 2),
            nn.ReLU(),
            nn.Linear(kwargs.get("future_len") * 2, kwargs.get("future_len") * 2)
        )

    def forward(self, history_intent_embeddings,mode,key_padding_mask=None,lane_mask=None):
        
        out = super().forward(history_intent_embeddings,mode,key_padding_mask,lane_mask)
        # (B*K, T, 2) -> (B, K, T, 2)

        final_predictions = out['predictions']
        B = final_predictions.shape[0]
        refine_embed = self.refine_embed(final_predictions.reshape(B, self.num_modes, self.future_len*2))
        for blk in self.refine_model:
            refine_embed = blk(src=refine_embed,src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)

        refine = self.refine_head(refine_embed).view(B, self.num_modes, self.future_len, 2)
        refine = refine + final_predictions
        return {
            "predictions": refine, # (B, K, T, 2)
            "propose": final_predictions, # (B, K, T, 2)
            "logits": out['logits'],            # (B, K)
            "probs": out['probs']
        }
    
class Regress_refine_v2(Regress_refine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intent_queries = nn.Parameter(torch.randn(1, self.num_modes, self.num_segments, self.embed_dim))


    def forward(self, history_intent_embeddings,mode,key_padding_mask=None,lane_mask=None):
        B = history_intent_embeddings.shape[0]

        # --- 阶段一: 并行意图规划 ---
        # (1, K, S, D) -> (B, K, S, D)
        queries = self.intent_queries.expand(B, -1, -1, -1)
        # 为高效计算，将 B 和 K 合并: (B*K, S, D)
        queries_flat = queries.reshape(B , self.num_segments* self.num_modes, self.embed_dim)
        
        # 填充意图占位符
        planned_intents = queries_flat
        for blk in self.query_init:
            planned_intents = blk(src=planned_intents, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
        
        # planned_intents 的形状是 (B*K, S, D)

        # --- 步骤 1.5: 预测全局模态概率 ---
        # 基于规划好的意图序列的平均池化来预测概率
        # (B*K, S, D) -> (B*K, D) -> (B, K, D)
        planned_intents = planned_intents.reshape(B , self.num_segments, self.num_modes, self.embed_dim)
        global_intent = planned_intents.mean(dim=1).view(B, self.num_modes, self.embed_dim)
        mode_logits = self.prob_head(global_intent).squeeze(-1) # (B, K)

        # --- 阶段二: 非自回归初步轨迹生成 ---
        # 将所有意图一次性送入MoE，生成初步轨迹
        # (B*K, S, D) -> (B*K*S, D)
        all_intents_flat = planned_intents.reshape(-1, self.embed_dim)
        expert_logits = self.gating_network(all_intents_flat)
        
        # (B*K*S, steps*2)
        initial_segments_flat = self._moe_execution(all_intents_flat, expert_logits)
        
        # (B*K*S, steps*2) -> (B*K, S, steps*2) -> (B*K, T, 2)
        trajectories_flat = initial_segments_flat.view(
            B * self.num_modes, self.num_segments, self.future_steps * 2
        ).reshape(B * self.num_modes, self.future_len, 2)
        
        # (B*K, T, 2) -> (B, K, T, 2)
        final_predictions = trajectories_flat.view(B, self.num_modes, self.future_len, 2)
        refine_embed = self.refine_embed(final_predictions.reshape(B, self.num_modes, self.future_len*2))
        for blk in self.refine_model:
            refine_embed = blk(src=refine_embed,src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)

        refine = self.refine_head(refine_embed).view(B, self.num_modes, self.future_len, 2)
        refine = refine + final_predictions
        return {
            "predictions": refine, # (B, K, T, 2)
            "propose": final_predictions, # (B, K, T, 2)
            "logits": mode_logits,            # (B, K)
            "probs": F.softmax(mode_logits, dim=-1)
        }
class RefinementDecoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_modes: int,
                 future_len: int,
                 future_steps: int,
                 num_experts: int,
                 top_k: int,
                 num_heads: int = 8,
                 mlp_ratio = 4.0,
                 qkv_bias = False,
                 drop = 0.2,
                 attn_drop = 0.2,
                 mlp_drop =0.2,
                 drop_path= 0.2,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 query_cross_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_len = future_len
        self.future_steps = future_steps
        self.num_segments = future_len // future_steps
        self.num_experts = num_experts
        self.top_k = top_k

        # --- 1. 意图规划器 (Planner) 模块 ---
        # 可学习的查询，为每个模态的每个未来片段都创建一个"意图占位符"
        # 形状: (1, K, S, D)  K=num_modes, S=num_segments
        self.intent_queries = nn.Parameter(torch.randn(1, self.num_modes, self.num_segments, self.embed_dim))
        
        # 交叉注意力块，用于填充意图占位符
        self.intent_planner = nn.ModuleList(Inter_map_hist_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                    attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
                ) for _ in range(query_cross_layers))

        # --- 2. 初步执行器 & 轨迹优化器 (Executor & Refiner) 模块 ---
        self.context_norm = nn.LayerNorm(embed_dim)
        # 交叉注意力块，用于在优化阶段查询地图
        self.map_cross_attn = nn.ModuleList(Inter_cross_self_Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                    attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
                ) for _ in range(self.num_segments))

        # GRUCell用于在优化阶段更新状态
        self.gru_cell = nn.GRUCell(input_size=embed_dim, hidden_size=embed_dim)
        
        # 门控网络(Planner)和专家列表(Executor)保持不变
        self.gating_network = nn.Linear(embed_dim, num_experts)
        self.experts = nn.ModuleList([
            MLPExpert(embed_dim, future_steps) for _ in range(num_experts)
        ])
        
        # 轨迹段编码器也保持不变
        self.segment_embedder = nn.Sequential(
            nn.Linear(future_steps * 2, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

        # --- 3. 概率预测模块 ---
        # 概率头现在基于所有意图的平均值来预测，以获得更稳健的全局评估
        self.prob_head = nn.Linear(embed_dim, 1)

    def forward(self, actor_feat, lane_feat, actor_mask=None, lane_mask=None,his_seg=1):
        B = actor_feat.shape[0]

        # --- 阶段一: 并行意图规划 ---
        # (1, K, S, D) -> (B, K, S, D)
        queries = self.intent_queries.expand(B, -1, -1, -1)
        # 为高效计算，将 B 和 K 合并: (B*K, S, D)
        queries_flat = queries.reshape(B * self.num_modes, self.num_segments, self.embed_dim)
        
        # 填充意图占位符
        planned_intents = queries_flat
        for blk in self.intent_planner:
            planned_intents = blk(src=planned_intents, hist_feat=actor_feat.repeat_interleave(self.num_modes, dim=0), 
                                  map_feat=lane_feat.repeat_interleave(self.num_modes, dim=0),
                                  hist_mask=actor_mask.repeat_interleave(self.num_modes, dim=0) if actor_mask is not None else None, 
                                  map_mask=lane_mask.repeat_interleave(self.num_modes, dim=0) if lane_mask is not None else None)
        # planned_intents 的形状是 (B*K, S, D)

        # --- 步骤 1.5: 预测全局模态概率 ---
        # 基于规划好的意图序列的平均池化来预测概率
        # (B*K, S, D) -> (B*K, D) -> (B, K, D)
        global_intent = planned_intents.mean(dim=1).view(B, self.num_modes, self.embed_dim)
        mode_logits = self.prob_head(global_intent).squeeze(-1) # (B, K)

        # --- 阶段二: 非自回归初步轨迹生成 ---
        # 将所有意图一次性送入MoE，生成初步轨迹
        # (B*K, S, D) -> (B*K*S, D)
        all_intents_flat = planned_intents.reshape(-1, self.embed_dim)
        expert_logits = self.gating_network(all_intents_flat)
        
        # (B*K*S, steps*2)
        initial_segments_flat = self._moe_execution(all_intents_flat, expert_logits)
        
        # (B*K*S, steps*2) -> (B*K, S, steps*2) -> (B*K, T, 2)
        initial_trajectory = initial_segments_flat.view(
            B * self.num_modes, self.num_segments, self.future_steps * 2
        ).reshape(B * self.num_modes, self.future_len, 2)

        # --- 阶段三: 自回归轨迹优化/微调 ---
        # 初始化GRU的隐藏状态，可以使用第一个意图段
        thought_state = planned_intents[:, 0, :] # (B*K, D)
        
        refined_segments = []
        current_trajectory_segments = initial_segments_flat.view(B * self.num_modes, self.num_segments, -1)

        for i in range(self.num_segments):
            # a) 准备输入: 结合初步轨迹和预定意图
            current_segment_embed = self.segment_embedder(current_trajectory_segments[:, i, :])
            # 这里的输入是初步轨迹的编码 + 预先规划好的意图
            gru_input = current_segment_embed + planned_intents[:, i, :]
            
            # b) 更新状态并查询地图
            thought_state = self.gru_cell(gru_input, thought_state)
            thought_state_q = thought_state.view(B, self.num_modes, self.embed_dim)

            context_output = self.map_cross_attn[i](src=thought_state_q, src_kv=lane_feat, key_padding_mask=lane_mask)
            rich_thought_state = self.context_norm(thought_state_q + context_output) # (B*K, D)
            rich_thought_state = rich_thought_state.view(-1, self.embed_dim)
            # c) 预测修正量 (Residual)
            expert_logits_refine = self.gating_network(rich_thought_state)
            # MoE输出的是对当前段的修正
            residual_segment = self._moe_execution(rich_thought_state, expert_logits_refine)
            
            # d) 更新轨迹段
            refined_segment = current_trajectory_segments[:, i, :] + residual_segment
            refined_segments.append(refined_segment)
        
        # --- 步骤 5: 拼接和整理最终输出 ---
        # List[(B*K, steps*2)] -> (S, B*K, steps*2)
        stacked_segments = torch.stack(refined_segments, dim=0)
        
        # (S, B*K, steps*2) -> (B*K, S, steps*2) -> (B*K, T, 2)
        trajectories_flat = stacked_segments.permute(1, 0, 2).reshape(
            B * self.num_modes, self.future_len, 2
        )
        
        # (B*K, T, 2) -> (B, K, T, 2)
        final_predictions = trajectories_flat.view(B, self.num_modes, self.future_len, 2)

        return {
            "predictions": final_predictions,        # (B, K, T, 2) - 优化后的轨迹
            "initial_predictions": initial_trajectory.view(B, self.num_modes, self.future_len, 2), # (可选)初步轨迹
            "logits": mode_logits,                   # (B, K)
            "probs": F.softmax(mode_logits, dim=-1)
        }

    # _moe_execution 函数保持不变，因为它已经很高效了
    def _moe_execution(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # ... (和您原来的代码完全一样) ...
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)
        flat_indices = top_k_indices.view(-1)
        repeated_state = state.repeat_interleave(self.top_k, dim=0)
        y_flat = torch.zeros(
            state.shape[0] * self.top_k, 
            self.future_steps * 2, 
            device=state.device,
            dtype=state.dtype
        )
        for i in range(self.num_experts):
            mask = (flat_indices == i)
            if mask.any():
                expert_inputs = repeated_state[mask]
                expert_outputs = self.experts[i](expert_inputs)
                y_flat[mask] = expert_outputs.view(-1, self.future_steps*2)
        y = y_flat.view(state.shape[0], self.top_k, -1)
        weighted_y = top_k_probs.unsqueeze(-1) * y
        final_output = torch.sum(weighted_y, dim=1)
        return final_output
    
class QuerrySegmentDecoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_modes: int,
                 future_len: int,
                 future_steps: int,
                 num_experts: int,
                 top_k: int,
                 num_heads: int = 8,
                 mlp_ratio = 4.0,
                 qkv_bias = False,
                 drop = 0.2,
                 attn_drop = 0.2,
                 mlp_drop =0.2,
                 drop_path= 0.2,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 query_cross_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_len = future_len
        self.future_steps = future_steps
        self.num_segments = future_len // future_steps
        self.num_experts = num_experts
        self.top_k = top_k

        # --- 1. 初始化模块 ---
        # K个可学习的模态查询，作为K种不同未来的“种子”
        self.mode_queries = nn.Parameter(torch.randn(1, self.num_modes, self.embed_dim))
        self.intent_queries = nn.Parameter(torch.randn(1, self.num_segments, self.embed_dim))
        
        # 交叉注意力，用于融合历史意图和模态查询
        # self.history_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.history_norm = nn.LayerNorm(embed_dim)
        self.intent_norm = nn.LayerNorm(embed_dim)
        self.query_mode =nn.ModuleList(InterBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))
        
        self.query_mode_self =nn.ModuleList(Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))

        self.query_intent =nn.ModuleList(InterBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))
        
        self.query_intent_self =nn.ModuleList(Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))

        self.query_intent_mode_dense =nn.ModuleList(InterBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))
        self.query_intent_mode_self =nn.ModuleList(Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))
        self.query_intent_self =nn.ModuleList(Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(query_cross_layers))

        # self.gating_network = nn.Linear(embed_dim, num_experts)
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, 64), 
            nn.GELU(), 
            nn.Linear(64, num_experts),
        )
        
        # 专家列表(Executor)，每个专家生成一个运动原语
        self.experts = nn.ModuleList([
            MLPExpert(embed_dim, future_steps) for _ in range(num_experts)
        ])
        

        # --- 3. 最终输出模块 ---
        # 预测每个模态的概率
        self.prob_head = nn.Sequential(
            nn.Linear(embed_dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )


    def forward(self, history_intent_embeddings,mode,key_padding_mask=None,lane_mask=None):
        """
        Args:
            history_intent_embeddings (torch.Tensor): 编码器输出的历史意图序列。
                                                      形状: (B, T_hist_segments, D)，例如 (32, 5, 128)
        """
        B = history_intent_embeddings.shape[0]

        # --- 步骤 1: 初始化 K 个模态的“思考状态” ---
        mode = self.mode_queries.expand(B, -1, -1) # (B, K, D)
        
        for blk in self.query_mode:
            mode = blk(src=mode, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
        for blk in self.query_mode_self:
            mode = blk(src=mode)

        mode = self.history_norm(mode) # (B, K, D)

        # --- 步骤 2: 预测全局模态概率 ---
        # 基于对历史的初始理解，直接预测每个模态的可能性
        mode_logits = self.prob_head(mode).squeeze(-1) # (B, K)
        
        intent = self.intent_queries.expand(B, -1, -1)

        for blk in self.query_intent:
            intent = blk(src=intent, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
        for blk in self.query_intent_self:
            intent = blk(src=intent)
        
        intent = self.intent_norm(intent)

        mode_intent = intent[:,None, :, :] + mode[:,:, None, :]
        B, M, I, D = mode_intent.shape
        mode_intent = mode_intent.reshape(B, M * I, D)
        for blk in self.query_intent_mode_dense:
            mode_intent = blk(src=mode_intent, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
        mode_intent = mode_intent.reshape(B, M, I, D)

        mode_intent = mode_intent.permute(0,2,1,3).reshape(B*I, M, D)
        for blk in self.query_intent_mode_self:
            mode_intent = blk(src=mode_intent)
        mode_intent = mode_intent.reshape(B, I, M, D).permute(0,2,1,3)

        mode_intent = mode_intent.reshape(B*M, I, D)
        for blk in self.query_intent_self:
            mode_intent = blk(src=mode_intent)
        mode_intent = mode_intent.reshape(B, M, I, D)

        all_intents_flat = mode_intent.reshape(-1, self.embed_dim)
        expert_logits = self.gating_network(all_intents_flat)
        
        # (B*K*S, steps*2)
        initial_segments_flat = self._moe_execution(all_intents_flat, expert_logits)
        

        final_predictions = initial_segments_flat.reshape(B, self.num_modes, self.num_segments, self.future_steps, 2)
        final_predictions = final_predictions.reshape(B, self.num_modes, -1, 2)

        return {
            "predictions": final_predictions, # (B, K, T, 2)
            "pi": mode_logits,            # (B, K)
            "segment_logits_per_mode": expert_logits,
            "states": mode
        }
    def _moe_execution(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        高效、向量化的 MoE 执行函数。

        Args:
            state (torch.Tensor): 当前的思考状态, 形状 (N, D)，其中 N = B * K。
            logits (torch.Tensor): 门控网络输出的 logits, 形状 (N, num_experts)。

        Returns:
            torch.Tensor: 加权求和后的轨迹段输出, 形状 (N, future_steps * 2)。
        """
        # 1. 选择 Top-K 专家及其权重
        probs = F.softmax(logits, dim=-1)
        # top_k_probs: (N, top_k), top_k_indices: (N, top_k)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # 归一化权重
        top_k_probs = F.normalize(top_k_probs, p=1, dim=-1)

        # 2. 准备分发 (Dispatch)
        # 创建一个扁平化的索引，指明每个 "token-expert" 对属于哪个专家
        # (N, top_k) -> (N * top_k)
        flat_indices = top_k_indices.view(-1)
        
        # 将输入状态重复 top_k 次，以匹配扁平化的索引
        # (N, D) -> (N * top_k, D)
        repeated_state = state.repeat_interleave(self.top_k, dim=0)

        # 3. 批量执行专家网络
        # 初始化一个空的输出张量
        y_flat = torch.zeros(
            state.shape[0] * self.top_k, 
            self.future_steps * 2, 
            device=state.device,
            dtype=state.dtype
        )
        
        # 这个循环遍历专家数量 (e.g., 9)，而不是批次大小
        for i in range(self.num_experts):
            # 找到所有分配给当前专家 i 的任务
            mask = (flat_indices == i)
            
            if mask.any():
                # 收集所有需要该专家处理的输入状态
                expert_inputs = repeated_state[mask]
                # 一次性地送入专家网络进行计算
                expert_outputs = self.experts[i](expert_inputs)
                # 将结果放回扁平化的输出张量中
                y_flat[mask] = expert_outputs.view(-1, self.future_steps*2)
        
        # 4. 加权聚合 (Combine)
        # (N * top_k, steps * 2) -> (N, top_k, steps * 2)
        y = y_flat.view(state.shape[0], self.top_k, -1)
        
        # 使用 top_k 的权重进行加权求和
        # (N, top_k, 1) * (N, top_k, steps * 2) -> (N, top_k, steps * 2)
        weighted_y = top_k_probs.unsqueeze(-1) * y
        
        # (N, top_k, steps * 2) -> (N, steps * 2)
        final_output = torch.sum(weighted_y, dim=1)
        
        return final_output