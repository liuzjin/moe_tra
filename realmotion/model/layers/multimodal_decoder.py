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

        # --- 1. 可学习的意图查询向量 ---
        # 这是新架构的核心。每个向量将学会代表一种特定的驾驶意图。
        # (num_experts, dim)
        self.intention_queries = nn.Parameter(torch.randn(self.num_experts, self.embed_dim))

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
            self.query_self_blocks = nn.ModuleList( Block(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path,
                ) for i in range(query_self_layers))
        
            
        # --- 3. Logit生成器 ---
        # 从交叉注意力的输出（即每个专家的专业化特征）中，计算出该专家的得分。
        
        # self.logit_head = nn.Linear(self.embed_dim, 1)
        self.logit_head = nn.Sequential(
                            nn.Linear(self.embed_dim, self.embed_dim // 2),
                            nn.LayerNorm(self.embed_dim // 2),
                            nn.GELU(),
                            nn.Linear(self.embed_dim // 2, 1)
                                    )
        # --- 4. 专家网络列表 (与之前相同) ---
        self.experts = nn.ModuleList([
            MLPExpert(self.embed_dim, self.future_steps)
            for _ in range(self.num_experts)
        ])

    def forward(self, context, training= True, key_padding_mask=None):
        batch_size = context.shape[0]

        # --- 核心步骤 1: 使用意图查询进行交叉注意力 ---
        
        # 将意图查询从 (num_experts, D) 扩展到 (B, num_experts, D) 以匹配批次大小
        expert_features = self.intention_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Q: 意图查询, K/V: 上下文
        # expert_features 的形状为 (B, num_experts, D)
        # 它包含了为每个专家量身定制的特征表示
        for blk in self.query_cross_blocks:
            expert_features = blk(
                src=expert_features, 
                src_kv=context, 
                key_padding_mask=key_padding_mask,
            )
        if self.query_self_atten:
            for blk in self.query_self_blocks:
                expert_features = blk(src=expert_features)

        # --- 核心步骤 2: 从专业化特征计算门控得分(logits) ---
        
        # (B, num_experts, D) -> (B, num_experts, 1) -> (B, num_experts)
        logits = self.logit_head(expert_features).squeeze(-1)
        probs = logits.softmax(dim=-1)
        if not self.intent_label:
            aux_loss = torch.tensor(0.0, device=context.device)
        # --- 核心步骤 3: 使用专业化特征进行轨迹预测 ---
        predictions = []
        for i in range(self.num_experts):
            # 提取第i个专家的特征 (B, D)
            current_expert_feature = expert_features[:, i, :]
            pred = self.experts[i](current_expert_feature)
            predictions.append(pred)
        
        # (B, num_experts, future_steps, 2)
        all_predictions = torch.stack(predictions, dim=1)

        if training:
            if not self.intent_label:
                mean_probs = torch.mean(F.softmax(logits, dim=-1), dim=0)
                # 计算每个专家被分配到的任务比例
                # 这里我们使用 logits 的 softmax 作为 "soft" 分配
                fraction_of_examples = torch.mean(F.softmax(logits, dim=-1), dim=0)
                
                # 负载均衡损失 = sum(每个专家被选中的概率 * 每个专家被分配的任务比例)
                # 这个损失会惩罚门控网络总是选择少数几个专家的情况
                aux_loss = self.num_experts * torch.sum(mean_probs * fraction_of_examples)

            output = {
                "predictions": all_predictions,  # (B, num_experts, T, 2)
                "logits": logits,                  # (B, num_experts)
                "mode":expert_features
            }
            if not self.intent_label:
                output["aux_loss"] = aux_loss
            return output
        else:
            # 推理时，只计算 Top-K 专家以提高效率
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

            indices_for_gather = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.future_steps, 2)
            top_k_predictions = torch.gather(all_predictions, 1, indices_for_gather)
            return {
                "predictions": top_k_predictions, # (B, top_k, T, 2) -> Top-K的轨迹
                "probs": top_k_probs,              # (B, num_experts) -> 【新增】返回完整的概率分布，方便分析
                "top_k_indices": top_k_indices,    # (B, top_k) -> Top-K的专家索引
                "mode":expert_features
            }