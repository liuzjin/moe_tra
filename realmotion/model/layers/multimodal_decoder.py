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
            pi = pi.softmax(dim=-1)
        else:
            pi = None

        
        return {"predictions": loc, # (B, top_k, T, 2) -> Top-K的轨迹
                "probs": pi, 
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
                 top_k: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_experts = num_experts
        self.top_k = top_k
        
        # --- 1. 多模态特征生成器 ---
        # 这个模块负责将输入的单一编码向量，扩展为每个专家一个的特征向量。
        # 这样做的好处是，每个专家可以从一个略有不同的“视角”开始。
        # 如果你希望所有专家输入完全相同，可以去掉这个模块。
        self.multimodal_feature_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * num_experts),
            nn.LayerNorm(embed_dim * num_experts) # LayerNorm在这里效果通常很好
        )
        
        # --- 2. 门控网络 (Gating Network) ---
        # 一个独立的、轻量级的网络，只负责预测每个专家的得分(logits)。
        # 它的输入是原始的编码向量，以做出全局判断。
        self.gating_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_experts)
        )

        # --- 3. 专家网络列表 (Expert Networks) ---
        self.experts = nn.ModuleList([
            MLPExpert(embed_dim, future_steps)
            for _ in range(self.num_experts)
        ])

    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练模式下的前向传播：计算所有专家。
        """
        # x shape: (Batch, 1, embed_dim)
        x_squeezed = x.squeeze(1) # -> (Batch, embed_dim)
        
        # 1. 生成多模态特征
        # (B, D) -> (B, num_experts * D) -> (B, num_experts, D)
        expert_features = self.multimodal_feature_generator(x_squeezed).view(-1, self.num_experts, self.embed_dim)
        
        # 2. 计算门控得分
        logits = self.gating_network(x_squeezed) # (B, num_experts)
        logits = logits.softmax(dim=-1)
        
        # 3. 并行计算所有专家的轨迹
        predictions = []
        for i in range(self.num_experts):
            # 为第i个专家提取其对应的特征
            current_expert_feature = expert_features[:, i, :] # (B, D)
            pred = self.experts[i](current_expert_feature)
            predictions.append(pred)
            
        # (B, num_experts, future_steps, 2)
        all_predictions = torch.stack(predictions, dim=1)
        
        return all_predictions, logits

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """
        推理模式下的前向传播：只计算Top-K专家。
        """
        x_squeezed = x.squeeze(1) # -> (Batch, embed_dim)
        
        # 1. 计算门控得分
        logits = self.gating_network(x_squeezed) # (B, num_experts)
        logits = logits.softmax(dim=-1)
        
        # 2. 选择 Top-K 的专家
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1) # (B, k), (B, k)
        
        # 3. 只为被选中的专家生成特征并计算轨迹
        # 这是一个路由操作，我们可以通过循环实现，或者用更高效的gather
        batch_size = x.shape[0]
        
        # 创建一个占位符张量来存储结果
        # 我们只返回 top_k 个预测，而不是所有专家的位置都占上
        top_k_predictions = torch.zeros(batch_size, self.top_k, self.future_steps, 2, device=x.device)
        
        # 生成多模态特征 (我们只生成一次，然后根据需要索引)
        expert_features = self.multimodal_feature_generator(x_squeezed).view(-1, self.num_experts, self.embed_dim)

        for i in range(batch_size):
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j]
                feature = expert_features[i, expert_idx, :]
                top_k_predictions[i, j] = self.experts[expert_idx](feature)
                
        return top_k_predictions, top_k_logits, top_k_indices

    def forward(self, encoder, training, key_padding_mask=None) -> Dict:
        x = encoder[:, 0]
        if training:
            # 训练时，计算所有专家
            predictions, logits = self.forward_train(x)
            return {
                "predictions": predictions,  # (B, num_experts, T, 2)
                "logits": logits,            # (B, num_experts)
            }
        else:
            # 推理时，只计算Top-K
            top_k_preds, top_k_logits, top_k_indices = self.forward_inference(x)
            # [32, 2, 10, 2] [32, 2]
            # 为了与训练时的输出格式保持一致，我们可以创建一个稀疏的完整预测张量
            # (这在需要提交所有模态结果的场景下很有用)
            sparse_predictions = torch.zeros(x.shape[0], self.num_experts, self.future_steps, 2, device=x.device)
            # 使用 scatter_ 将 top-k 的结果填充到正确的位置
            # top_k_indices需要扩展维度以匹配
            indices_for_scatter = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.future_steps, 2)
            sparse_predictions.scatter_(1, indices_for_scatter, top_k_preds)
            
            return {
                "predictions": top_k_preds,       # (B, num_experts, T, 2) (稀疏的)
                "logits": top_k_logits,                        # (B, num_experts)
                "top_k_indices": top_k_indices           # (B, top_k)
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
                 query_self_layers=1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 future_steps=10, 
                 num_experts=9, 
                 top_k=2,
                 num_heads=8): # 增加了注意力头的数量作为参数
        super().__init__()
        self.embed_dim = dim
        self.future_steps = future_steps
        self.num_experts = num_experts
        self.top_k = top_k

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
        self.query_self_blocks = nn.ModuleList( Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path,
            ) for i in range(query_self_layers))
        
            
        # --- 3. Logit生成器 ---
        # 从交叉注意力的输出（即每个专家的专业化特征）中，计算出该专家的得分。
        self.logit_head = nn.Linear(self.embed_dim, 1)

        # --- 4. 专家网络列表 (与之前相同) ---
        self.experts = nn.ModuleList([
            MLPExpert(self.embed_dim, self.future_steps)
            for _ in range(self.num_experts)
        ])

    def forward(self, context, training= True, key_padding_mask=None):
        """
        统一的前向传播函数。
        
        Args:
            context (torch.Tensor): 编码器的输出，形状为 (Batch, SeqLen, embed_dim)。
                                    这是地图和历史轨迹融合后的上下文信息。
            training (bool): 是否为训练模式。
        """
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
        for blk in self.query_self_blocks:
            expert_features = blk(src=expert_features)

        # --- 核心步骤 2: 从专业化特征计算门控得分(logits) ---
        
        # (B, num_experts, D) -> (B, num_experts, 1) -> (B, num_experts)
        logits = self.logit_head(expert_features).squeeze(-1)
        probs = logits.softmax(dim=-1)

        # --- 核心步骤 3: 使用专业化特征进行轨迹预测 ---

        if training:
            # 训练时，我们需要计算所有专家的输出以进行 "Winner-Takes-All" 损失计算
            predictions = []
            for i in range(self.num_experts):
                # 提取第i个专家的特征 (B, D)
                current_expert_feature = expert_features[:, i, :]
                pred = self.experts[i](current_expert_feature)
                predictions.append(pred)
            
            # (B, num_experts, future_steps, 2)
            all_predictions = torch.stack(predictions, dim=1)

            return {
                "predictions": all_predictions,  # (B, num_experts, T, 2)
                "probs": probs                  # (B, num_experts)
            }
        else:
            # 推理时，只计算 Top-K 专家以提高效率
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

            # 为了高效地只计算 top-k 专家的轨迹，我们首先需要收集对应的特征
            # 使用 gather 从 expert_features 中精确地挑选出 top-k 特征
            # top_k_indices (B, K) -> (B, K, D) for gathering
            indices_for_gather = top_k_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            top_k_features = torch.gather(expert_features, 1, indices_for_gather)

            # 现在我们有了一个紧凑的张量 (B, K, D) 只包含需要计算的特征
            top_k_predictions = torch.zeros(batch_size, self.top_k, self.future_steps, 2, device=context.device)

            # 循环遍历，将每个特征路由到正确的专家
            # 这里的循环比之前的版本更高效，因为特征提取已经全部在GPU上并行完成
            for i in range(batch_size):
                for j in range(self.top_k):
                    expert_idx = top_k_indices[i, j].item()
                    feature = top_k_features[i, j]
                    top_k_predictions[i, j] = self.experts[expert_idx](feature)
            
            return {
                "predictions": top_k_predictions, # (B, top_k, T, 2) -> Top-K的轨迹
                "probs": probs,              # (B, num_experts) -> 【新增】返回完整的概率分布，方便分析
                "top_k_probs": top_k_probs,       # (B, top_k) -> Top-K的概率值
                "top_k_indices": top_k_indices    # (B, top_k) -> Top-K的专家索引
            }