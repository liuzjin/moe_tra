from typing import List

import torch
import torch.nn as nn


from .layers.agent_embedding import AgentEmbeddingLayer, HistoryCompressor
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import  MoE_QueryDecoder, MultimodalDecoder, QueryBasedMoeDecoder, RegressionSegmentDecoder, SimpleSegmentalMoeDecoder,HierarchicalGatingDecoder
from .layers.mtr_decoder import TransformerDecoder
from .layers.transformer_blocks import Block, InteractionModule


class MoeMotion(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        mlp_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        mlp_drop=0.2,
        history_len=50,
        future_steps=60,
        future_len=60,
        moe=True,
        moe_type="cross",
        query_cross_layers=2,
        query_self_atten=True,
        query_self_layers=1,
        num_experts=9,
        top_k=2,
        intent_label=True,
        attn_moe_mlp=False,
        modes=6
    ) -> None:
        super().__init__()
        
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.num_segments = 3
        self.segment_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.num_segments, embed_dim)
        )
        self.hist_compress = HistoryCompressor(embed_dim, 30, 10)
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)
        self.intent_label = intent_label

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.moe = moe
        if moe:
            if moe_type == "cross":
                self.decoder = QueryBasedMoeDecoder(
                    dim=embed_dim, 
                    mlp_ratio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    mlp_drop=mlp_drop,
                    query_cross_layers=query_cross_layers,
                    query_self_atten=query_self_atten,
                    query_self_layers=query_self_layers,
                    future_steps=future_steps, 
                    future_len=future_len,
                    num_experts=num_experts, 
                    top_k=top_k,
                    intent_label=intent_label)
            elif moe_type == "mlp":
                self.decoder = SimpleSegmentalMoeDecoder(embed_dim,mlp_dim, future_steps,
                    his_num_segments=self.num_segments, num_experts=num_experts, top_k=top_k, 
                    intent_label=intent_label, drop=mlp_drop)
            elif moe_type == "hire_moe":
                self.decoder = HierarchicalGatingDecoder(embed_dim, future_steps, intents=num_experts, top_k=top_k)
            elif moe_type == "cross_moe_mlp":
                self.decoder = MoE_QueryDecoder(
                    dim=embed_dim,
                    num_layers=query_cross_layers, # 解码器层数
                    mlp_ratio=mlp_ratio,
                    future_len=future_len,
                    future_steps=future_steps,
                    num_modes=modes,
                    use_moe_ffn=attn_moe_mlp, # 控制是否在FFN中使用MoE
                    num_experts=num_experts,
                    top_k=top_k,
                    
                )
            elif moe_type == "intent_regre":
                self.decoder = RegressionSegmentDecoder(
                    embed_dim=embed_dim,
                    num_modes=modes,
                    future_len=future_len,
                    future_steps=future_steps,
                    num_experts=num_experts,
                    top_k=top_k,
                    num_heads= 8,
                    mlp_ratio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    mlp_drop=mlp_drop,
                    query_cross_layers=query_cross_layers,
                    )
        else:
            self.decoder = MultimodalDecoder(embed_dim, future_steps,self.num_segments)
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim*self.num_segments, 256), nn.ReLU(), nn.Linear(256, future_len * 2)
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
        state_dict = {
            k[len('net.') :]: v for k, v in ckpt.items() if k.startswith('net.')
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data, mode):
        hist_valid_mask = data['x_valid_mask']
        hist_key_valid_mask = data['x_key_valid_mask']
        hist_feat = torch.cat(
            [
                data['x_positions_diff'],
                data['x_velocity_diff'][..., None],
                hist_valid_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[hist_feat_key_valid].permute(0, 2, 1).contiguous()
        )
        actor_feat = self.hist_compress(actor_feat)
        segment = actor_feat.shape[-2]

        actor_feat_tmp = torch.zeros(
            B * N, segment, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, segment, actor_feat.shape[-1])
        actor_feat = actor_feat + self.segment_pos_embed
        actor_feat = actor_feat.reshape(B, -1, actor_feat.shape[-1])

        lane_valid_mask = data['lane_valid_mask']
        lane_normalized = data['lane_positions'] - data['lane_centers'].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers_agents = data['x_centers'].unsqueeze(2).repeat(1, 1, segment, 1)  # [B, N, segment, 2]
        x_angles_agents = data['x_angles'][:, :, -1].unsqueeze(2).repeat(1, 1, segment)  # [B, N, segment, 1]
        x_centers = torch.cat([x_centers_agents.reshape(B, N*segment, -1), data['lane_centers']], dim=1)
        angles_agents = x_angles_agents.reshape(B, N*segment)
        angles = torch.cat([angles_agents, data['lane_angles']], dim=1)

        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_indices = data['x_attr'][..., 2].long().unsqueeze(-1).repeat(1, 1, segment).view(B, N*segment)
        actor_type_embed = self.actor_type_embed[actor_type_indices]


        # x_centers = torch.cat([data['x_centers'], data['lane_centers']], dim=1)
        # angles = torch.cat([data['x_angles'][:, :, -1], data['lane_angles']], dim=1)
        # x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        # pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        # pos_embed = self.pos_embed(pos_feat)
        # actor_type_embed = self.actor_type_embed[data['x_attr'][..., 2].long()]
       
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)

        key_valid_mask = torch.cat(
            [data['x_key_valid_mask'].unsqueeze(-1).repeat(1, 1, segment).reshape(B, -1), data['lane_key_valid_mask']], dim=1
        )
        x_encoder = x_encoder + pos_embed
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)
        x_encoder = self.norm(x_encoder)

        if self.moe:
            gt_intent_sequence=data['intent'][:,0] if mode and self.intent_label else  None
            y_hat = self.decoder(x_encoder, mode, key_padding_mask=~key_valid_mask, gt_intent_sequence=gt_intent_sequence)
        else:
            x_agent = x_encoder[:, :segment]
            y_hat = self.decoder(x_agent)
        x_others = x_encoder[:, segment:segment*(N), :]
        x_others = x_others.reshape(B, N-1,segment, -1) 
        x_others = x_others.reshape(B, N-1, -1)
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)
        ret_dict = {
            'y_hat': y_hat,
            'y_hat_others': y_hat_others,
        }
        return ret_dict
