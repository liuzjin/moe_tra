from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MoeDecoder, MultimodalDecoder, QueryBasedMoeDecoder
from .layers.mtr_decoder import TransformerDecoder
from .layers.transformer_blocks import Block, InteractionModule


class MoeMotion_I(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps=60,
        moe=True,
        use_transformer_decoder=False,
        query_cross_layers=2,
        query_self_atten=True,
        query_self_layers=1,
        num_experts=9,
        top_k=2,
        intent_label=True
    ) -> None:
        super().__init__()
        self.use_transformer_decoder = use_transformer_decoder
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

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
            if use_transformer_decoder:
                self.decoder = QueryBasedMoeDecoder(
                    dim=embed_dim, 
                    mlp_ratio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    query_cross_layers=query_cross_layers,
                    query_self_atten=query_self_atten,
                    query_self_layers=query_self_layers,
                    future_steps=future_steps, 
                    num_experts=num_experts, 
                    top_k=top_k,
                    intent_label=intent_label)
            else:
                self.decoder = MoeDecoder(embed_dim, future_steps, num_experts=num_experts, top_k=top_k, intent_label=intent_label)
        else:
            self.decoder = MultimodalDecoder(embed_dim, future_steps)
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, future_steps * 2)
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
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_valid_mask = data['lane_valid_mask']
        lane_normalized = data['lane_positions'] - data['lane_centers'].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data['x_centers'], data['lane_centers']], dim=1)
        angles = torch.cat([data['x_angles'][:, :, -1], data['lane_angles']], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)
        actor_type_embed = self.actor_type_embed[data['x_attr'][..., 2].long()]
       
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask = torch.cat(
            [data['x_key_valid_mask'], data['lane_key_valid_mask']], dim=1
        )
        x_type_mask = torch.cat([actor_feat.new_ones(*actor_feat.shape[:2]),
                                 lane_feat.new_zeros(*lane_feat.shape[:2])], dim=1).bool()

        x_encoder = x_encoder + pos_embed
        if isinstance(self, MoeMotion):
            # read memory for stream process
            if 'memory_dict' in data and data['memory_dict'] is not None:
                rel_pos = data['origin'] - data['memory_dict']['origin']
                rel_ang = (data['theta'] - data['memory_dict']['theta'] + torch.pi) % (2 * torch.pi) - torch.pi
                rel_ts = data['timestamp'] - data['memory_dict']['timestamp']
                memory_pose = torch.cat([
                    rel_ts.unsqueeze(-1), rel_ang.unsqueeze(-1), rel_pos
                ], dim=-1).float().to(x_encoder.device)
                memory_x_encoder = data['memory_dict']['x_encoder']
                memory_valid_mask = data['memory_dict']['x_mask']
                memory_type_mask = data['memory_dict']['x_type_mask']
            else:
                memory_pose = x_encoder.new_zeros(x_encoder.size(0), self.pose_dim)
                memory_x_encoder = x_encoder
                memory_valid_mask = key_valid_mask
                memory_type_mask = x_type_mask
            cur_pose = torch.zeros_like(memory_pose)

            # scene interaction
            new_x_encoder = x_encoder
            C = x_encoder.size(-1)
            # # new_x_encoder = self.scene_interact(new_x_encoder, memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask)
            # new_actor_feat = self.scene_interact(new_x_encoder[x_type_mask].reshape(B, -1, C), memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask)
            # new_lane_feat = self.scene_interact(new_x_encoder[~x_type_mask].reshape(B, -1, C), memory_x_encoder[~memory_type_mask].reshape(B, -1, C), cur_pose, memory_pose, key_padding_mask=~memory_valid_mask[~memory_type_mask].reshape(B, -1))
            # new_x_encoder = torch.cat([new_actor_feat, new_lane_feat], dim=1)
            # x_encoder = new_x_encoder * key_valid_mask.unsqueeze(-1) + x_encoder * ~key_valid_mask.unsqueeze(-1)

            new_actor_feat = self.scene_interact(new_x_encoder[x_type_mask].reshape(B, -1, C), memory_x_encoder, cur_pose, memory_pose, key_padding_mask=~memory_valid_mask)
            lane_feat = new_x_encoder[~x_type_mask].reshape(B, -1, C)
            new_x_encoder = torch.cat([new_actor_feat, lane_feat], dim=1)
            x_encoder = new_x_encoder * key_valid_mask.unsqueeze(-1) + x_encoder * ~key_valid_mask.unsqueeze(-1)
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)
        x_encoder = self.norm(x_encoder)

        if self.moe:
            y_hat = self.decoder(x_encoder, mode, key_padding_mask=~key_valid_mask)
        else:
            x_agent = x_encoder[:, 0]
            y_hat = self.decoder(x_agent)
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)
        
        cos, sin = data['theta'].cos(), data['theta'].sin()
        rot_mat = data['theta'].new_zeros(B, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos


        if isinstance(self, MoeMotion):
            # traj interaction
            if 'memory_dict' in data and data['memory_dict'] is not None:
                memory_y_hat = data['memory_dict']['glo_y_hat']
                memory_x_mode = data['memory_dict']['x_mode']
                ori_idx = ((data['timestamp'] - data['memory_dict']['timestamp']) / 0.1).long() - 1
                memory_traj_ori = torch.gather(memory_y_hat, 2, ori_idx.reshape(
                    B, 1, -1, 1).repeat(1, memory_y_hat.size(1), 1, memory_y_hat.size(-1)))
                memory_y_hat = torch.bmm((memory_y_hat - memory_traj_ori).reshape(B, -1, 2), rot_mat
                                        ).reshape(B, memory_y_hat.size(1), -1, 2)
                traj_embed = self.traj_embed(y_hat.detach().reshape(B, y_hat.size(1), -1))
                memory_traj_embed = self.traj_embed(memory_y_hat.reshape(B, memory_y_hat.size(1), -1))
                x_mode = self.traj_interact(x_mode, memory_x_mode, cur_pose, memory_pose,
                                                    cur_pos_embed=traj_embed,
                                                    memory_pos_embed=memory_traj_embed)
                y_hat_diff = self.stream_loc(x_mode).reshape(B, y_hat.size(1), -1, 2)
                y_hat = y_hat + y_hat_diff

        ret_dict = {
            'y_hat': y_hat,
            'y_hat_others': y_hat_others,
        }

        glo_y_hat = torch.bmm(y_hat['predictions'].detach().reshape(B, -1, 2), torch.inverse(rot_mat))
        glo_y_hat = glo_y_hat.reshape(B, y_hat['predictions'].size(1), -1, 2)

        if isinstance(self, MoeMotion):
            memory_dict = {
                'x_encoder': x_encoder,
                'x_mode': y_hat['mode'],
                'glo_y_hat': glo_y_hat,
                'x_mask': key_valid_mask,
                'x_type_mask': x_type_mask,
                'origin': data['origin'],
                'theta': data['theta'],
                'timestamp': data['timestamp'],
            }
            ret_dict['memory_dict'] = memory_dict

        return ret_dict

class MoeMotion(MoeMotion_I):
    def __init__(self, 
                 use_stream_encoder=True,
                 use_stream_decoder=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_stream_encoder = use_stream_encoder
        self.use_stream_decoder = use_stream_decoder
        self.embed_dim = kwargs['embed_dim']
        self.pose_dim = 4
        self.scene_interact = InteractionModule(
            dim=kwargs['embed_dim'],
            pose_dim=self.pose_dim,
            num_heads=kwargs['num_heads'],
            mlp_ratio=kwargs['mlp_ratio'],
            qkv_bias=kwargs['qkv_bias'],
        )

        self.traj_interact = InteractionModule(
            dim=kwargs['embed_dim'],
            pose_dim=self.pose_dim,
            num_heads=kwargs['num_heads'],
            mlp_ratio=kwargs['mlp_ratio'],
            qkv_bias=kwargs['qkv_bias'],
        )

        self.stream_loc = nn.Sequential(
            nn.Linear(kwargs['embed_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, kwargs['embed_dim']),
            nn.ReLU(),
            nn.Linear(kwargs['embed_dim'], kwargs['future_steps'] * 2),
        )
        self.traj_embed = nn.Sequential(
            nn.Linear(kwargs['future_steps'] * 2, kwargs['embed_dim']),
            nn.GELU(),
            nn.Linear(kwargs['embed_dim'], kwargs['embed_dim']),
        )
        
