from collections import defaultdict
import math
import os
from pathlib import Path
import pickle
import random
from typing import Dict
import pytorch_lightning as pl
from realmotion.metrics.accuracy import CustomAccuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.nn.utils.rnn import pad_sequence
from realmotion.metrics import MR, minADE, minFDE, brier_minFDE
from realmotion.metrics.lapulas import LaplaceNLLLoss
from realmotion.utils.optim import WarmupCosLR
from realmotion.utils.submission_av2 import SubmissionAv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        optim: dict = None,
    ) -> None:
        super(BaseLightningModule, self).__init__()
        self.pre_ensemble = True
        self.time_list = []
        self.optim = optim
        # self.save_hyperparameters()

        self.model = model
        self.metrics = MetricCollection(
            {
                'minADE1': minADE(k=1),
                'minADE6': minADE(k=6),
                'minFDE1': minFDE(k=1),
                'minFDE6': minFDE(k=6),
                'MR': MR(),
                'b-minFDE6': brier_minFDE(k=6),
                
            }
        )

    def forward(self, data, mode):
        return self.model(data, mode)
    def cal_loss(self, out, data, tag=''):
        y_hat, pi, y_hat_others = out['y_hat'], out['pi'], out['y_hat_others']
        new_y_hat = out.get('new_y_hat', None)
        y, y_others = data['target'][:, 0], data['target'][:, 1:]
        if new_y_hat is None:
            l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            # y_hat:[32, 6, 60, 2]   y:[32, 60, 2]
        else:
            l2_norm = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
        if new_y_hat is not None:
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0

        others_reg_mask = data['target_mask'][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + new_agent_reg_loss
        disp_dict = {
            f'{tag}loss': loss.item(),
            f'{tag}reg_loss': agent_reg_loss.item(),
            f'{tag}cls_loss': agent_cls_loss.item(),
            f'{tag}others_reg_loss': others_reg_loss.item(),
        }
        if new_y_hat is not None:
            disp_dict[f'{tag}reg_loss_refine'] = new_agent_reg_loss.item()

        return loss, disp_dict

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data, True)
        
        out['pi'] = out['y_hat']['logits']
        out['y_hat'] = out['y_hat']['predictions']
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data, False)
        out['pi'] = out['y_hat']['logits']
        out['y_hat'] = out['y_hat']['predictions']
        _, loss_dict = self.cal_loss(out, data)
        metrics = self.metrics(out, data['target'][:, 0])

        self.log(
            'val/reg_loss',
            loss_dict['reg_loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def on_test_start(self) -> None:
        save_dir = Path('./submission')
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )

    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        self.submission_handler.format_data(data, out['y_hat'], out['pi'])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            nn.GRUCell,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    '%s.%s' % (module_name, param_name) if module_name else param_name
                )
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                'params': [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                'weight_decay': self.optim.weight_decay,
            },
            {
                'params': [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.optim.lr, weight_decay=self.optim.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.optim.lr,
            min_lr=self.optim.min_lr,
            warmup_ratio=self.optim.warmup_ratio,
            epochs=self.optim.epochs,
        )
        return [optimizer], [scheduler]

class RegLightningModule(BaseLightningModule):
    def cal_loss(self, out, data, tag=''):
        y_hat, pi, y_hat_others = out['y_hat'], out['pi'], out['y_hat_others']
        propose = out.get('propose', None)
        y, y_others = data['target'][:, 0], data['target'][:, 1:]
        l2_norm = torch.norm(propose[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_propose_best = propose[torch.arange(y_hat_others.shape[0]), best_mode]
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        agent_pro_reg_loss = F.smooth_l1_loss(y_propose_best[..., :2], y)
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        others_reg_mask = data['target_mask'][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        loss = agent_pro_reg_loss + agent_reg_loss + agent_cls_loss + others_reg_loss 
        disp_dict = {
            f'{tag}loss': loss.item(),
            f'{tag}pro_reg_loss': agent_pro_reg_loss.item(),
            f'{tag}reg_loss': agent_reg_loss.item(),
            f'{tag}cls_loss': agent_cls_loss.item(),
            f'{tag}others_reg_loss': others_reg_loss.item(),
        }

        return loss, disp_dict

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data, True)
        
        out['pi'] = out['y_hat']['logits']
        out['propose'] = out['y_hat']['propose']
        out['y_hat'] = out['y_hat']['predictions']
        
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss
    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data, False)
        out['pi'] = out['y_hat']['logits']
        out['propose'] = out['y_hat']['propose']
        out['y_hat'] = out['y_hat']['predictions']
        
        _, loss_dict = self.cal_loss(out, data)
        metrics = self.metrics(out, data['target'][:, 0])

        
        for k, v in loss_dict.items():
            self.log(
                f'val/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )


class MoeLightningModule(BaseLightningModule):
    def __init__(self,
                 total_epochs=100,
                 modes=6,
                 top_k=2,
                 n_step=10,
                 history_frames=50,
                 num_experts=9,
                 intent_label=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.total_epochs = total_epochs
        self.modes = modes
        self.top_k = top_k
        self.n_step = n_step
        self.history_frames = history_frames
        self.n = 60 // n_step
        self.initial_ss_ratio = 1.0 # 初始teacher forcing概率
        self.final_ss_ratio = 0.1   # 最终teacher forcing概率
        self.intent_label = intent_label
        self.num_experts = num_experts
        self.metrics = MetricCollection(
            {
                'minADE1': minADE(k=1),
                'minADE6': minADE(k=6),
                'minFDE1': minFDE(k=1),
                'minFDE6': minFDE(k=6),
                'MR': MR(),
                'b-minFDE6': brier_minFDE(k=6)
            }
        )
    
    def cal_loss(self, out, data):
        # --- 1. 准备真值数据 ---
        # 确保只取需要的总长度，例如 60 步
        gt_traj = data['target'][:, 0, :] 
        y_others = data['target'][:, 1:]
        others_reg_mask = data['target_mask'][:, 1:]

        # --- 2. 准备模型输出 ---
        # 从新的模型输出 'out' 中获取数据
        # 注意：为了清晰，我们假设 'out' 就是模型直接的返回，不再有 'y_hat' 嵌套
        predictions = out['y_hat']['predictions']             # (B, K, T, 2) - K个多模态轨迹
        y_hat_others = out.get('y_hat_others')       # 其他智能体的预测
        mode_logits = out['y_hat']['pi']                  # (B, K) - 全局模态概率
        segment_logits_per_mode = out['y_hat']['segment_logits_per_mode'] # (B, K, S, N) - 分段意图logits

        # --- 3. 计算核心损失 ---

        # a) 多模态回归损失 (Oracle Loss)
        #    找到 K 个模态中，离真值最近的那个
        gt_expanded = gt_traj.unsqueeze(1) # (B, 1, T, 2)
        # 计算每个模态的平均L2误差
        l2_dist_per_mode = torch.norm(predictions - gt_expanded, p=2, dim=-1).mean(dim=-1) # (B, K)
        
        # best_mode_indices 是每个样本中最优模态的索引 (B,)
        # 这是我们后续损失计算的 "伪标签" 或 "Oracle"
        _, best_mode_indices = torch.min(l2_dist_per_mode, dim=-1)
        
        # 从所有预测中，选出每个样本对应的最优轨迹
        # predictions[torch.arange(batch_size), best_mode_indices]
        y_hat_best = predictions[torch.arange(predictions.shape[0]), best_mode_indices] # (B, T, 2)
        
        # 回归损失只基于这个最优轨迹进行计算
        regression_loss = F.smooth_l1_loss(y_hat_best, gt_traj)

        # b) 全局模态概率损失
        #    目标是让门控网络学会给最优的那个模态（best_mode_indices）赋予最高概率
        #    这变成了一个标准的分类问题
        mode_gating_loss = F.cross_entropy(mode_logits, best_mode_indices.detach())

        # c) 分段意图分类损失 (Gating Loss for Segments)
        segment_gating_loss = torch.tensor(0.0, device=gt_traj.device)
        # 只在训练时、且开启了意图标签模式、且数据中真的有标签时，才计算
        if self.intent_label:
            gt_intent_seq = data.get('intent') # (B, S)
            gt_intent_seq = gt_intent_seq[:, 0, :]
            if gt_intent_seq is not None:
                # 从 (B, K, S, N) 的分段logits中，只选出那个被认定为最优模态的logits
                # 使用 gather 高效地选取
                best_segment_logits = segment_logits_per_mode[torch.arange(gt_traj.shape[0]), best_mode_indices] # (B, S, N)
                
                # 计算交叉熵损失
                # input: (B*S, N), target: (B*S)
                segment_gating_loss = F.cross_entropy(
                    best_segment_logits.reshape(-1, self.num_experts), 
                    gt_intent_seq.reshape(-1)
                )
        else:
            segment_probs_per_mode = torch.softmax(segment_logits_per_mode, dim=-1)
            avg_expert_usage = torch.mean(segment_probs_per_mode, dim=[0, 1])
            segment_gating_loss = torch.mean(torch.sum(avg_expert_usage**2, dim=1)) * self.num_experts


        # d) 其他智能体的回归损失 (保持不变)
        others_reg_loss = torch.tensor(0.0, device=gt_traj.device)
        if y_hat_others is not None and y_others.numel() > 0:
            if others_reg_mask.sum() > 0:
                others_reg_loss = F.smooth_l1_loss(
                    y_hat_others[others_reg_mask], y_others[others_reg_mask]
                )

        # --- 4. 计算总损失 ---
        # 权重 g_weight_mode, g_weight_segment, o_weight 是需要调整的超参数
        g_weight_mode = 0.0
        g_weight_segment = 1.0
        o_weight = 1.0 # 其他智能体的损失权重

        
        total_loss = (regression_loss + 
                    g_weight_mode * mode_gating_loss + 
                    g_weight_segment * segment_gating_loss +
                    o_weight * others_reg_loss)
        
        # --- 5. 构建日志字典 ---
        loss_dict = {
            'total_loss': total_loss.item(),
            'regression_loss': regression_loss.item(),
            'mode_gating_loss': mode_gating_loss.item(),
            'segment_gating_loss': segment_gating_loss.item(),
            'others_reg_loss': others_reg_loss.item(),
        }

        return total_loss, loss_dict
    
    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        self.train()
        out = self(data, True)
        loss, loss_dict = self.cal_loss(out,data)

        self.log_dict({f'train/{k}': v for k, v in loss_dict.items()}, prog_bar=True)
        self.log('train/loss', loss, prog_bar=True)
        return loss
    def cal_eval_loss(self, out, data):
        """
        计算用于监控的评估损失。
        这个损失反映了模型在没有"上帝视角"下的真实性能。
        """
        gt_traj = data['target'][:, 0, :]
        y_others = data['target'][:, 1:]
        others_reg_mask = data['target_mask'][:, 1:]
        
        # --- 1. 准备模型输出 ---
        predictions = out['y_hat']['predictions'] # (B, K, T, 2)
        mode_logits = out['y_hat']['pi']      # (B, K)


        # --- 2. Top-1 回归损失 ---
        #    找到概率最高的模态，并计算其与真值的回归误差。
        #    这是评估模型单轨迹预测性能的核心。
        _, top1_indices = torch.max(mode_logits, dim=-1)
        top1_predictions = predictions[torch.arange(predictions.shape[0]), top1_indices]
        
        top1_regression_loss = F.smooth_l1_loss(top1_predictions, gt_traj)

        # --- 3. 分段意图准确率 (Gating Accuracy) ---
        #    如果提供了意图标签，我们可以计算门控的准确率，而不是损失。
        #    这能更直观地反映门控网络的性能。
        segment_gating_acc = torch.tensor(0.0, device=gt_traj.device)
        if self.intent_label and 'intent' in data:
            gt_intent_seq = data.get('intent') # (B, S)
            gt_intent_seq = gt_intent_seq[:, 0, :]
            if gt_intent_seq is not None:
                # 找到 Top-1 模态对应的专家规划路径
                segment_logits_per_mode = out['y_hat']['segment_logits_per_mode'] # (B, K, S, N)
                top1_segment_logits = segment_logits_per_mode[torch.arange(predictions.shape[0]), top1_indices] # (B, S, N)
                
                # 计算预测的意图
                pred_intent_seq = torch.argmax(top1_segment_logits, dim=-1) # (B, S)
                
                # 计算准确率
                segment_gating_acc = (pred_intent_seq == gt_intent_seq).float().mean()
                
        # --- 4. 其他智能体的损失 (保持可选) ---
        others_reg_loss = torch.tensor(0.0, device=gt_traj.device)
        y_hat_others = out.get('y_hat_others')
        others_reg_loss = F.smooth_l1_loss(
        y_hat_others[others_reg_mask], y_others[others_reg_mask]) 

        # --- 5. 计算总损失 ---
        # 在评估时，总损失主要由 top-1 回归损失定义
        total_loss = top1_regression_loss + others_reg_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'top1_regression_loss': top1_regression_loss.item(),
            'segment_gating_accuracy': segment_gating_acc.item(), # 改为记录准确率
            'others_reg_loss': others_reg_loss.item(),
        }
        
        return total_loss, loss_dict
    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        self.eval()
        out = self(data, False)
        
        # --- 2. 计算并记录验证损失 ---
        # 调用下面新写的 cal_eval_loss
        loss, loss_dict = self.cal_loss(out, data)
        
        # 添加 stage 前缀 (val/ or test/) 并记录
        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, 
                    on_step=False, on_epoch=True, sync_dist=True)
        segment_expert_logits = out['y_hat']['segment_logits_per_mode']
        segment_expert_probs = torch.softmax(segment_expert_logits, dim=-1) # (B, K, S, N)
        topk_probs, _ = torch.topk(segment_expert_probs, k=self.top_k, dim=-1)
        confidence_per_segment = torch.sum(topk_probs, dim=-1) # (B, K, S)
        trajectory_confidence_score = torch.sum(
            torch.log(confidence_per_segment + 1e-9), # 加 epsilon 防止 log(0)
            dim=2 
        ) # 最终形状: (B, K)
        out = {
            'y_hat': out['y_hat']['predictions'],
            'pi': trajectory_confidence_score,
            'y_hat_others': out['y_hat_others'],
        }
        metrics = self.metrics(out, data['target'][:, 0])
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def test_step(self, data, batch_idx) -> None:
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            out = self(cur_data)
            all_outs.append(out)
        self.submission_handler.format_data(data[-1], all_outs[-1]['y_hat'], all_outs[-1]['pi'])


class Hierarchical_Moe(MoeLightningModule):

    def cal_loss(self, out, data):
        gt_traj = data['target'][:, 0]
        y_others = data['target'][:, 1:]
        others_reg_mask = data['target_mask'][:, 1:]
        gt_macro_intent = data.get('intent')[:,0]  # (B,)
        
        # --- 1. 回归损失 ---
        # 训练时 out['predictions'] 的形状是 (B, 1, T, 2)
        predictions = out['y_hat']['predictions'] # (B, T, 2)
        y_hat_others = out.get('y_hat_others')
        mode_logits = out['y_hat']['pi']
        intent_plans_logits = out['y_hat']['intent_plans_logits'] 

        gt_expanded = gt_traj.unsqueeze(1) # (B, 1, T, 2)
        l2_dist_per_mode = torch.norm(predictions - gt_expanded, p=2, dim=-1).mean(dim=-1) # (B, K)
        _, best_mode_indices = torch.min(l2_dist_per_mode, dim=-1)

        y_hat_best = predictions[torch.arange(gt_traj.shape[0]), best_mode_indices]
        regression_loss = F.smooth_l1_loss(y_hat_best, gt_traj)

        others_reg_loss = torch.tensor(0.0, device=gt_traj.device)
        if y_hat_others is not None and y_others.numel() > 0:
            if others_reg_mask.sum() > 0:
                others_reg_loss = F.smooth_l1_loss(
                    y_hat_others[others_reg_mask], y_others[others_reg_mask]
                )

        mode_gating_loss = F.cross_entropy(mode_logits, best_mode_indices.detach())
        
        if self.intent_label:
                best_plan_logits = intent_plans_logits[torch.arange(gt_traj.shape[0]), best_mode_indices] # (B, S, N)
                planner_loss_with_gt = F.cross_entropy(best_plan_logits.reshape(-1, self.num_experts), gt_macro_intent.reshape(-1))
        else:
                planner_loss_with_gt = torch.tensor(0.0, device=gt_traj.device)
        g_weight_mode = 1.0
        g_weight_segment = 0.1
        o_weight = 1.0 # 其他智能体的损失权重

        
        total_loss = (regression_loss + 
                    g_weight_mode * mode_gating_loss + 
                    g_weight_segment * planner_loss_with_gt +
                    o_weight * others_reg_loss)
        
        # --- 5. 构建日志字典 ---
        loss_dict = {
            'total_loss': total_loss.item(),
            'regression_loss': regression_loss.item(),
            'mode_gating_loss': mode_gating_loss.item(),
            'segment_gating_loss': planner_loss_with_gt.item(),
            'others_reg_loss': others_reg_loss.item(),
        }

        return total_loss, loss_dict

class Intent_linearModule(MoeLightningModule):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics_new = self.metrics.clone(prefix="new_")
        self.validation_epoch_cache = {
            'intent_weights': [],      # [B, num_modes, 60, num_intents]
            'trajectory_preds': [],    # [B, num_modes, 60, 2]
            'trajectory_gt': [],       # [B, 60, 2]
            'best_modes': [],          # [B]
            'intent_assignments': []   # 每个样本的主导意图
        }
    def plot_intent_attention_over_time(self, attention_weights, ground_truth_modes=None):
        """
        attention_weights: [batch, future_steps, num_intents]
        ground_truth_modes: 可选，真实的模态标签
        
        """
        

        for batch_idx in range(attention_weights.shape[0]):
        # batch_idx = 0  # 可视化第一个样本
        
        # 取softmax后的权重
            weights = attention_weights[batch_idx].detach().cpu().numpy()  # [60, num_intents]
            
            # plt.figure(figsize=(14, 6))
            
            # # 热力图
            # sns.heatmap(weights.T, 
            #             cmap='YlOrRd', 
            #             xticklabels=np.arange(0, 60, 5),
            #             yticklabels=np.arange(weights.shape[1]),
            #             cbar_kws={'label': 'Attention Weight'})
            
            # plt.title(f'Intent Attention Over Time - Sample {batch_idx}')
            # plt.xlabel('Future Time Step')
            # plt.ylabel('Intent ID')
            
            # # 如果有真值，标注出来
            # if ground_truth_modes is not None:
            #     true_mode = ground_truth_modes[batch_idx]
            #     plt.axhline(y=true_mode, color='blue', linestyle='--', 
            #             label=f'Ground Truth Mode: {true_mode}')
            #     plt.legend()
            
            # plt.tight_layout()
            # plt.savefig(f'{batch_idx}intent_attention_timeline.png', dpi=150)
            # plt.show()
            
            # 统计：意图切换频率
            intent_choice = np.argmax(weights, axis=1)
            switches = np.sum(intent_choice[:-1] != intent_choice[1:])
            print(f"{batch_idx}意图切换次数: {switches}/59 步")
    
    def analyze_intent_traj_distribution(
            self,
            attention_weights: torch.Tensor,   # [B, M, T, K]
            absolute_trajectory: torch.Tensor, # [B, M, T, 2]  ← 你的 new_y_hat
            trajectory_gt: torch.Tensor,       # [B, T, 2]
            best_modes: torch.Tensor,          # [B]
            t_start: int = 0,                  # 可选：只分析 [t_start, t_end] 区间
            t_end: int = 60,
    ) -> Dict[str, float]:
        """
        只在相对位移空间度量同一意图下的预测一致性。
        返回指标均基于位移，cumsum 后的终点仅作参考。
        """
        B, M, T, K = attention_weights.shape
        device = attention_weights.device

        # 1. 还原相对位移
        rel_pred = absolute_trajectory[:, :, 1:t_end, :] - absolute_trajectory[:, :, 0:t_end-1, :]  # [B, M, T-1, 2]
        # 补零对齐长度
        rel_pred = F.pad(rel_pred, (0, 0, 1, 0), "constant", 0.0)  # [B, M, T, 2]

        # 2. 真值相对位移
        rel_gt = trajectory_gt[:, 1:t_end, :] - trajectory_gt[:, 0:t_end-1, :]
        rel_gt = F.pad(rel_gt, (0, 0, 1, 0), "constant", 0.0)  # [B, T, 2]

        # 3. 主导意图（最佳模态 + 时间平均）
        dominant_intents = []
        for b in range(B):
            mode_idx = best_modes[b].item()
            # 时间维度平均权重 → [K]
            mode_weights = attention_weights[b, mode_idx, t_start:t_end].mean(dim=0)
            dominant_intent = torch.argmax(mode_weights).item()
            dominant_intents.append(dominant_intent)

        # 4. 按意图聚合相对位移
        intent_disp_cache = defaultdict(list)   # 相对位移
        intent_endpoint_cache = defaultdict(list)  # cumsum 后终点（仅参考）

        for b in range(B):
            intent_id = dominant_intents[b]
            mode_idx = best_modes[b].item()

            # 相对位移
            disp = rel_pred[b, mode_idx].detach().cpu()  # [T, 2]
            intent_disp_cache[intent_id].append(disp)

            # cumsum 后终点（仅参考）
            abs_traj = absolute_trajectory[b, mode_idx].detach().cpu()
            endpoint = abs_traj[-1, :2].numpy()
            intent_endpoint_cache[intent_id].append(endpoint)

        # 5. 位移空间统计
        metrics = {}
        all_disp_vars = []        # 每意图位移方差
        all_endpoint_stds = []    # 每意图终点方差（参考）

        for intent_id in range(K):
            disps = intent_disp_cache[intent_id]
            if len(disps) < 3:
                metrics[f'intent_{intent_id}_disp_var'] = 0.0
                metrics[f'intent_{intent_id}_endpoint_std'] = 0.0
                metrics[f'intent_{intent_id}_usage_count'] = len(disps)
                continue

            # 位移方差：逐时间步求方差再平均
            disp_tensor = torch.stack(disps, dim=0)  # [N, T, 2]
            per_step_var = torch.var(disp_tensor, dim=0).mean(dim=-1)  # [T]
            avg_disp_var = per_step_var.mean().item()
            all_disp_vars.append(avg_disp_var)

            # 终点方差（参考）
            endpoints = np.stack(intent_endpoint_cache[intent_id], axis=0)
            endpoint_std = np.std(endpoints, axis=0).mean()
            all_endpoint_stds.append(endpoint_std)

            metrics[f'intent_{intent_id}_disp_var'] = avg_disp_var
            metrics[f'intent_{intent_id}_endpoint_std'] = endpoint_std
            metrics[f'intent_{intent_id}_usage_count'] = len(disps)

        # 6. 全局统计
        if all_disp_vars:
            metrics['avg_displacement_var'] = np.mean(all_disp_vars)
            metrics['max_displacement_var'] = np.max(all_disp_vars)
            metrics['avg_endpoint_std'] = np.mean(all_endpoint_stds)
            metrics['intent_usage_entropy'] = self._compute_usage_entropy(
                [metrics[f'intent_{i}_usage_count'] for i in range(K)]
            )

        # 7. 位移方差阈值（cumsum 场景）
        if metrics['avg_displacement_var'] > 0.08:
            print(f"🔴 位移方差 {metrics['avg_displacement_var']:.3f} 过高 "
                f"→ 同一意图下相对位移差异大，路由失效")
        else:
            print(f"✅ 位移方差 {metrics['avg_displacement_var']:.3f} 正常 "
                f"→ 相对位移一致性良好")

        return metrics
    def _compute_usage_entropy(self, usage_counts):
        """计算意图使用分布的熵"""
        total = sum(usage_counts)
        if total == 0:
            return 0.0
        probs = [count / total for count in usage_counts if count > 0]
        entropy = -sum(p * np.log(p) for p in probs)
        return entropy

    # ==================== 可视化方法 ====================

    def visualize_intent_traj_distribution(self, 
                                         intent_endpoint_cache: Dict[int, np.ndarray],
                                         epoch: int,
                                         save_dir: str = 'val_diagnostics'):
        """可视化每个意图的终点分布"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(4, 8, figsize=(32, 16))  # 假设32个意图
        axes = axes.flatten()
        
        for intent_id, endpoints in intent_endpoint_cache.items():
            if len(endpoints) < 2:
                continue
            
            endpoints = np.stack(endpoints, axis=0)  # [N, 2]
            
            ax = axes[intent_id]
            scatter = ax.scatter(endpoints[:, 0], endpoints[:, 1], 
                               alpha=0.6, s=20, c=range(len(endpoints)), 
                               cmap='viridis')
            ax.set_title(f'Intent {intent_id}\nN={len(endpoints)}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # 隐藏未使用的子图
        for i in range(len(intent_endpoint_cache), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Intent-wise Endpoint Distribution - Epoch {epoch}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'intent_endpoints_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path

    # ==================== 重写validation_step ====================

    # def validation_step(self, data, batch_idx):
    #     if isinstance(data, list):
    #         data = data[-1]
        
    #     # 1. 前向传播（启用中间输出）
    #     out = self(data, False)
        
    #     # 2. 计算损失
    #     loss, loss_dict = self.cal_loss(out, data)
        
    #     # 3. 提取关键数据用于诊断
    #     # attention_weights形状: [B, num_modes, 60, num_intents]
    #     attention_weights = out['y_hat']['weights']
    #     trajectory_preds = out['y_hat']['y_hat']  # [B, num_modes, 60, 2]
    #     trajectory_gt = data['target'][:, 0]  # [B, 60, 2]
        
    #     # 获取最佳模态（用于归因）
    #     with torch.no_grad():
    #         l2_norm = torch.norm(trajectory_preds[..., :2] - trajectory_gt.unsqueeze(1), 
    #                            dim=-1).sum(dim=-1)  # [B, num_modes]
    #         best_modes = torch.argmin(l2_norm, dim=-1)  # [B]
        
    #     # 4. 进行意图-轨迹分布分析
    #     distrib_metrics = self.analyze_intent_traj_distribution(
    #         attention_weights, trajectory_preds, trajectory_gt, best_modes
    #     )
        
    #     # 5. 缓存数据（用于epoch结束时的全局分析）
    #     self.validation_epoch_cache['intent_weights'].append(attention_weights.detach().cpu())
    #     self.validation_epoch_cache['trajectory_preds'].append(trajectory_preds.detach().cpu())
    #     self.validation_epoch_cache['trajectory_gt'].append(trajectory_gt.detach().cpu())
    #     self.validation_epoch_cache['best_modes'].append(best_modes.detach().cpu())
        
    #     # 6. 记录指标（分离scalar和分布统计）
    #     scalar_metrics = {f"val/{k}": v for k, v in loss_dict.items()}
        
    #     # 分布统计指标（使用专门的prefix）
    #     distrib_metrics = {f"val/distrib/{k}": v for k, v in distrib_metrics.items()}
        
    #     # 合并记录
    #     self.log_dict(scalar_metrics, 
    #                  on_step=False, on_epoch=True, sync_dist=True)
    #     self.log_dict(distrib_metrics, 
    #                  on_step=False, on_epoch=True, sync_dist=True)
        
    #     # 7. 原始验证逻辑（计算ADE/FDE）
    #     out_for_metrics = {
    #         'y_hat': out['y_hat']['y_hat'],
    #         'pi': out['y_hat']['pi'],
    #         'new_y_hat': out['y_hat']['new_y_hat'],
    #         'new_pi': out['y_hat']['new_pi'],
    #     }
        
    #     metrics = self.metrics(out_for_metrics, data['target'][:, 0])
        
    #     if out_for_metrics['new_y_hat'] is not None:
    #         out_for_metrics['y_hat'] = out_for_metrics['new_y_hat']
    #         out_for_metrics['pi'] = out_for_metrics['new_pi']
    #         metrics_new = self.val_metrics_new(out_for_metrics, data['target'][:, 0])
    #         self.log_dict(metrics_new, 
    #                      prog_bar=True, on_step=False, on_epoch=True, 
    #                      batch_size=1, sync_dist=True)
        
    #     self.log_dict(metrics, 
    #                  prog_bar=True, on_step=False, on_epoch=True, 
    #                  batch_size=1, sync_dist=True)

    # ==================== 重写on_validation_epoch_end ====================

    # def on_validation_epoch_end(self):
    #     """在验证epoch结束时生成全局分析报告"""
    #     if not self.validation_epoch_cache['intent_weights']:
    #         return
        
    #     print(f"\n{'='*80}")
    #     print(f"VALIDATION EPOCH {self.current_epoch} - INTENT TRAJECTORY DISTRIBUTION REPORT")
    #     print(f"{'='*80}")
        
    #     # 1. 聚合所有批次的数据
    #     all_weights = torch.cat(self.validation_epoch_cache['intent_weights'], dim=0)
    #     all_preds = torch.cat(self.validation_epoch_cache['trajectory_preds'], dim=0)
    #     all_gt = torch.cat(self.validation_epoch_cache['trajectory_gt'], dim=0)
    #     all_best_modes = torch.cat(self.validation_epoch_cache['best_modes'], dim=0)
        
    #     # 2. 全局分布分析（在验证集上）
    #     global_metrics = self.analyze_intent_traj_distribution(
    #         all_weights,  # 增加batch维度
    #         all_preds,
    #         all_gt,
    #         all_best_modes
    #     )
        
    #     # 3. 打印关键诊断信息
    #     print(f"\n📊 关键指标:")
    #     print(f"  平均终点标准差: {global_metrics.get('avg_endpoint_std', 0):.3f}m")
    #     print(f"  意图使用熵: {global_metrics.get('intent_usage_entropy', 0):.3f}")
    #     print(f"  意图稳定性: {global_metrics.get('intent_stability_mean', 0):.3f}")
        
    #     # 4. 生成可视化（仅在前几个epoch和最后一个epoch）
    #     if self.current_epoch in [0, 1, 2] or self.current_epoch == self.trainer.max_epochs - 1:
    #         # 构建意图终点缓存用于可视化
    #         intent_endpoint_cache = defaultdict(list)
    #         batch_size = all_preds.shape[0]
            
    #         for b in range(batch_size):
    #             mode_idx = all_best_modes[b]
    #             intent_weights = all_weights[b, mode_idx].mean(dim=0)  # [num_intents]
    #             dominant_intent = torch.argmax(intent_weights).item()
                
    #             endpoint = all_preds[b, mode_idx, -1, :2].numpy()
    #             intent_endpoint_cache[dominant_intent].append(endpoint)
            
    #         # 生成并记录可视化
    #         viz_path = self.visualize_intent_traj_distribution(
    #             intent_endpoint_cache, 
    #             self.current_epoch,
    #             save_dir=os.path.join("./figures", 'diagnostics')
    #         )
            
    #         if self.logger:
    #             # 记录到TensorBoard
    #             self.logger.experiment.add_image(
    #                 'diagnostics/intent_endpoint_distribution',
    #                 plt.imread(viz_path),
    #                 self.current_epoch,
    #                 dataformats='HWC'
    #             )
            
    #         print(f"📈 可视化已保存: {viz_path}")
        
    #     # 5. 自动诊断与建议
    #     self._auto_diagnose(global_metrics)
        
    #     # 6. 清空缓存
    #     self.validation_epoch_cache = {
    #         k: [] for k in self.validation_epoch_cache.keys()
    #     }
        
    #     print(f"{'='*80}\n")

    def _auto_diagnose(self, metrics: Dict[str, float]):
        """自动分析并给出改进建议"""
        print("\n💡 自动诊断与建议:")
        
        endpoint_std = metrics.get('avg_endpoint_std', 0)
        if endpoint_std < 0.6:
            print("  🔴 终点标准差过小 (< 0.6m)")
            print("     → 建议: 增加专家网络容量或添加残差分支")
            print("     → 参考: 实施方案G")
        
        usage_entropy = metrics.get('intent_usage_entropy', 0)
        if usage_entropy < 2.0:
            print("  🔴 意图使用熵过低 (< 2.0)")
            print("     → 建议: 添加负载均衡损失或减少意图数量")
            print("     → 参考: 实施方案B")
        
        stability = metrics.get('intent_stability_mean', 0)
        if stability > 0.95:
            print("  ⚠️  意图过于稳定 (> 0.95)")
            print("     → 建议: 添加早期时间步的意图多样性损失")
            print("     → 参考: 实施方案F")


    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data, False)
        _, loss_dict = self.cal_loss(out, data)

        self.log_dict({f"val/{k}": v for k, v in loss_dict.items()}, 
                    on_step=False, on_epoch=True, sync_dist=True)

        out = {
            'y_hat': out['y_hat']['y_hat'],
            'pi': out['y_hat']['pi'],
            'new_y_hat': out['y_hat']['new_y_hat'],
            'new_pi': out['y_hat']['new_pi'],
        }

        metrics = self.metrics(out, data['target'][:, 0])
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        if out['new_y_hat'] is not None:
            metrics_new = self.val_metrics_new(out, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def cal_loss(self, out, data, tag=""):
        y_hat, pi, y_hat_others = out["y_hat"]['y_hat'], out["y_hat"]["pi"], out["y_hat_others"]
        scal, scal_new = out["y_hat"]["scal"], out["y_hat"]["scal_new"]
        new_y_hat = out["y_hat"].get("new_y_hat", None)
        new_pi = out["y_hat"].get("new_pi", None)
        dense_predict = out["y_hat"].get("dense_pred", None)
        vq_loss =0.1* out["y_hat"].get("vq_loss", 0)

        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # loss for output of state query
        if dense_predict is not None:
            dense_reg_loss = F.smooth_l1_loss(dense_predict, y)
        else:
            dense_reg_loss = 0

        # loss for output of mode query
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)
        
        # self.plot_intent_attention_over_time(out['y_hat']['weights'][torch.arange(y_hat.shape[0]), best_mode])
        # loss for final output
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0
        if new_pi is not None:
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = 0

        # loss for other agents
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        # Laplace loss, which is not necessary
        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y)

        predictions['traj'] = new_y_hat
        predictions['scale'] = scal_new
        predictions['probs'] = new_pi
        laplace_loss_new = self.laplace_loss.compute(predictions, y)

        # total loss
        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + \
                new_agent_reg_loss + dense_reg_loss + new_pi_reg_loss +  vq_loss 
        loss = loss + laplace_loss + laplace_loss_new

        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
        }
        if new_y_hat is not None:
            disp_dict[f"{tag}reg_loss_refine"] = new_agent_reg_loss.item()
        if new_pi is not None:
            disp_dict[f"{tag}reg_loss_new_pi"] = new_pi_reg_loss.item()
        if dense_predict is not None:
            disp_dict[f"{tag}reg_loss_dense"] = dense_reg_loss.item()


        return loss, disp_dict

class StreamLightningModule(Intent_linearModule):
    def __init__(self,
                 num_grad_frame=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_grad_frame = num_grad_frame
    
    def training_step(self, data, batch_idx):
        total_step = len(data)
        num_grad_frames = min(self.num_grad_frame, total_step)
        num_no_grad_frames = total_step - num_grad_frames

        memory_dict = None
        self.eval()
        with torch.no_grad():
            for i in range(num_no_grad_frames):
                cur_data = data[i]
                cur_data['memory_dict'] = memory_dict
                out = self(cur_data, False)
                memory_dict = out['memory_dict']
        
        self.train()
        sum_loss = 0
        loss_dict = {}
        for i in range(num_grad_frames):
            cur_data = data[i + num_no_grad_frames]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data, True)
            cur_loss, cur_loss_dict = self.cal_loss(out, cur_data, f'{i}_')
            loss_dict.update(cur_loss_dict)
            sum_loss += cur_loss
            memory_dict = out['memory_dict']
        loss_dict['loss'] = sum_loss.item()
        for k, v in loss_dict.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return sum_loss
    
    def validation_step(self, data, batch_idx):
        memory_dict = None
        reg_loss_dict = {}
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data, False)
            _, cur_loss_dict = self.cal_loss(out, cur_data,f'{i}_')
            reg_loss_dict.update(cur_loss_dict)
            memory_dict = out['memory_dict']
            all_outs.append(out)
        

        out = {
            'y_hat': all_outs[-1]['y_hat']['y_hat'],
            'pi': all_outs[-1]['y_hat']['pi'],
            'new_y_hat': all_outs[-1]['y_hat']['new_y_hat'],
            'new_pi': all_outs[-1]['y_hat']['new_pi'],
        }

        metrics = self.metrics(out, data[-1]['target'][:, 0])
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        if out['new_y_hat'] is not None:
            metrics_new = self.val_metrics_new(out, data[-1]['target'][:, 0])

        for k, v in reg_loss_dict.items():
            self.log(
                f'val/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
    
    def test_step(self, data, batch_idx) -> None:
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            all_outs.append(out)
        self.submission_handler.format_data(data[-1], all_outs[-1]['y_hat'], all_outs[-1]['pi'])



class Cross_moe_mlp_Module(BaseLightningModule):
    def __init__(self,
                 num_grad_frame=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.metrics = MetricCollection(
            {
                'minADE1': minADE(k=1),
                'minADE6': minADE(k=6),
                'minFDE1': minFDE(k=1),
                'minFDE6': minFDE(k=6),
                'MR': MR(),
                'b-minFDE6': brier_minFDE(k=6),
            }
        )
    def forward(self, data, mode):
        return self.model(data, mode)
    def training_step(self, data, batch_idx):
        out = self(data, True)
        out['pi'] = out['y_hat']['pi']
        out['y_hat'] = out['y_hat']['predictions']
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss

    def validation_step(self, data, batch_idx):
        out = self(data, False)
        out['pi'] = out['y_hat']['pi']
        out['y_hat'] = out['y_hat']['predictions']
        _, loss_dict = self.cal_loss(out, data)
        metrics = self.metrics(out, data['target'][:, 0])

        self.log(
            'val/reg_loss',
            loss_dict['reg_loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
