import math
from pathlib import Path
import pickle
import random
import pytorch_lightning as pl
from realmotion.metrics.accuracy import CustomAccuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.nn.utils.rnn import pad_sequence
from realmotion.metrics import MR, minADE, minFDE, brier_minFDE
from realmotion.utils.optim import WarmupCosLR
from realmotion.utils.submission_av2 import SubmissionAv2


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


class StreamLightningModule(BaseLightningModule):
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
                out = self(cur_data)
                memory_dict = out['memory_dict']
        
        self.train()
        sum_loss = 0
        loss_dict = {}
        for i in range(num_grad_frames):
            cur_data = data[i + num_no_grad_frames]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            cur_loss, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i + num_no_grad_frames}_')
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
            out = self(cur_data)
            _, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i}_')
            reg_loss_dict[f'val/step{i}_reg_loss'] = cur_loss_dict[f'step{i}_reg_loss']
            memory_dict = out['memory_dict']
            all_outs.append(out)
        
        metrics = self.metrics(all_outs[-1], data[-1]['target'][:, 0])

        self.log_dict(
            reg_loss_dict,
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


class MoeLightningModule(BaseLightningModule):
    def __init__(self,
                 total_epochs=100,
                 modes=6,
                 k=2,
                 n_step=10,
                 history_frames=50,
                 num_experts=9,
                 intent_label=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.total_epochs = total_epochs
        self.modes = modes
        self.k = k
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
                'b-minFDE6': brier_minFDE(k=6),
                'IntentAcc': CustomAccuracy()
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
        if self.training and self.intent_label and 'intent' in data:
            gt_intent_seq = data.get('intent') # (B, S)
            gt_intent_seq = gt_intent_seq[:, 0, :]
            if gt_intent_seq is not None:
                # 从 (B, K, S, N) 的分段logits中，只选出那个被认定为最优模态的logits
                # 使用 gather 高效地选取
                indices_for_gather = best_mode_indices.view(-1, 1, 1, 1).expand(
                    -1, 1, self.n, self.num_experts
                )
                # best_segment_logits: (B, 1, S, N) -> (B, S, N)
                best_segment_logits = torch.gather(segment_logits_per_mode, 1, indices_for_gather).squeeze(1)

                # 计算交叉熵损失
                segment_gating_loss = F.cross_entropy(
                    best_segment_logits.reshape(-1, self.num_experts), 
                    gt_intent_seq.reshape(-1)
                )

        # d) 其他智能体的回归损失 (保持不变)
        others_reg_loss = torch.tensor(0.0, device=gt_traj.device)
        if y_hat_others is not None and y_others.numel() > 0:
            if others_reg_mask.sum() > 0:
                others_reg_loss = F.smooth_l1_loss(
                    y_hat_others[others_reg_mask], y_others[others_reg_mask]
                )
        aux_loss = torch.tensor(0.0, device=gt_traj.device)
        if not self.intent_label:
            # (B, K, S, N)
            segment_probs_per_mode = F.softmax(out['y_hat']['segment_logits_per_mode'], dim=-1)
            # 对 K 和 B 维度取平均，得到每个分段S对每个专家N的使用频率
            # (S, N)
            avg_usage = torch.mean(segment_probs_per_mode, dim=[0, 1])
            # 鼓励使用率的平方和最小，即鼓励均匀分布
            aux_loss = torch.mean(avg_usage**2) * self.num_experts

        # --- 4. 计算总损失 ---
        # 权重 g_weight_mode, g_weight_segment, o_weight 是需要调整的超参数
        g_weight_mode = 1.0
        g_weight_segment = 0.5
        aux_weight = 1.0
        o_weight = 1.0 # 其他智能体的损失权重

        velocity = predictions[:, :, :-1] - predictions[:, :, 1:]
        acceleration = velocity[:, :, :-1] - velocity[:, :, 1:]
        jerk_loss = torch.mean(torch.norm(acceleration, p=2, dim=-1).mean(dim=-1))
        jerk_weight = 0.01
        
        total_loss = (regression_loss + 
                    g_weight_mode * mode_gating_loss + 
                    g_weight_segment * segment_gating_loss +
                    aux_weight * aux_loss +
                    o_weight * others_reg_loss+
                    jerk_weight * jerk_loss)
        
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
        self.eval()
        out = self(data, False)
        
        # --- 2. 计算并记录验证损失 ---
        # 调用下面新写的 cal_eval_loss
        loss, loss_dict = self.cal_eval_loss(out, data)
        
        # 添加 stage 前缀 (val/ or test/) 并记录
        self.log_dict({f"{k}": v for k, v in loss_dict.items()}, 
                    on_step=False, on_epoch=True, sync_dist=True)
        mode_logits = out['y_hat']['pi']      # (B, K)
        _, top1_indices = torch.max(mode_logits, dim=-1)
        segment_logits_per_mode = out['y_hat']['segment_logits_per_mode'] # (B, K, S, N)
        top1_segment_logits = segment_logits_per_mode[torch.arange(out['y_hat']['predictions'].shape[0]), top1_indices] # (B, S, N)
        pred_intent_seq = torch.argmax(top1_segment_logits, dim=-1) # (B, S)
                
        out = {
            'y_hat': out['y_hat']['predictions'],
            'pi': out['y_hat']['pi'],
            'y_hat_others': out['y_hat_others'],
            'intent': pred_intent_seq,
            'intent_target': data['intent'][:, 0],
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
        gt_macro_intent = data.get('intent')[:,0,0]  # (B,)
        
        # --- 1. 回归损失 ---
        # 训练时 out['predictions'] 的形状是 (B, 1, T, 2)
        predictions = out['y_hat']['predictions'].squeeze(1) # (B, T, 2)
        y_hat_others = out.get('y_hat_others')
        regression_loss = F.smooth_l1_loss(predictions, gt_traj)
        others_reg_loss = F.smooth_l1_loss(
                    y_hat_others[others_reg_mask], y_others[others_reg_mask]
                )
        
        # --- 2. 顶层门控分类损失 ---
        macro_logits = out['y_hat']['logits'] # (B, M)
        gating_loss = F.cross_entropy(macro_logits, gt_macro_intent)
        
        # --- 总损失 ---
        total_loss = regression_loss + others_reg_loss + 1.0 * gating_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'regression_loss': regression_loss.item(),
            'gating_loss': gating_loss.item(),
            'others_reg_loss': others_reg_loss.item(),
        }
        return total_loss, loss_dict

    def validation_step(self, data, batch_idx):
        self.eval()
        out = self(data, False)
    
        mode_logits = out['y_hat']['logits']      # (B, K)
        _, top1_indices = torch.max(mode_logits, dim=-1)
        out = {
            'y_hat': out['y_hat']['predictions'],
            'pi': out['y_hat']['pi'],
            'y_hat_others': out['y_hat_others'],
            'intent': top1_indices,
            'intent_target': data['intent'][:, 0],
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