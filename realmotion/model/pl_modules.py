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
                'IntentAcc': CustomAccuracy()
            }
        )

    def forward(self, data):
        return self.model(data)

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
        out = self(data)
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
        out = self(data)
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
def get_teacher_forcing_ratio(
    epoch: int, 
    total_epochs: int, 
    schedule_type: str = 'linear', 
    initial_ratio: float = 1.0, 
    final_ratio: float = 0.0,
    inverse_sigmoid_k: float = None
) -> float:
    
    # --- 1. 线性衰减 (Linear Decay) ---
    if schedule_type == 'linear':
        if epoch >= total_epochs:
            return final_ratio
        # 确保 total_epochs > 0 以避免除零错误
        if total_epochs <= 0:
            return initial_ratio
        
        decay_rate = (initial_ratio - final_ratio) / total_epochs
        current_ratio = initial_ratio - (epoch * decay_rate)
        return max(final_ratio, current_ratio) # 确保不会低于最终值

    # --- 2. 指数衰减 (Exponential Decay) ---
    elif schedule_type == 'exponential':
        if initial_ratio <= final_ratio:
            return final_ratio
        if total_epochs <= 0:
            return initial_ratio

        # decay_rate = (final / initial) ^ (1 / total_epochs)
        decay_rate = (final_ratio / initial_ratio) ** (1.0 / total_epochs)
        
        current_ratio = initial_ratio * (decay_rate ** epoch)
        return current_ratio

    # --- 3. 逆Sigmoid衰减 (Inverse Sigmoid Decay) ---
    elif schedule_type == 'inverse_sigmoid':
        # 如果没有提供k值，使用一个合理的经验值
        if inverse_sigmoid_k is None:
            # k值越大，曲线越平缓，teacher forcing保持高位的时间越长
            inverse_sigmoid_k = float(total_epochs / 10)
            if inverse_sigmoid_k < 1.0:
                 warnings.warn(f"Calculated k for inverse_sigmoid is {inverse_sigmoid_k:.2f} which is very small. Consider setting it manually.")
                 inverse_sigmoid_k = 1.0

        if inverse_sigmoid_k <= 0:
             raise ValueError("k for inverse_sigmoid_decay must be positive.")

        return inverse_sigmoid_k / (inverse_sigmoid_k + math.exp(epoch / inverse_sigmoid_k))

    elif schedule_type == 'cosine':
        if epoch >= total_epochs:
            return final_ratio
        if total_epochs <= 0:
            return initial_ratio
            
        # 计算余弦曲线的部分
        cosine_part = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        
        # 将曲线映射到 [initial_ratio, final_ratio] 的范围
        current_ratio = final_ratio + (initial_ratio - final_ratio) * cosine_part
        return current_ratio
    # --- 4. 如果策略名无效，则报错 ---
    else:
        raise ValueError(
            f"Unknown schedule_type: '{schedule_type}'. "
            f"Please choose from 'linear', 'exponential', or 'inverse_sigmoid'."
        )


def nan_hook(module, input, output):
    if isinstance(output, torch.Tensor) and torch.any(torch.isnan(output)):
        print(f"NaN found in the output of layer: {module}")
        # 在这里可以设置一个断点或者抛出异常来中断程序
        # import pdb; pdb.set_trace()

class RegressionLightningModule(BaseLightningModule):
    def __init__(self,
                 total_epochs=100,
                 modes=6,
                 k=2,
                 n_step=10,
                 logits_max=True,
                 history_frames=50,
                 schedule_type='linear',
                 **kwargs):
        super().__init__(**kwargs)
        self.total_epochs = total_epochs
        self.modes = modes
        self.k = k
        self.n_step = n_step
        self.logits_max = logits_max
        self.history_frames = history_frames
        self.n = 60 // n_step
        self.initial_ss_ratio = 1.0 # 初始teacher forcing概率
        self.final_ss_ratio = 0.1   # 最终teacher forcing概率
        self.schedule_type = schedule_type

        for name, module in self.model.named_modules():
            module.register_forward_hook(nan_hook)
    
    def forward(self, data, mode):
        return self.model(data, mode)
    def cal_loss(self, out, data, tag):
        gt_segment, y_others = data['target'][:, 0], data['target'][:, 1:]
        others_reg_mask = data['target_mask'][:, 1:]
        
        gt_action = data['intent'][:,0,0]
        predictions = out['y_hat']['predictions']
        y_hat_others = out['y_hat_others']
        probs = out['y_hat']['probs']

        
        # 1. 门控损失 (分类)
        gating_loss = F.cross_entropy(probs, gt_action)
        if gating_loss.isinf():
            print("gating_loss is Inf!")
        
        # 2. 回归损失 ("最佳匹配者负责制")
        gt_expanded = gt_segment.unsqueeze(1) # -> (B, 1, T, 2)
        l2_norm = torch.norm(predictions[..., :2] - gt_expanded, dim=-1).sum(dim=-1)
        
        _, best_mode_indices = torch.min(l2_norm, dim=-1) # (B,)

        best_predictions = torch.gather(
            predictions, 1, 
            best_mode_indices.view(-1, 1, 1, 1).expand(-1, 1, predictions.size(2), 2)
        ).squeeze(1)
        
        regression_loss = F.smooth_l1_loss(best_predictions, gt_segment)
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )


        # 3. 总损失
        total_loss = regression_loss +  gating_loss  + others_reg_loss
        

        loss_dict = {
            f'{tag}loss': total_loss.item(),
            f'{tag}gating_loss': gating_loss.item(),
            f'{tag}regression_loss': regression_loss.item(),
            f'{tag}others_reg_loss': others_reg_loss.item(),
        }
        return total_loss, loss_dict
    
    def training_step(self, data, batch_idx):
        self.train()
        teacher_forcing_ratio = get_teacher_forcing_ratio(
            self.current_epoch, self.total_epochs, self.schedule_type, self.initial_ss_ratio, self.final_ss_ratio
        )
        # self.log('train/teacher_forcing_ratio', teacher_forcing_ratio, on_step=False, on_epoch=True)
        batch_size = data[0]['x_positions'].shape[0]  # 获取批次大小
        self.log('train/teacher_forcing_ratio', teacher_forcing_ratio, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        
        total_loss = 0.0 
        current_input = data[0]   
        for i in range(self.n):

            out = self(current_input, True)
            if current_input['target'].shape[1] != out['y_hat_others'].shape[1]+1:
                raise ValueError(
                    f"Expected output shape: {out['y_hat']['predictions'].shape[1]}, "
                    f"but got {current_input['target'].shape[1]}"
                )
            self.last_input = current_input
            loss, loss_dict = self.cal_loss(out,current_input, tag=f'step{i}_')
            if torch.isnan(loss):
                print(batch_idx)
                
            
            self.log_dict({f'train/{k}': v for k, v in loss_dict.items()}, prog_bar=True)
            total_loss += loss

            if i == self.n - 1:
                break

            use_teacher_forcing = (random.random() < teacher_forcing_ratio)
            # use_teacher_forcing = False
            if use_teacher_forcing:
                current_input = data[i+1]
            else:
                if self.logits_max:
                    # 策略1：选择logits最大的模态
                    _, best_mode_indices = torch.max(out['logits'], dim=1) # (B,)
                    
                else:
                    # 策略2：选择与真值最接近的模态
                    gt_expanded = current_input['target'][:, 0].unsqueeze(1) # -> (B, 1, T, 2)
                    l2_norm = torch.norm(out['y_hat']['predictions'] - gt_expanded, dim=-1).sum(dim=-1)
                    _, best_mode_indices = torch.min(l2_norm, dim=-1) # (B,)

                pred_segment = torch.gather(
                        out['y_hat']['predictions'], 1,
                        best_mode_indices.view(-1, 1, 1, 1).expand(-1, 1, self.n_step, 2)
                    )
                pred = torch.cat([pred_segment, out['y_hat_others']], dim=1)
                
                current_input = self.update_state_one_with_agent_alignment(current_input, pred, i,data[i+1])
        self.log('train/total_loss', total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, data, batch_idx):
        self.eval()
        beam_size = self.modes # 使用所有模态作为beam size
        
        history_data = data[0]['x_positions']
        gt_full_future_traj = data[0]['target'] # (B, 60, 2)
        
        # --- Beam Search 初始化 ---
        # `beams` is a list of tuples: (cumulative_log_prob, full_trajectory, last_input_state, memory)
        beams = [(torch.zeros(history_data.shape[0], device=self.device), # log_probs
                  None, # 初始轨迹
                  data[0],                # 初始输入
                  None)]                  # 专家索引

        # --- 自回归循环 ---
        for i in range(self.n): # 预测6个段落
            all_new_beams = []
            for log_probs, trajectories, last_input, expert in beams:
                # a. 准备输入并预测
                
                out = self(last_input, False)
                step_log_probs = torch.log(out['y_hat']['top_k_probs'])
                experts = out['y_hat']['top_k_indices']
                
                for j in range(self.k):

                    new_log_probs = log_probs + step_log_probs[:, j]
                    expert_idx = experts[:, [j]]  # (B,)
                    pred_av = out['y_hat']['predictions'][:, [j], ...] # (B, 10, 2)
                    pred = torch.cat([pred_av, out['y_hat_others']], dim=1)
                    
                    next_input = self.update_state_one_with_agent_alignment(last_input, pred, i)

                    if trajectories is None:
                        trajectories_pred = pred
                        expert_pred = expert_idx
                    else:
                        trajectories_pred = torch.cat([trajectories, pred], dim=2)
                        expert_pred = torch.cat([expert, expert_idx], dim=1)
                    
                    all_new_beams.append((new_log_probs, trajectories_pred, next_input,  expert_pred))
            
            # --- 筛选 Top-K Beams ---
            # 根据累积概率排序
            sorted_beams = sorted(all_new_beams, key=lambda x: x[0].sum(), reverse=True)
            beams = sorted_beams[:beam_size]
        
        # --- 循环结束，评估结果 ---
        # 选择最终概率最高的轨迹
        final_predictions = {'y_hat': [x[1][:, 0] for x in beams], 
                             'pi': [torch.exp(x[0]) for x in beams],
                             "intent": [x[3] for x in beams]}
    
        final_predictions['y_hat'] = torch.stack(final_predictions['y_hat'], dim=1) # (B, beam_size, 60, 2)
        final_predictions['pi'] = torch.stack(final_predictions['pi'], dim=1)
        final_predictions['pi'] = torch.softmax(final_predictions['pi'], dim=-1)
        final_predictions['intent'] = torch.stack(final_predictions['intent'], dim=1)
        final_predictions['intent_target'] = data[0]['intent'][:,0]
        # 计算评估指标

        metrics = self.metrics(final_predictions, gt_full_future_traj[:, 0])
        
        # unique_metrics = {
        # f"val_epoch{self.current_epoch}_step{self.global_step}/{k}": v 
        # for k, v in metrics.items()
        # }
        
        # self.log_dict(
        #     unique_metrics,  # 使用带唯一标识符的指标
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     sync_dist=True,
        # )

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
    
    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
    #                 print(f"!!! NaN/Inf gradient detected in parameter: {name} !!!")
    #                 # 在这里，你仍然可以访问 self.current_batch (如果保存了的话)
    #                 # 这说明当前批次的数据导致了梯度爆炸
    #                 # 注意：需要自己保存当前批次，因为 on_after_backward 没有 batch 参数
    # def on_before_optimizer_step(self, optimizer):
    #     """优化器步骤前的检查"""
    #     # 检查梯度
    #     total_norm = 0
    #     nan_grad_count = 0
        
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
    #                 nan_grad_count += 1
    #                 print(f"🚨 梯度问题在: {name}")
    #                 # 分析当前导致问题的batch
    #                 if hasattr(self, 'current_batch'):
    #                     self._analyze_problematic_batch()
    #             else:
    #                 param_norm = param.grad.detach().data.norm(2)
    #                 total_norm += param_norm.item() ** 2
        
    #     total_norm = total_norm ** 0.5
    #     print(f"梯度范数: {total_norm:.6f}, 有问题的梯度数: {nan_grad_count}")
        
    #     # 梯度裁剪
    #     if total_norm > 1.0:  # 调整阈值
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    
    # def _check_batch_data(self, batch, batch_idx, context):
    #     """检查batch数据"""
    #     def check_tensor(tensor, name):
    #         if torch.is_tensor(tensor):
    #             has_nan = torch.isnan(tensor).any()
    #             has_inf = torch.isinf(tensor).any()
    #             if has_nan or has_inf:
    #                 print(f"🚨 {context} - Batch {batch_idx} - {name}: "
    #                       f"NaN={has_nan}, Inf={has_inf}")
    #             # 检查极端值
    #             if tensor.numel() > 0:
    #                 max_val = tensor.max().item()
    #                 min_val = tensor.min().item()
    #                 if abs(max_val) > 1e6 or abs(min_val) > 1e6:
    #                     print(f"⚠️  {context} - Batch {batch_idx} - {name}: "
    #                           f"极端值 range=({min_val:.6f}, {max_val:.6f})")
        
    #     if torch.is_tensor(batch):
    #         check_tensor(batch, "数据")
    #     elif isinstance(batch, (list, tuple)):
    #         for i, item in enumerate(batch):
    #             check_tensor(item, f"item_{i}")
    #     elif isinstance(batch, dict):
    #         for key, value in batch.items():
    #             check_tensor(value, f"key_{key}")
    
    # def _analyze_problematic_batch(self):
    #     """分析导致问题的batch"""
    #     if not hasattr(self, 'current_batch') or self.current_batch is None:
    #         return
            
    #     batch = self.current_batch
    #     batch_idx = getattr(self, 'current_batch_idx', 'unknown')
        
    #     print(f"\n🔍 分析问题Batch {batch_idx}:")
        
    #     if torch.is_tensor(batch):
    #         self._print_tensor_stats(batch, "数据")
    #     elif isinstance(batch, (list, tuple)):
    #         for i, item in enumerate(batch):
    #             if torch.is_tensor(item):
    #                 self._print_tensor_stats(item, f"输入[{i}]")
    #     elif isinstance(batch, dict):
    #         for key, value in batch.items():
    #             if torch.is_tensor(value):
    #                 self._print_tensor_stats(value, f"输入[{key}]")
    
    # def _print_tensor_stats(self, tensor, name):
        # """打印张量统计信息"""
        # print(f"  {name}: shape={tuple(tensor.shape)}")
        # print(f"    范围: [{tensor.min().item():.8f}, {tensor.max().item():.8f}]")
        # print(f"    均值: {tensor.mean().item():.8f} ± {tensor.std().item():.8f}")
        
        # # 检查数值分布
        # if tensor.numel() > 0:
        #     abs_tensor = tensor.abs()
        #     large_vals = (abs_tensor > 1000).sum().item()
        #     if large_vals > 0:
        #         print(f"    ⚠️  有 {large_vals} 个绝对值大于1000的值")
    
    def test_step(self, data, batch_idx) -> None:
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            out = self(cur_data)
            all_outs.append(out)
        self.submission_handler.format_data(data[-1], all_outs[-1]['y_hat'], all_outs[-1]['pi'])

    def update_state_one(
        self,
        state, 
        predict,
        n, 
        next_state=None,
        dt=0.1 
    ):
        cur_data = state.copy()
        last_origin = cur_data['origin'] # (B, 1, 2)
        last_theta = cur_data['theta']   # (B, 1)

        # 构造旋转矩阵的逆 (用于从局部转全局)
        cos_theta = torch.cos(last_theta)
        sin_theta = torch.sin(last_theta)
        # 注意这里是转置/逆
        row1 = torch.stack([cos_theta, sin_theta], dim=1)    # (B, 2)
        row2 = torch.stack([-sin_theta, cos_theta], dim=1)   # (B, 2)
        rotate_mat_inv = torch.stack([row1, row2], dim=1)  # (B, 2, 2)
        cur_data['x_valid_mask'] = torch.cat([cur_data['x_valid_mask'][:,:,  self.n_step:], cur_data['x_valid_mask'][:,:, -self.n_step:]], dim=2)
        
        old_position = torch.cat([cur_data['x_positions'], predict + cur_data['x_centers'].unsqueeze(-2)], dim=2)
        raw_pred_position = torch.matmul(old_position, rotate_mat_inv.unsqueeze(1)) + last_origin.unsqueeze(1).unsqueeze(1)
        predict_position_diff = raw_pred_position[:, :, 1:] - raw_pred_position[:, :,:-1]
        predict_diff_norm = torch.norm(predict_position_diff, dim=-1)
        predict_velocity = predict_diff_norm / dt
        cur_data['x_velocity'] = torch.cat([cur_data['x_velocity'][:, :, self.n_step:], predict_velocity[:, :, -self.n_step:]], dim=2)
        x_vel = cur_data['x_velocity'][..., -(self.n_step+1):]
        x_vel_diff = x_vel[:, :, 1:] - x_vel[:, :, :-1]
        cur_data['x_velocity_diff'] = torch.cat([cur_data['x_velocity_diff'][:, :, self.n_step:], x_vel_diff], dim=2)
        pre_headings = torch.arctan2(predict_position_diff[:, :, -self.n_step:, 1], predict_position_diff[:, :, -self.n_step:, 0])
        cur_data['x_angles'] = torch.cat([cur_data['x_angles'][:, :, self.n_step:], pre_headings], dim=2)

        origin = raw_pred_position[:, 0, -1]
        theta = cur_data['x_angles'][:, 0, -1]

        cur_data['origin'] = origin
        cur_data['theta'] = theta
        cos_theta_new = torch.cos(theta)
        sin_theta_new = torch.sin(theta)
        row1_new = torch.stack([cos_theta_new, -sin_theta_new], dim=1) # (B, 2)
        row2_new = torch.stack([sin_theta_new, cos_theta_new], dim=1)  # (B, 2)
        rotate_mat = torch.stack([row1_new, row2_new], dim=1) # (B, 2, 2)
        new_position = torch.matmul(raw_pred_position - origin.unsqueeze(1).unsqueeze(1), rotate_mat.unsqueeze(1))
        cur_data['x_positions'] = new_position[:, :, self.n_step:]
        pos_ctr = new_position[:, :, -1]
        cur_data['x_centers'] = pos_ctr
        pos_diff_update = new_position[:, :, self.n_step:] - new_position[:, :, self.n_step-1:-1]
        cur_data['x_positions_diff'][:, :, 1:] = pos_diff_update[:, :, 1:]
        cur_data['timestamp'] = torch.ones((cur_data['x_positions'].shape[0]), device=cur_data['x_positions'].device) * (n * self.n_step + self.history_frames) * 0.1
        
        if next_state != None:
            cur_data['target'] = next_state['target']
            cur_data['target_mask'] = next_state['target_mask']
            cur_data['intent'] = next_state['intent']
        
        return cur_data
    def update_state_one_with_agent_alignment(
        self,
        state, 
        predict,
        n, 
        next_state = None,
        dt: float = 0.1 
    ):
        """
        自回归更新函数，增加了基于agent_indices的智能体对齐和状态合并逻辑。

        Args:
            state (Dict): 当前的批次状态字典。
            predict (torch.Tensor): 模型对当前批次中智能体的预测，
                                    形状为 (B, K_old, self.n_step, 2)。
            n (int): 当前是第 n 次自回归。
            next_state (Optional[Dict]): 下一个时间窗口的批次数据字典。
                                        如果提供了，将用于更新agent集合。
            dt (float): 时间步长。

        Returns:
            Dict: 更新后的批次状态字典。
        """
        
        # --------------------------------------------------------------------------
        # 步骤 1: 像之前一样，对当前批次中的所有智能体进行物理状态更新 (局部 -> 全局 -> 新局部)
        # --------------------------------------------------------------------------
        # last_origin/theta 形状: (B, 2) / (B,)
        last_origin = state['origin'] 
        last_theta = state['theta']
        batch_size = last_origin.shape[0]
        history_len = state['x_positions'].shape[2]

        cos_theta = torch.cos(last_theta)
        sin_theta = torch.sin(last_theta)
        row1 = torch.stack([cos_theta, sin_theta], dim=1)
        row2 = torch.stack([-sin_theta, cos_theta], dim=1)
        rotate_mat_inv = torch.stack([row1, row2], dim=1)  # (B, 2, 2)

        updated_valid_mask = torch.cat([
            state['x_valid_mask'][:, :, self.n_step:], 
            torch.ones_like(state['x_valid_mask'][:, :, -self.n_step:])
        ], dim=2)
        
        predict_positions = predict + state['x_centers'].unsqueeze(-2)
        old_and_new_local_pos = torch.cat([state['x_positions'], predict_positions], dim=2)

        raw_pred_position = torch.matmul(
            old_and_new_local_pos, rotate_mat_inv.unsqueeze(1)
        ) + last_origin.unsqueeze(1).unsqueeze(1)

        predict_position_diff = raw_pred_position[:, :, 1:] - raw_pred_position[:, :, :-1]
        predict_diff_norm = torch.norm(predict_position_diff, p=2, dim=-1)
        predict_velocity = predict_diff_norm / dt
        
        updated_velocity = torch.cat([
            state['x_velocity'][:, :, self.n_step:], 
            predict_velocity[:, :, -self.n_step:]
        ], dim=2)
        
        x_vel = updated_velocity[..., -(self.n_step + 1):]
        x_vel_diff = x_vel[:, :, 1:] - x_vel[:, :, :-1]
        updated_velocity_diff = torch.cat([state['x_velocity_diff'][:, :, self.n_step:], x_vel_diff], dim=2)
        
        pre_headings = torch.atan2(predict_position_diff[:, :, -self.n_step:, 1], 
                                predict_position_diff[:, :, -self.n_step:, 0])
        updated_angles = torch.cat([state['x_angles'][:, :, self.n_step:], pre_headings], dim=2)

        new_origin = raw_pred_position[:, 0, -1]
        new_theta = updated_angles[:, 0, -1]

        cos_theta_new = torch.cos(new_theta)
        sin_theta_new = torch.sin(new_theta)
        row1_new = torch.stack([cos_theta_new, -sin_theta_new], dim=1)
        row2_new = torch.stack([sin_theta_new, cos_theta_new], dim=1)
        rotate_mat = torch.stack([row1_new, row2_new], dim=1)

        new_local_position = torch.matmul(
            raw_pred_position - new_origin.unsqueeze(1).unsqueeze(1),
            rotate_mat.unsqueeze(1)
        )
        
        updated_positions = new_local_position[:, :, -history_len:]
        updated_centers = new_local_position[:, :, -1]
        pos_diff_update = new_local_position[:, :, 1:] - new_local_position[:, :, :-1]
        updated_positions_diff = pos_diff_update[:, :, -history_len:]
        
        # --------------------------------------------------------------------------
        # 步骤 2: 如果没有 next_state，直接用更新后的物理状态构建并返回结果
        # --------------------------------------------------------------------------
        if next_state is None:
            final_state = state.copy()
            final_state.update({
                'origin': new_origin, 'theta': new_theta,
                'x_valid_mask': updated_valid_mask, 'x_velocity': updated_velocity,
                'x_velocity_diff': updated_velocity_diff, 'x_angles': updated_angles,
                'x_positions': updated_positions, 'x_centers': updated_centers,
                'x_positions_diff': updated_positions_diff,
                'timestamp': torch.ones(batch_size, device=predict.device) * (n * self.n_step + self.history_frames) * 0.1
            })
            return final_state

        # --------------------------------------------------------------------------
        # 步骤 3: 核心逻辑 - 逐个场景进行智能体对齐和状态合并
        # --------------------------------------------------------------------------
        old_agent_indices_list = state['agent_indices']
        new_agent_indices_list = next_state['agent_indices']
        
        # --- 【修正】只定义需要我们手动更新和对齐的 agent-wise 键 ---
        # 移除了 'target', 'target_mask', 'intent'
        agent_wise_keys_to_align = [
            'x_positions', 'x_centers', 'x_positions_diff', 'x_angles',
            'x_velocity', 'x_velocity_diff', 'x_valid_mask', 'x_attr',
        ]
        # agent_wise_keys_to_align = [
        #     'x_positions', 'x_velocity_diff'
        # ]

        # 收集最终批次数据的列表
        final_tensors_list = {key: [] for key in agent_wise_keys_to_align}
        
        for i in range(batch_size):
            # --- 为当前场景 (i) 准备数据和ID映射 ---
            old_ids = old_agent_indices_list[i]
            new_ids = new_agent_indices_list[i]
            old_id_to_idx = {id_val.item(): idx for idx, id_val in enumerate(old_ids)}
            
            # 收集当前场景最终状态的行
            rows_to_stack = {key: [] for key in agent_wise_keys_to_align}

            # --- 遍历 next_state 中的所有智能体 ---
            for new_idx, agent_id_val in enumerate(new_ids):
                agent_id = agent_id_val.item()
                
                if agent_id in old_id_to_idx:
                    # --- Case 1: 公共智能体 -> 使用自回归更新的状态 ---
                    old_idx = old_id_to_idx[agent_id]
                    
                    rows_to_stack['x_positions'].append(updated_positions[i, old_idx])
                    rows_to_stack['x_centers'].append(updated_centers[i, old_idx])
                    rows_to_stack['x_positions_diff'].append(updated_positions_diff[i, old_idx])
                    rows_to_stack['x_angles'].append(updated_angles[i, old_idx])
                    rows_to_stack['x_velocity'].append(updated_velocity[i, old_idx])
                    rows_to_stack['x_velocity_diff'].append(updated_velocity_diff[i, old_idx])
                    rows_to_stack['x_valid_mask'].append(updated_valid_mask[i, old_idx])
                    rows_to_stack['x_attr'].append(state['x_attr'][i, old_idx])
                    
                else:
                    # --- Case 2: 新出现的智能体 -> 直接复制 next_state 的状态 ---
                    for key in agent_wise_keys_to_align:
                        # 从 next_state 中复制对应的行
                        rows_to_stack[key].append(next_state[key][i, new_idx])
            
            # 将收集到的行堆叠起来
            for key, rows in rows_to_stack.items():
                if rows:
                    final_tensors_list[key].append(torch.stack(rows, dim=0))

        # --- 重新打包为最终的批次状态字典 ---
        final_state_padded = {}
        
        # 使用 pad_sequence 对齐 agent 维度 (只处理我们对齐过的键)
        for key, tensor_list in final_tensors_list.items():
            if tensor_list:
                is_mask = 'mask' in key
                padding_value = False if is_mask else 0.0
                final_state_padded[key] = pad_sequence(
                    tensor_list, batch_first=True, padding_value=padding_value
                )
        
        # --- 【核心修改】直接从 next_state 继承 target, target_mask, 和 intent ---
        final_state_padded['target'] = next_state['target']
        final_state_padded['target_mask'] = next_state['target_mask']
        final_state_padded['intent'] = next_state['intent']
        # final_state_padded['x_centers'] = next_state['x_centers']
        # final_state_padded['x_positions_diff'] = next_state['x_positions_diff']
        # final_state_padded['x_angles'] = next_state['x_angles']
        # final_state_padded['x_velocity'] = next_state['x_velocity']
        # # final_state_padded['x_velocity_diff'] = next_state['x_velocity_diff']
        # final_state_padded['x_valid_mask'] = next_state['x_valid_mask']
        # final_state_padded['x_attr'] = next_state['x_attr']

        # --- 处理场景级别的元数据 ---
        final_state_padded['origin'] = new_origin
        final_state_padded['theta'] = new_theta
        final_state_padded['timestamp'] = torch.ones(
            batch_size, device=predict.device
        ) * (n * self.n_step + self.history_frames) * 0.1
        
        # 从 next_state 继承其他所有非 agent-wise 的数据 (包括地图信息)
        for key, value in next_state.items():
            # 只复制那些我们还没有处理过的键
            if key not in final_state_padded:
                final_state_padded[key] = value
                
        # 计算新的 key_valid_mask
        if 'x_valid_mask' in final_state_padded:
            final_state_padded['x_key_valid_mask'] = final_state_padded['x_valid_mask'].any(-1)

        return final_state_padded