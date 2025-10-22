# accuracy.py
import torch
from torchmetrics import Metric

class CustomAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # 添加状态变量来跟踪正确预测和总样本数
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: dict, target: torch.Tensor = None):
        # 处理两种情况：
        # 1. 如果target为None，则假定target包含在preds中
        # 2. 如果target不为None，则直接使用
        
        if isinstance(preds, dict):
            # 从预测字典中提取意图预测
            predicted_classes = preds['intent']  # [B, 6, 1]
            target = preds['intent_target']
            
        # 确保target是正确的形状
        if target.dim() > 1:
            target = target.squeeze(-1)  # [B]
        
        # 计算正确的预测数量
        correct = (predicted_classes == target).sum()
        total = target.numel()
        
        # 更新状态
        self.correct += correct
        self.total += total

    def compute(self):
        # 返回累计的准确率
        return self.correct.float() / self.total if self.total > 0 else torch.tensor(0.0)