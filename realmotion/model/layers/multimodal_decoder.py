import math
import numpy as np
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

class Bezier_decoder(MultimodalDecoder):
    def __init__(self,bezier_points=5, **kwargs):
        super().__init__(**kwargs)
        self.bezier_points = bezier_points
        self.loc = nn.Sequential(
            nn.Linear(kwargs['embed_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, kwargs['embed_dim']),
            nn.ReLU(),
            nn.Linear(kwargs['embed_dim'], self.bezier_points * 2),
        )
        # self.loc = GMMPredictor(self.bezier_points)



    def bezier_curve_from_control_points_torch(self, control_points, num_points=50):
        """
        从贝塞尔控制点批量生成轨迹点（支持任意阶贝塞尔曲线）。
        
        Args:
            control_points (Tensor): Shape [B, K, N_ctrl, D]
                - B: batch size
                - K: number of modes (e.g., MDN components)
                - N_ctrl: number of control points (n+1 for degree-n Bezier)
                - D: spatial dimension (e.g., 2 for (x, y))
            num_points (int): number of trajectory points to sample (default: 50)
        
        Returns:
            trajectory (Tensor): Shape [B, K, num_points, D]
        """
        B, K, N_ctrl, D = control_points.shape
        device = control_points.device
        dtype = control_points.dtype
        
        n = N_ctrl - 1  # Bezier degree
        
        # Precompute binomial coefficients: C(n, i) for i = 0..n
        # Use log to avoid overflow for large n (though n is usually small, e.g., 3~5)
        log_binom = torch.lgamma(torch.tensor(n + 1, dtype=dtype, device=device)) \
                    - torch.lgamma(torch.arange(n + 1, dtype=dtype, device=device) + 1) \
                    - torch.lgamma(torch.tensor(n + 1, dtype=dtype, device=device) - torch.arange(n + 1, dtype=dtype, device=device))
        binom_coeffs = torch.exp(log_binom)  # shape: [n+1]

        # Sample t in [0, 1]
        t = torch.linspace(0.0, 1.0, num_points, device=device, dtype=dtype)  # [num_points]
        
        # Expand t for broadcasting: [num_points, n+1]
        i = torch.arange(n + 1, dtype=dtype, device=device)  # [n+1]
        t = t.unsqueeze(1)  # [num_points, 1]
        
        # Compute Bernstein basis: B_i^n(t) = C(n,i) * (1-t)^(n-i) * t^i
        # Use log-space for numerical stability (optional but safe)
        log_bernstein = (
            torch.log(binom_coeffs).unsqueeze(0) +               # [1, n+1]
            (n - i) * torch.log(1 - t + 1e-8) +                  # [num_points, n+1]
            i * torch.log(t + 1e-8)                              # [num_points, n+1]
        )
        bernstein = torch.exp(log_bernstein)  # [num_points, n+1]
        
        # Normalize to mitigate log-sum-exp errors (optional but recommended)
        bernstein = bernstein / bernstein.sum(dim=1, keepdim=True)
        
        # Now compute: trajectory = bernstein @ control_points
        # bernstein: [num_points, n+1]
        # control_points: [B, K, n+1, D]
        # We want: [B, K, num_points, D]
        
        # Reshape control_points to [B*K, n+1, D]
        ctrl_flat = control_points.view(B * K, N_ctrl, D)
        
        # Matrix multiply: [num_points, n+1] @ [B*K, n+1, D] -> [B*K, num_points, D]
        traj_flat = torch.bmm(bernstein.unsqueeze(0).expand(B * K, -1, -1), ctrl_flat)
        
        # Reshape back to [B, K, num_points, D]
        trajectory = traj_flat.view(B, K, num_points, D)
        
        return trajectory
    
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B, -1) 
        aggregated_x = self.aggregation_layer(x_flat) # 输出形状: (B, D), 例如 (48, 128)
        x = self.multimodal_proj(aggregated_x).view(-1, 6, self.embed_dim)
        remaining_control_points  = self.loc(x).view(-1, 6, self.bezier_points, 2)
        origin_points = torch.zeros(B, 6, 1, 2, device=remaining_control_points.device, 
                            dtype=remaining_control_points.dtype)
        loc = torch.cat([origin_points, remaining_control_points], dim=2)  # [B, 6, bezier_points, 2]
        
        pred_p = self.bezier_curve_from_control_points_torch(loc, self.future_steps)
        if self.return_prob:
            pi = self.pi(x).squeeze(-1)
            probs = F.softmax(pi, dim=-1)
        else:
            pi = None

        
        return {"y_hat": pred_p, # (B, top_k, T, 2) -> Top-K的轨迹
                "logits": pi, 
                "probs": probs,
                 "mode": x }


class Multi_Bezier_decoder(MultimodalDecoder):
    """
    分段贝塞尔曲线解码器
    预测多段连接的贝塞尔曲线，并采样成轨迹点。
    """
    def __init__(self, embed_dim, future_steps, num_input_tokens=5, return_prob=True, 
                 num_bezier_segments=3, bezier_points=3):
        """
        Args:
            num_bezier_segments: 轨迹被切分成几段贝塞尔曲线 (建议 3)
            bezier_degree: 每一段曲线的阶数 (建议 3, 即 Cubic Bezier)
        """
        super().__init__(embed_dim, future_steps, num_input_tokens, return_prob)
        
        self.num_bezier_segments = num_bezier_segments
        self.bezier_degree = bezier_points
        
        # 每一段我们需要预测 degree 个点 (因为起点固定为上一段终点)
        # 例如 3阶曲线需要4个点(P0, P1, P2, P3)。
        # P0 是已知的，网络需要预测 P1, P2, P3。
        points_per_segment = self.bezier_degree
        total_ctrl_points = self.num_bezier_segments * points_per_segment

        # 覆盖父类的 self.loc
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            # 输出维度: [6, segments * degree * 2]
            nn.Linear(embed_dim, total_ctrl_points * 2),
        )

    def get_bernstein_matrix(self, n, num_points, device):
        """
        生成伯恩斯坦基矩阵 (缓存友好型)
        Returns: [num_points, n+1]
        """
        # 计算二项式系数
        c_n = torch.tensor([1.0], device=device)
        for i in range(1, n + 1):
            c_n = torch.cat([c_n, torch.tensor([c_n[-1] * (n - i + 1) / i], device=device)])
        
        t = torch.linspace(0.0, 1.0, num_points, device=device).unsqueeze(1) # [T, 1]
        i = torch.arange(n + 1, device=device).float().unsqueeze(0) # [1, n+1]
        
        # Bernstein basis: C(n,i) * t^i * (1-t)^(n-i)
        basis = c_n * torch.pow(t, i) * torch.pow(1 - t, n - i)
        return basis

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B, -1) 
        
        # 1. 聚合与多模态投影
        aggregated_x = self.aggregation_layer(x_flat) 
        x_modes = self.multimodal_proj(aggregated_x).view(-1, 6, self.embed_dim) # [B, 6, D]
        
        # 2. 预测控制点偏移量 (Relative Offsets)
        # Shape: [B, 6, Segments, Degree, 2]
        pred_offsets = self.loc(x_modes).view(B, 6, self.num_bezier_segments, self.bezier_degree, 2)
        
        # 3. 构建绝对控制点 (核心逻辑：C0 连续拼接)
        # 每一段的位移向量 (Displacement Vector) 是该段最后一个预测点
        # pred_offsets[:, :, :, -1, :] 代表每一段的终点相对于该段起点的偏移
        segment_displacements = pred_offsets[..., -1, :] # [B, 6, Seg, 2]
        
        # 计算每一段的绝对起点 (Accumulated Origins)
        # 第0段起点是(0,0)，第1段起点是第0段的终点...
        accumulated_displacements = torch.cumsum(segment_displacements, dim=2)
        # 在时间维度前面补一个 (0,0)
        zeros = torch.zeros(B, 6, 1, 2, device=x.device, dtype=x.dtype)
        segment_origins = torch.cat([zeros, accumulated_displacements[..., :-1, :]], dim=2) # [B, 6, Seg, 2]
        
        # 将起点广播并加到预测的偏移量上
        # segment_origins: [B, 6, Seg, 1, 2]
        # pred_offsets:    [B, 6, Seg, Degree, 2]
        # full_segment_ctrls (P1...P3): [B, 6, Seg, Degree, 2]
        full_segment_ctrls = segment_origins.unsqueeze(3) + pred_offsets
        
        # 把起点 P0 拼接到每一段的控制点列表中
        # specific_P0 (P0): [B, 6, Seg, 1, 2]
        specific_P0 = segment_origins.unsqueeze(3)
        # final_ctrl_points: [B, 6, Seg, Degree+1, 2] -> 每一段完整的4个点
        final_ctrl_points = torch.cat([specific_P0, full_segment_ctrls], dim=3)
        
        # 4. 贝塞尔采样
        # 确定每一段需要采样多少个点
        # 为了保证总点数 = future_steps，我们可能需要处理除不尽的情况
        # 策略：每段采样 ceil(T/S) 个点，拼接后截取前 T 个点
        points_per_seg = (self.future_steps + self.num_bezier_segments - 1) // self.num_bezier_segments + 1
        # +1 是为了保证拼接时去掉重复点后仍然够长
        
        # 准备矩阵乘法
        # Reshape to [Batch_Total, Degree+1, 2]
        flat_ctrls = final_ctrl_points.view(-1, self.bezier_degree + 1, 2)
        
        # 获取基矩阵 [points, degree+1]
        basis_matrix = self.get_bernstein_matrix(self.bezier_degree, points_per_seg, x.device)
        
        # 矩阵乘法生成轨迹: [Batch_Total, points, 2]
        flat_traj_segments = torch.matmul(basis_matrix.unsqueeze(0), flat_ctrls)
        
        # 5. 拼接轨迹并去重
        # Reshape back: [B, 6, Seg, points, 2]
        traj_segments = flat_traj_segments.view(B, 6, self.num_bezier_segments, points_per_seg, 2)
        
        # 拼接：除了最后一段，前面的段都去掉最后一个点 (因为 Seg[i]的终点 == Seg[i+1]的起点)
        trajs_list = []
        for i in range(self.num_bezier_segments):
            seg = traj_segments[:, :, i, :, :]
            if i < self.num_bezier_segments - 1:
                trajs_list.append(seg[:, :, :-1, :])
            else:
                trajs_list.append(seg)
                
        full_traj = torch.cat(trajs_list, dim=2) # [B, 6, Total_T, 2]
        
        # 截取最终长度
        pred_loc = full_traj[:, :, :self.future_steps, :]

        # 6. 计算概率 (逻辑与父类一致)
        if self.return_prob:
            pi_logits = self.pi(x_modes).squeeze(-1) # [B, 6]
            probs = F.softmax(pi_logits, dim=-1)
        else:
            pi_logits = None
            probs = None

        return {
            "y_hat": pred_loc, # (B, 6, T, 2)
            "logits": pi_logits, 
            "pi": probs,
            "mode": x_modes,
            # debug用: 返回控制点可以方便可视化
            "control_points": final_ctrl_points 
        }


class LearnableTime_Bezier_Decoder(MultimodalDecoder):
    """
    方案一：分段贝塞尔 + C1连续性约束 + 可学习时间比例
    """
    def __init__(self, embed_dim, future_steps, num_input_tokens=5, return_prob=True, 
                 num_bezier_segments=3, bezier_degree=3):
        super().__init__(embed_dim, future_steps, num_input_tokens, return_prob)
        
        self.num_segments = num_bezier_segments
        self.degree = bezier_degree
        self.future_steps = future_steps # e.g. 60 frames
        
        # -----------------------------------------------------------
        # 定义输出维度
        # 1. 第一段: 需要预测 P1, P2, P3 (3个点)
        # 2. 后续段: 为了保证 C1 连续，P1 是受限的(只需预测长度标量), 
        #    只需要预测 P2, P3 (2个点)
        #    因此自由度 = 3 + (num_segments-1) * 2
        
        # -----------------------------------------------------------
        num_geo_params = self.degree * 2 + (1 + (self.degree - 1) * 2 ) * (self.num_segments - 1)
        # 如果 degree=3, seg=3, param = 2 * (3 + 2*2) = 14
        
        # 时间参数: 每一段的时间比例 logits (num_segments 个标量)
        num_time_params = self.num_segments
        
        self.total_params = num_geo_params + num_time_params

        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.total_params),
        )


    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B, -1)
        
        aggregated_x = self.aggregation_layer(x_flat)
        x_modes = self.multimodal_proj(aggregated_x).view(-1, 6, self.embed_dim)
        
        # 预测所有参数
        preds = self.loc(x_modes) # [B, 6, total_params]
        
        # --- 1. 解析时间参数 ---
        # 提取最后 num_segments 个值作为时间 logits
        time_logits = preds[..., -self.num_segments:] # [B, 6, Seg]
        # Softmax 归一化得到每一段的时间比例
        time_props = F.softmax(time_logits, dim=-1) # [B, 6, Seg]
        # 计算每一段的时间长度 (单位: 帧数)
        seg_durations = time_props * self.future_steps
        # 计算时间断点 (Cutoff times): t_0=0, t_1, t_2...
        cutoffs = torch.cumsum(seg_durations, dim=-1) # [B, 6, Seg]
        cutoffs = torch.cat([torch.zeros_like(cutoffs[..., :1]), cutoffs], dim=-1) # [B, 6, Seg+1]

        # --- 2. 解析几何控制点 (关键：C1 约束) ---
        geo_params = preds[..., :-self.num_segments] 
        # 我们需要逐步构建控制点列表
        # list of [B, 6, degree+1, 2]
        ctrl_points_list = [] 
        
        # 指针，用于从 geo_params 中切片取值
        ptr = 0
        
        # 起点 P0 始终是 (0,0)
        current_end_pos = torch.zeros(B, 6, 2, device=x.device)
        # 记录上一段末端的切线向量 (P_n - P_{n-1})
        prev_tangent = None 
        
        for i in range(self.num_segments):
            # 当前段的 P0 等于上一段的 P_n
            p0 = current_end_pos.unsqueeze(2) # [B, 6, 1, 2]
            
            if i == 0:
                # 第一段：完全自由预测 P1, P2... Pn
                # 取出 degree 个点 (e.g. 3个: P1, P2, P3)
                num_pts = self.degree
                pts_flat = geo_params[..., ptr : ptr + num_pts*2]
                ptr += num_pts * 2
                
                # 预测的是相对位移，累加得到绝对坐标
                # 这里简化处理：预测的是相对于 P0 的绝对坐标 (也可以做成累加offset)
                pts_rel = pts_flat.view(B, 6, num_pts, 2)
                pts_abs = p0 + pts_rel 
                
                segment_ctrls = torch.cat([p0, pts_abs], dim=2) # [B, 6, 4, 2]
            
            else:
                # 后续段：实施 C1 约束
                # 1. 计算 P1: 必须在 prev_tangent 方向上
                # 预测一个标量 lambda (取exp保证为正)
                lambda_val = torch.exp(geo_params[..., ptr : ptr+1]) # [B, 6, 1]
                ptr += 1
                
                # Normalize prev tangent to avoid scale issues
                tangent_dir = F.normalize(prev_tangent, dim=-1) 
                p1 = p0.squeeze(2) + lambda_val * tangent_dir # [B, 6, 2]
                
                # 2. 预测剩下的点 P2...Pn
                num_rem = self.degree - 1 # e.g. 2个
                pts_flat = geo_params[..., ptr : ptr + num_rem*2]
                ptr += num_rem * 2
                pts_rel = pts_flat.view(B, 6, num_rem, 2)
                
                # 这些点通常相对于 P1 做 offset 比较好学习
                pts_abs = p1.unsqueeze(2) + pts_rel 
                
                segment_ctrls = torch.cat([p0, p1.unsqueeze(2), pts_abs], dim=2)

            # 更新状态供下一段使用
            ctrl_points_list.append(segment_ctrls)
            current_end_pos = segment_ctrls[..., -1, :]
            prev_tangent = segment_ctrls[..., -1, :] - segment_ctrls[..., -2, :]
        
        # --- 3. 采样轨迹 (Soft Masking) ---
        # 构造查询时间 t: [0, 1, ..., 59]
        t_grid = torch.arange(self.future_steps, device=x.device).float() # [T]
        t_grid = t_grid.view(1, 1, -1) # [1, 1, T] 广播用
        
        final_pos = 0
        
        for i in range(self.num_segments):
            # 获取该段的时间范围 [start, end]
            t_start = cutoffs[..., i].unsqueeze(-1)   # [B, 6, 1]
            t_end = cutoffs[..., i+1].unsqueeze(-1)   # [B, 6, 1]
            
            # 计算 Soft Mask (使用 steep sigmoid 近似阶跃)
            # 这里的 beta=100 控制边缘陡峭程度
            beta = 10.0 
            mask = torch.sigmoid(beta * (t_grid - t_start)) * torch.sigmoid(beta * (t_end - t_grid))
            # 归一化 mask 使得对于每个 t，sum(mask) = 1 (近似)
            # 也可以在所有段算完后做 softmax，这里简化处理
            
            # 计算局部时间参数 u \in [0, 1]
            # 为了防止除以0，分母加个 epsilon
            duration = t_end - t_start + 1e-6
            u = (t_grid - t_start) / duration
            u = torch.clamp(u, 0.0, 1.0).unsqueeze(-1) # [B, 6, T, 1]
            
            # 计算该段贝塞尔曲线在 u 处的值
            # ctrl_points: [B, 6, 4, 2]
            # De Casteljau 或者 直接公式法。这里手写公式法 (Cubic)
            p = ctrl_points_list[i] # [B, 6, 4, 2]
            p0 = p[..., 0, :]
            p1 = p[..., 1, :]
            p2 = p[..., 2, :]
            p3 = p[..., 3, :]
            
            # (1-u)^3 P0 + 3u(1-u)^2 P1 + 3u^2(1-u) P2 + u^3 P3
            # 注意广播维度
            bezier_pos = (1-u)**3 * p0.unsqueeze(2) + \
                         3 * u * (1-u)**2 * p1.unsqueeze(2) + \
                         3 * u**2 * (1-u) * p2.unsqueeze(2) + \
                         u**3 * p3.unsqueeze(2)  # [B, 6, T, 2]
            
            # 累加到最终结果
            final_pos = final_pos + mask.unsqueeze(-1) * bezier_pos
            
        pred_loc = final_pos # [B, 6, T, 2]

        if self.return_prob:
            pi_logits = self.pi(x_modes).squeeze(-1)
            probs = F.softmax(pi_logits, dim=-1)
        else:
            pi_logits, probs = None, None

        return {
            "y_hat": pred_loc,
            "logits": pi_logits,
            "pi": probs,
            "cutoffs": cutoffs # Debug用：可以看网络学到的切换时间
        }

class BSpline_Decoder(MultimodalDecoder):
    """
    方案二：B样条曲线 (隐式分段，数学平滑)
    """
    def __init__(self, embed_dim, future_steps, num_input_tokens=5, return_prob=True, 
                 num_control_points=12, degree=3):
        """
        Args:
            num_control_points: 控制点总数 (越多越灵活，但太少会拟合不足)
                                建议: degree + segments, 比如 3阶 + 3段 ~ 6-8个点
                                为了覆盖复杂意图，AV2建议设为 10-15
            degree: B样条阶数 (建议3)
        """
        super().__init__(embed_dim, future_steps, num_input_tokens, return_prob)
        
        self.num_ctrl = num_control_points
        self.degree = degree
        
        # 预测除原点外的控制点 (num_ctrl - 1)
        self.loc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, (self.num_ctrl - 1) * 2),
        )
        
        # --- 预计算 B样条基矩阵 M ---
        # 结果 shape: [future_steps, num_ctrl]
        # 注册为 buffer，这样 device 会自动管理，且不作为参数更新
        self.register_buffer("basis_matrix", self._precompute_basis(future_steps, num_control_points, degree))

    def _precompute_basis(self, T, n, p):
        """
        使用 Cox-de Boor 递归公式计算均匀 B-Spline 基矩阵
        T: 时间步数
        n: 控制点数量
        p: 阶数 (degree)
        """
        # 1. 定义节点向量 (Knot Vector)
        # Clamped Knot Vector: 开头结尾重复 p+1 次，中间均匀分布
        # 节点总数 m = n + p + 1
        m = n + p + 1
        
        # 内部节点数
        num_inner = m - 2 * (p + 1)
        
        # 构造节点向量 u_vec: [0,0,0,0, 0.2, 0.4..., 1,1,1,1]
        inner_knots = torch.linspace(0, 1, num_inner + 2)[1:-1]
        knots = torch.cat([
            torch.zeros(p + 1),
            inner_knots,
            torch.ones(p + 1)
        ])
        
        # 2. 评估时间点 t (归一化到 [0,1])
        t = torch.linspace(0, 1, T)
        
        # 3. Cox-de Boor 递归计算基函数值 N_{i,p}(t)
        # 为了矩阵计算，我们计算所有 i 和所有 t
        
        # 初始化 0阶基函数
        # Basis shape: [m-1, T] -> 代表 N_{i,0}(t)
        basis = torch.zeros(m - 1, T)
        for i in range(m - 1):
            # N_{i,0}(t) = 1 if knots[i] <= t < knots[i+1]
            mask = (t >= knots[i]) & (t < knots[i+1])
            # 处理 t=1 的边界情况 (属于最后一个区间)
            if knots[i+1] == 1.0: 
                mask = mask | (t == 1.0)
            basis[i, mask] = 1.0
            
        # 递归 p 次
        for d in range(1, p + 1):
            new_basis = torch.zeros(m - 1 - d, T)
            for i in range(m - 1 - d):
                # Term 1
                numer1 = (t - knots[i])
                denom1 = (knots[i+d] - knots[i])
                term1 = 0.0
                if denom1 != 0:
                    term1 = (numer1 / denom1) * basis[i]
                
                # Term 2
                numer2 = (knots[i+d+1] - t)
                denom2 = (knots[i+d+1] - knots[i+1])
                term2 = 0.0
                if denom2 != 0:
                    term2 = (numer2 / denom2) * basis[i+1]
                    
                new_basis[i] = term1 + term2
            basis = new_basis
            
        # 最终 basis shape 是 [n, T]
        # 转置为 [T, n] 以便做 M * C
        return basis.t()

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B, -1)
        
        aggregated_x = self.aggregation_layer(x_flat)
        x_modes = self.multimodal_proj(aggregated_x).view(-1, 6, self.embed_dim)
        
        # 1. 预测控制点 (偏移量)
        pred_offsets = self.loc(x_modes).view(B, 6, self.num_ctrl - 1, 2)
        
        # 2. 构造完整控制点矩阵
        # P0 固定为 (0,0)
        zeros = torch.zeros(B, 6, 1, 2, device=x.device)
        ctrl_points = torch.cat([zeros, pred_offsets], dim=2) # [B, 6, num_ctrl, 2]
        
        # 3. 生成轨迹 (矩阵乘法)
        # Basis Matrix: [T, num_ctrl]
        # Ctrl Points:  [B, 6, num_ctrl, 2]
        # Result:       [B, 6, T, 2]
        # 这里的 matmul 需要广播
        # basis: [1, 1, T, N]
        basis = self.basis_matrix.unsqueeze(0).unsqueeze(0) 
        
        # ctrl: [B, 6, N, 2] -> 也可以看作 [... N, 2]
        # [1, 1, T, N] x [B, 6, N, 2] -> [B, 6, T, 2]
        # PyTorch matmul 会自动广播前面的维度，只要最后两维匹配
        pred_loc = torch.matmul(basis, ctrl_points) 
        
        if self.return_prob:
            pi_logits = self.pi(x_modes).squeeze(-1)
            probs = F.softmax(pi_logits, dim=-1)
        else:
            pi_logits, probs = None, None

        return {
            "y_hat": pred_loc,
            "logits": pi_logits, 
            "pi": probs,
            # "control_points": ctrl_points
        }

class Cross_BSpline_Decoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        future_steps=60,      # T
        num_modes = 6,     
        num_heads = 8,
        mlp_ratio = 4.0,
        qkv_bias = False,
        drop = 0.2,
        attn_drop = 0.2,
        drop_path= 0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        query_cross_layers=2,
        num_control_points=12, 
        degree=3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes

        # 融合权重（可学习）
        self.fusion_w = nn.Parameter(torch.ones(3))

        self.mode_queries = nn.Parameter(torch.randn( self.num_modes, self.embed_dim))
        self.query_mode =nn.ModuleList(Inter_cross_self_Block(
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
        self.predictor = BSpline_GMM_Predictor(future_len=future_steps, dim=embed_dim, num_control_points=num_control_points, degree=degree)
 
    def forward(
        self,
        context: torch.Tensor,      # [B, N, D]
        segment: int,   # [B, 2]
        key_padding_mask=None,
        lane_mask=None,
    ):
        """
        Returns:
            trajectories: [B, M, T, 2]  # M 条未来轨迹
            confidences:  [B, M]       # 每条轨迹的未归一化置信度（logits）
        """
        B = context.shape[0]
        # self.visualize_intent_table(self.intent_bank)

        # --- 步骤 1: 初始化 K 个模态的“思考状态” ---
        mode = self.mode_queries.expand(B, -1, -1) # (B, K, D)
        
        for blk in self.query_mode:
            mode = blk(src=mode, src_kv=context, key_padding_mask=key_padding_mask)
        y_hat, pi, scal = self.predictor(mode)

        return {
        # "weights": weights,
        "mode": mode, 
        "y_hat": y_hat,      # [B, M, T, 2]
        "pi": pi,              # [B, M]
        "scal": scal,             # [B, M, T, 2]
    }

class MLPExpert(nn.Module):
    def __init__(self, d_model, prediction_horizon, drop=0.0, num_heads=8,
                 mlp_ratio=4.0,moe_drop=0.0, qkv_bias=False, attn_drop=0.0, drop_path=0.0,
                 query_cross_layers=2,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.prediction_horizon = prediction_horizon
        hidden_dim = d_model * 2 

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(moe_drop),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(moe_drop),
            nn.Linear(hidden_dim, prediction_horizon * 2) 
        )
        self.query_map =nn.ModuleList(Inter_cross_self_Block(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                ) for i in range(0))


    def forward(self, expert_feature, map_context):

        # (Batch, d_model) -> (Batch, prediction_horizon * 2)
        for blk in self.query_map:
            expert_feature = blk(src=expert_feature, src_kv=map_context)
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

class IntentPlanner(nn.Module):
    """
    高层规划器：只负责生成 K 个候选的未来意图序列。
    """
    def __init__(self, embed_dim, num_modes, num_segments, num_experts, 
                    query_cross_layers, num_heads, mlp_ratio,qkv_bias,
                    drop,attn_drop,drop_path,act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_modes = num_modes
        self.num_segments = num_segments
        self.num_experts = num_experts

        self.mode_queries = nn.Parameter(torch.randn(1, num_modes, embed_dim))
        self.query_init = nn.ModuleList([Inter_cross_self_Block(
                                        dim=embed_dim,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        drop=drop,
                                        attn_drop=attn_drop,
                                        drop_path=drop_path,
                                        act_layer=act_layer,
                                        norm_layer=norm_layer,
                                        ) for _ in range(query_cross_layers)])
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
        self.history_norm = nn.LayerNorm(embed_dim)

        self.gru_cell = nn.GRUCell(input_size=embed_dim, hidden_size=embed_dim)
        self.gating_network = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, num_experts))
        
        # 一个可学习的输入，启动GRU循环
        self.start_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, history_intent_embeddings, key_padding_mask=None):
        B = history_intent_embeddings.shape[0]

        # 1. 初始化 K 个模态的规划起点
        queries = self.mode_queries.expand(B, -1, -1)
        for blk in self.query_init:
            plan_state = blk(src=queries, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
        plan_state = self.history_norm(plan_state).view(B * self.num_modes, -1)

        # 2. 自回归生成意图序列
        # (B*K, D)
        plan_input = self.start_token.expand(B * self.num_modes, -1, -1).squeeze(1)
        all_intent_logits = []

        for _ in range(self.num_segments):
            plan_state = self.gru_cell(plan_input, plan_state)
            # 输出当前步骤的意图 logits
            intent_logits = self.gating_network(plan_state) # (B*K, num_experts)
            all_intent_logits.append(intent_logits)

            # 更新下一轮的输入：这里可以用上一轮的logits或者状态来生成，简化起见直接用状态
            plan_input = plan_state # 或者更复杂的，比如 nn.Linear(plan_state)

        # 3. 整理输出
        # List[(B*K, N)] -> (S, B*K, N) -> (B*K, S, N)
        stacked_logits = torch.stack(all_intent_logits, dim=0).permute(1, 0, 2)
        # -> (B, K, S, N)
        return stacked_logits.view(B, self.num_modes, self.num_segments, self.num_experts)

class TrajectoryExecutor(nn.Module):
    """
    低层执行器：接收一个确定的意图计划，生成轨迹。
    """
    def __init__(self, embed_dim, num_segments, future_steps, num_experts, top_k, **kwargs):
        super().__init__()
        self.num_segments = num_segments
        self.future_steps = future_steps
        self.top_k = top_k
        self.num_experts = num_experts
        # 只需要一个GRU来维持轨迹生成的状态
        self.gru_cell = nn.GRUCell(input_size=embed_dim, hidden_size=embed_dim)
        self.experts = nn.ModuleList([MLPExpert(embed_dim, future_steps) for _ in range(num_experts)])
        self.segment_embedder = nn.Sequential(
                                nn.Linear(future_steps * 2, embed_dim), 
                                nn.ReLU(), 
                                nn.Linear(embed_dim, embed_dim))
        
    def forward(self, initial_state, intent_plan):
        """
        Args:
            initial_state (torch.Tensor): (B, D) GRU的初始状态，可以来自历史编码
            intent_plan (torch.Tensor): (B, S, N) 一个确定的意图计划
        """
        B = initial_state.shape[0]
        exec_state = initial_state
        exec_input = torch.zeros_like(initial_state)
        
        future_segments = []
        for i in range(self.num_segments):
            exec_state = self.gru_cell(exec_input, exec_state)
            
            # 直接使用来自 planner 的意图 logits
            intent_logits_step_i = intent_plan[:, i, :] # (B, N)
            
            # 使用 MoE 执行
            segment_output = self.moe_executor(exec_state, intent_logits_step_i)
            future_segments.append(segment_output)
            
            # 更新下一轮输入
            exec_input = self.segment_embedder(segment_output)
            
        # 拼接轨迹
        stacked_segments = torch.stack(future_segments, dim=1) # (B, S, steps*2)
        trajectory = stacked_segments.view(B, self.num_segments * self.future_steps, 2)
        return trajectory

    def moe_executor(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
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

class HierarchicalDecoder(nn.Module):
    """
    顶层模块，整合规划器和执行器
    """
    def __init__(self, embed_dim,
                num_modes, 
                future_len,
                future_steps,
                num_experts,
                top_k, 
                query_cross_layers,
                num_heads = 8,
                mlp_ratio = 4.0,
                qkv_bias = False,
                drop = 0.2,
                attn_drop = 0.2,
                drop_path= 0.2,): # 传入所有需要的参数
        super().__init__()
        num_segments = future_len // future_steps
        self.num_modes= num_modes
        self.planner = IntentPlanner(embed_dim=embed_dim, 
                                    num_modes=num_modes, 
                                    num_segments=num_segments,
                                    num_experts=num_experts, 
                                    query_cross_layers=query_cross_layers,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    drop=drop,
                                    attn_drop=attn_drop,
                                    drop_path=drop_path)
        self.executor = TrajectoryExecutor(embed_dim=embed_dim, 
                                        num_segments=num_segments,
                                        future_steps=future_steps, 
                                        num_experts=num_experts,
                                        top_k=top_k)
        
        # 用于为 Executor 创建初始状态
        self.executor_init_head = nn.Linear(embed_dim, embed_dim)
        # 最终的全局概率头
        # self.prob_head = nn.Linear( * num_segments, 1) # 基于整个计划序列的特征来预测概率
        self.prob_head = nn.Sequential(
            nn.Linear(num_segments *num_experts, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )

    def forward(self, history_intent_embeddings,mode, key_padding_mask=None,lane_mask=None):
        B = history_intent_embeddings.shape[0]
        
        # 1. 高层规划：生成 K 个候选意图计划
        # (B, K, S, N)
        k_intent_plans_logits = self.planner(history_intent_embeddings, key_padding_mask)
        
        # 2. 低层执行：为每个计划生成一条轨迹
        all_predictions = []
        # (B, D) - 从历史中提炼一个统一的执行起点
        executor_initial_state = self.executor_init_head(history_intent_embeddings.mean(dim=1))

        for k in range(self.num_modes):
            # 获取第 k 个计划 (B, S, N)
            current_plan = k_intent_plans_logits[:, k, :, :]
            # 执行该计划
            trajectory_k = self.executor(executor_initial_state, current_plan)
            all_predictions.append(trajectory_k)
            
        # 3. 整理输出
        # List[(B, T, 2)] -> (B, K, T, 2)
        final_predictions = torch.stack(all_predictions, dim=1)

        # 4. 计算全局概率
        # (B, K, S, N) -> (B, K, S*N) -> (B, K, D') -> (B, K, 1) -> (B, K)
        # 用一种方式将计划的logits序列转换为特征向量
        plan_features = k_intent_plans_logits.reshape(B, self.num_modes, -1)
        # 这里需要一个更复杂的头，比如一个小型Transformer或MLP
        mode_logits = self.prob_head(plan_features).squeeze(-1) # 这是一个简化的例子

        return {
            "predictions": final_predictions,
            "pi": mode_logits,
            "intent_plans_logits": k_intent_plans_logits, # 返回计划，用于损失计算
            "states":mode_logits
        }

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
                 moe_drop =0.2,
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
            MLPExpert(d_model=embed_dim, 
                    prediction_horizon=future_steps,
                    drop=drop, 
                    num_heads=num_heads,
                    moe_drop=moe_drop,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    query_cross_layers=query_cross_layers) for _ in range(num_experts)
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


    def forward(self, history_intent_embeddings,hist_seg,key_padding_mask=None,lane_mask=None):
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
        map_context = history_intent_embeddings[:, -lane_mask.shape[1]:,:]
        hist_context = history_intent_embeddings[:, :hist_seg,:]
        map_context = torch.cat([map_context, hist_context], dim=1)
        last_endpoint = torch.zeros(B*self.num_modes, 1, 2, device=thought_state.device, dtype=thought_state.dtype)
        for i in range(self.num_segments):
            # a) 更新思考状态
            thought_state = self.gru_cell(gru_input, thought_state)
            # thought_state = self.attn_layer(gru_input, thought_state, thought_state)[0]
            # b) 规划：决定下一步意图 (路由到专家)
            thought_state_q = thought_state.view(B, self.num_modes, self.embed_dim)
            context_output = self.query_attn[i](src=thought_state_q, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
            rich_thought_state = self.context_norm(thought_state_q + context_output).squeeze(1)
            

            expert_logits = self.gating_network(rich_thought_state) # (B*K, num_experts)
            states.append(expert_logits)

            segment_output_flat = self._moe_execution(rich_thought_state.view(B * self.num_modes, self.embed_dim),map_context, expert_logits.view(B * self.num_modes, -1))
            
            future_segments.append(segment_output_flat.reshape(B * self.num_modes, self.future_steps, 2))

            # d) 更新：将生成的轨迹段编码，作为下一次GRU的输入
            gru_input = self.segment_embedder(segment_output_flat)

        # --- 步骤 5: 拼接和整理输出 ---
        # 将分段列表堆叠起来
        # List[(B*K, steps*2)] -> (S, B*K, steps*2)
        stacked_segments = torch.cat(future_segments, dim=1)
        
        # 调整形状为最终轨迹格式
        # (S, B*K, steps*2) -> (B*K, S, steps*2) -> (B*K, T, 2)
        trajectories_flat = torch.cumsum(stacked_segments, dim=1)
        
        # (B*K, T, 2) -> (B, K, T, 2)
        final_predictions = trajectories_flat.view(B, self.num_modes, self.future_len, 2)
        states = torch.stack(states, dim=2)
        # states = states.reshape(B, -1, self.embed_dim)
        return {
            "predictions": final_predictions, # (B, K, T, 2)
            "pi": mode_logits,            # (B, K)
            "segment_logits_per_mode": states,
            "states": states
        }
    def _moe_execution(self, state: torch.Tensor,map_context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
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
        B, M, D    =map_context.shape
        map_context = map_context.unsqueeze(1).repeat_interleave(self.num_modes, dim=1).reshape(-1, M, D)
        repeated_state = state.repeat_interleave(self.top_k, dim=0)
        repeated_map_context = map_context.repeat_interleave(self.top_k, dim=0)
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
                expert_inputs_map = repeated_map_context[mask]
                # 一次性地送入专家网络进行计算
                expert_outputs = self.experts[i](expert_inputs.unsqueeze(1), expert_inputs_map)
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
    
class RegressionSegmentDecoder2(nn.Module):
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
                 moe_drop =0.2,
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
            MLPExpert(d_model=embed_dim, 
                    prediction_horizon=future_steps,
                    drop=drop, 
                    num_heads=num_heads,
                    moe_drop=moe_drop,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, 
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    query_cross_layers=query_cross_layers) for _ in range(num_experts)
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


    def forward(self, history_intent_embeddings,hist_seg,key_padding_mask=None,lane_mask=None):
        """
        Args:
            history_intent_embeddings (torch.Tensor): 编码器输出的历史意图序列。
                                                      形状: (B, T_hist_segments, D)，例如 (32, 5, 128)
        """
        B = history_intent_embeddings.shape[0]

        # --- 步骤 1: 初始化 K 个模态的“思考状态” ---
        queries = self.mode_queries.expand(B, -1, -1) # (B, K, D)
        
        for blk in self.query_init:
            queries = blk(src=queries, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
        
        initial_state = self.history_norm(queries) # (B, K, D)

        # --- 步骤 2: 预测全局模态概率 ---
        # 基于对历史的初始理解，直接预测每个模态的可能性
        mode_logits = self.prob_head(initial_state).squeeze(-1) # (B, K)
        
        # --- 步骤 3: 准备自回归生成 ---
        # 将所有模态展平到一个批次中，以进行高效的并行计算
        thought_state = initial_state.view(B * self.num_modes, self.embed_dim)
        gru_input = torch.zeros_like(thought_state)
        # gru_input = history_intent_embeddings[:, :7, :].mean(dim=1).unsqueeze(1).expand(-1, self.num_modes, -1)
        gru_input = gru_input.reshape(B * self.num_modes, self.embed_dim)
        
        states = []
        # --- 步骤 4: 自回归循环 ---
        last_pose = torch.zeros(B * self.num_modes, 3, device=thought_state.device, dtype=thought_state.dtype)

        # 存储最终拼接好的、在全局坐标系下的轨迹段
        absolute_segments = []
        for i in range(self.num_segments):
            # a) 更新思考状态
            thought_state = self.gru_cell(gru_input, thought_state)
            # thought_state = self.attn_layer(gru_input, thought_state, thought_state)[0]
            # b) 规划：决定下一步意图 (路由到专家)
            thought_state_q = thought_state.view(B, self.num_modes, self.embed_dim)
            context_output = self.query_attn[i](src=thought_state_q, src_kv=history_intent_embeddings, key_padding_mask=key_padding_mask)
            rich_thought_state = self.context_norm(thought_state_q + context_output).squeeze(1)
            

            expert_logits = self.gating_network(rich_thought_state) # (B*K, num_experts)
            states.append(expert_logits)

            segment_output_flat = self._moe_execution(rich_thought_state.view(B * self.num_modes, self.embed_dim),None, expert_logits.view(B * self.num_modes, -1))
            
            local_segment = segment_output_flat.view(B * self.num_modes, self.future_steps, 2)

            # b) 准备拼接所需的姿态信息
            # last_pose[:, 0] 是 x, [:, 1] 是 y, [:, 2] 是 heading
            last_x = last_pose[:, 0].unsqueeze(1)
            last_y = last_pose[:, 1].unsqueeze(1)
            last_heading = last_pose[:, 2]

            # c) 实施旋转 + 平移
            cos_h = torch.cos(last_heading)
            sin_h = torch.sin(last_heading)

            # 旋转矩阵 (批量)
            # rot_mat 的形状是 (B*K, 2, 2)
            rot_mat = torch.stack([
                torch.stack([cos_h, -sin_h], dim=1),
                torch.stack([sin_h, cos_h], dim=1)
            ], dim=1)

            # 将局部轨迹旋转到正确的朝向
            # (B*K, Steps, 2) @ (B*K, 2, 2) -> (B*K, Steps, 2)
            # 我们需要调整维度以进行批量矩阵乘法
            # (B*K, Steps, 1, 2) @ (B*K, 1, 2, 2) -> (B*K, Steps, 1, 2)
            rotated_segment = (local_segment.unsqueeze(2) @ rot_mat.unsqueeze(1)).squeeze(2)

            # 将旋转后的轨迹平移到正确的起点
            # (B*K, Steps, 2) + (B*K, 1, 2)
            global_segment = rotated_segment + torch.stack([last_x, last_y], dim=-1)
            
            absolute_segments.append(global_segment)

            # d) 更新下一轮的 "last_pose"
            # 新的位置是这一段的最后一个点
            new_x = global_segment[:, -1, 0]
            new_y = global_segment[:, -1, 1]
            
            # 新的朝向可以通过最后两个点的位置来近似计算
            # 为避免除以零，添加一个小的epsilon
            prev_point = global_segment[:, -2, :]
            last_point = global_segment[:, -1, :]
            delta = last_point - prev_point
            new_heading = torch.atan2(delta[:, 1], delta[:, 0])

            last_pose = torch.stack([new_x, new_y, new_heading], dim=1)


        # --- 步骤 5: 拼接和整理输出 ---
        # 将分段列表堆叠起来
        # List[(B*K, steps*2)] -> (S, B*K, steps*2)
        trajectories_flat = torch.cat(absolute_segments, dim=1)
        
        # (B*K, T, 2) -> (B, K, T, 2)
        final_predictions = trajectories_flat.view(B, self.num_modes, self.future_len, 2)
        states = torch.stack(states, dim=2)
        # states = states.reshape(B, -1, self.embed_dim)
        return {
            "predictions": final_predictions, # (B, K, T, 2)
            "pi": mode_logits,            # (B, K)
            "segment_logits_per_mode": states,
            "states": states
        }
    def _moe_execution(self, state: torch.Tensor,map_context: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
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
                expert_inputs_map = None
                # 一次性地送入专家网络进行计算
                expert_outputs = self.experts[i](expert_inputs.unsqueeze(1), expert_inputs_map)
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
        

        delta_segments = initial_segments_flat.view(
            B, self.num_modes, self.num_segments, self.future_steps, 2
        )
        absolute_segments = []

        last_endpoint = torch.zeros(
            B, self.num_modes, 1, 2, 
            device=delta_segments.device, 
            dtype=delta_segments.dtype
        )


        for i in range(self.num_segments):
            # 获取当前分段的位移预测
            # shape: (B, K, T_seg, 2)
            current_delta_segment = delta_segments[:, :, i, :, :]
            
            # 将上一段的终点加到当前段的每一个位移点上，得到当前段的绝对坐标
            current_absolute_segment = current_delta_segment + last_endpoint
            
            # 将计算好的绝对坐标分段存入列表
            absolute_segments.append(current_absolute_segment)
            
            # 更新下一轮循环所需要的“上一段的终点”
            # 取当前绝对坐标分段的最后一个点作为新的终点
            # shape: (B, K, 2) -> unsqueeze -> (B, K, 1, 2)
            last_endpoint = current_absolute_segment[:, :, -1, :].unsqueeze(2)

        # 将列表中所有的绝对坐标分段在时间维度上拼接起来
        # List of (B, K, T_seg, 2) -> (B, K, T_total, 2)
        final_predictions = torch.cat(absolute_segments, dim=2)

        # --- 准备 expert_logits 以便损失函数使用 ---
        # expert_logits 形状是 (B*K*S, N)
        # reshape 成 (B, K, S, N) 以便后续处理
        segment_logits_per_mode = expert_logits.view(B, self.num_modes, self.num_segments, self.num_experts)

        return {
            "predictions": final_predictions,           # (B, K, T, 2)
            "pi": mode_logits,                          # (B, K)
            "segment_logits_per_mode": segment_logits_per_mode, # (B, K, S, N)
            "states": mode_logits
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

class GMMPredictor_dense(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor_dense, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, input):
        res = self.gaussian(input)
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        input = input.max(dim=2)[0]  
        score = self.score(input).squeeze(-1)

        return res, score, scal


class GMMPredictor(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
    
    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 2) 
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        scal = scal.view(B, M, self._future_len, 2) 
        score = self.score(input).squeeze(-1)

        return res, score, scal



class BSpline_GMM_Predictor(nn.Module):
    def __init__(self, future_len=60, dim=128, num_control_points=12, degree=3):
        """
        Args:
            future_len: 预测的时间步长 (比如 60)
            dim: 输入特征维度 (比如 128)
            num_control_points: B样条控制点数量 (建议 10-15)
            degree: B样条阶数 (建议 3)
        """
        super(BSpline_GMM_Predictor, self).__init__()
        self._future_len = future_len
        self.num_ctrl = num_control_points
        
        # --- 1. 轨迹均值预测 (改用 B样条控制点) ---
        # 原来是预测 future_len * 2，现在预测 (num_control_points - 1) * 2
        # 我们假设 P0 固定为 (0,0)，所以少预测一个点
        self.ctrl_point_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, (self.num_ctrl - 1) * 2) 
        )
        
        # --- 2. 不确定性/尺度预测 (保持 MLP) ---
        # 这一部分不需要变，依然预测每个时间步的 laplace scale
        self.scale = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len * 2)
        )
        
        # --- 3. 意图打分 (保持 MLP) ---
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )

        # --- 4. 预计算 B样条基矩阵 ---
        # shape: [future_len, num_control_points]
        basis = self._precompute_basis(future_len, num_control_points, degree)
        self.register_buffer("basis_matrix", basis)

    def _precompute_basis(self, T, n, p):
        """
        Cox-de Boor 算法预计算基矩阵 (同之前的代码)
        """
        m = n + p + 1
        num_inner = m - 2 * (p + 1)
        inner_knots = torch.linspace(0, 1, num_inner + 2)[1:-1]
        knots = torch.cat([torch.zeros(p + 1), inner_knots, torch.ones(p + 1)])
        
        t = torch.linspace(0, 1, T)
        basis = torch.zeros(m - 1, T)
        
        # 0阶
        for i in range(m - 1):
            mask = (t >= knots[i]) & (t < knots[i+1])
            if knots[i+1] == 1.0: mask = mask | (t == 1.0)
            basis[i, mask] = 1.0
            
        # 递归升阶
        for d in range(1, p + 1):
            new_basis = torch.zeros(m - 1 - d, T)
            for i in range(m - 1 - d):
                numer1 = (t - knots[i])
                denom1 = (knots[i+d] - knots[i])
                term1 = (numer1 / denom1) * basis[i] if denom1 != 0 else 0.0
                
                numer2 = (knots[i+d+1] - t)
                denom2 = (knots[i+d+1] - knots[i+1])
                term2 = (numer2 / denom2) * basis[i+1] if denom2 != 0 else 0.0
                
                new_basis[i] = term1 + term2
            basis = new_basis
            
        # 返回转置 [T, n]
        return basis.t()

    def forward(self, input):
        """
        input: [B, M, dim]  (Batch, Modes, HiddenDim)
        """
        B, M, _ = input.shape
        
        # --- A. 计算 B样条轨迹 (res) ---
        # 1. 预测控制点偏移量 [B, M, N-1, 2]
        pred_offsets = self.ctrl_point_head(input).view(B, M, self.num_ctrl - 1, 2)
        
        # 2. 拼接原点 P0=(0,0) -> [B, M, N, 2]
        zeros = torch.zeros(B, M, 1, 2, device=input.device)
        ctrl_points = torch.cat([zeros, pred_offsets], dim=2)
        
        # 3. 矩阵乘法生成轨迹
        # basis: [T, N] -> 扩充为 [1, 1, T, N]
        # ctrl_points: [B, M, N, 2]
        # result: [B, M, T, 2]
        basis = self.basis_matrix.unsqueeze(0).unsqueeze(0)
        res = torch.matmul(basis, ctrl_points) 
        
        # --- B. 计算不确定性 (scal) ---
        # 保持原来的逻辑，直接输出 [B, M, T, 2]
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        scal = scal.view(B, M, self._future_len, 2) 
        
        # --- C. 计算分数 (score) ---
        score = self.score(input).squeeze(-1) # [B, M]

        return res, score, scal
    
class VQIntentBank(nn.Module):
    def __init__(self, K, D, beta=0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(K, D))
        nn.init.uniform_(self.codebook, -1/math.sqrt(D), 1/math.sqrt(D))
        self.beta = beta
        self.K = K
    def forward(self, z):               # z: (B*M*T, D)
        # 计算距离
        d = (z.pow(2).sum(1, keepdim=True)
             + self.codebook.pow(2).sum(1)
             - 2 * z @ self.codebook.t())      # (BMT, K)
        idx = d.argmin(1)                       # (BMT,)
        z_q = F.embedding(idx, self.codebook)   # (BMT, D)
        # commitment loss
        loss = F.mse_loss(z_q.detach(), z) + \
               self.beta * F.mse_loss(z_q, z.detach())
        # straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q.view_as(z), idx.view(-1), loss

class MultiModalIntentDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_intents = 8,       # 意图表大小 K
        future_steps=60,      # T
        num_modes = 6,          # M (e.g., 6 for Argoverse2)
        output_dim= 2,
        num_heads = 8,
        mlp_ratio = 4.0,
        qkv_bias = False,
        drop = 0.2,
        attn_drop = 0.2,
        drop_path= 0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        query_cross_layers=2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.output_dim = output_dim

        # === 共享组件 ===
        # self.intent_bank = nn.Parameter(torch.randn(num_intents, embed_dim))
        # nn.init.xavier_uniform_(self.intent_bank)
        # self.intent_bank = VQIntentBank(num_intents, embed_dim)
        self.intent_bank_s = nn.Parameter(torch.randn(num_intents, embed_dim))   # 短程
        self.intent_bank_m = nn.Parameter(torch.randn(6, embed_dim))   # 中程
        self.intent_bank_l = nn.Parameter(torch.randn(4, embed_dim))   # 长程
        nn.init.xavier_uniform_(self.intent_bank_s)
        nn.init.xavier_uniform_(self.intent_bank_m)
        nn.init.xavier_uniform_(self.intent_bank_l)
        # 融合权重（可学习）
        self.fusion_w = nn.Parameter(torch.ones(3))

        self.mode_queries = nn.Parameter(torch.randn( self.num_modes, self.embed_dim))
        
        # 可学习的时间嵌入 [T, D]
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )
        self.query_mode =nn.ModuleList(Inter_cross_self_Block(
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
        self.query_intent =nn.ModuleList(Inter_cross_self_Block(
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
        self.intent_mode_querry =nn.ModuleList(Inter_cross_self_Block(
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

        self.dense_predict = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.predictor = GMMPredictor(future_steps)
        self.predictor_dense = GMMPredictor_dense(future_steps)
        # 5. 温度（可选固定或可学习）
        self.temp = 1.0  # 或设为 nn.Parameter(torch.tensor(1.0))



    def visualize_intent_table(self, intent_table, num_samples=1000):
        import torch
        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns
        """
        intent_table: 你的意图表 [num_intents, intent_dim]
        """
        # 1. 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        intent_2d = tsne.fit_transform(intent_table.detach().cpu().numpy())
        
        # 2. 绘制
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(intent_2d[:, 0], intent_2d[:, 1], 
                            c=np.arange(len(intent_table)), 
                            cmap='tab10', s=100, alpha=0.7)
        
        plt.colorbar(scatter, label='Intent ID')
        plt.title('Intent Embedding Space (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        
        # 3. 分析：计算类内距离
        distances = torch.pdist(intent_table).mean().item()
        print(f"平均意图间距离: {distances:.4f}")
        
        plt.savefig('intent_embedding.png', dpi=150)
        plt.show()

    def pyramid_intent_seq(self, mode_dense):
        """
        返回：多尺度意图嵌入融合后的 [B, M, T, D]
        步骤：
        1. 每级降采样 → 查询对应意图表 → 得到该尺度的意图嵌入
        2. 插值回 T 长度
        3. 加权融合（可学习权重）
        """
        B, M, T, D = mode_dense.shape
        K = 8  # 每级意图数

        # 1. 三级降采样 + 意图嵌入（einsum 向量化）
        feat_s = mode_dense                                          # [B, M, T, D]
        feat_m = F.avg_pool1d(mode_dense.view(B*M, T, D).transpose(1,2), kernel_size=4, stride=4).transpose(1,2)  # [B*M, T//4, D]
        feat_l = F.avg_pool1d(mode_dense.view(B*M, T, D).transpose(1,2), kernel_size=16, stride=16).transpose(1,2)  # [B*M, T//16, D]
        feat_m = feat_m.view(B, M, -1, D)
        feat_l = feat_l.view(B, M, -1, D)
        # 意图嵌入：每级查询自己的意图表 → [B*M, T', K] 权重 → 加权求和得嵌入
        def embed_intent(feat, bank):
            # feat: [B*M, T', D]   bank: [K, D]
            logits = torch.einsum('bmtd,kd->bmtk', feat, bank)  # [B*M, T', K]
            weights = F.softmax(logits, dim=-1)               # [B*M, T', K]
            embed = torch.einsum('bmtk,kd->bmtd', weights, bank)  # [B*M, T', D]
            return embed

        embed_s = embed_intent(feat_s, self.intent_bank_s)   # [B*M, T, D]
        embed_m = embed_intent(feat_m, self.intent_bank_m)   # [B*M, T//4, D]
        embed_l = embed_intent(feat_l, self.intent_bank_l)   # [B*M, T//16, D]

        # 2. 插值回 T 长度（嵌入特征）
        def upsample_embed(embed, T_target):
            B, M, T_prime, D = embed.shape
            embed_3d = embed.permute(0, 1, 3, 2).reshape(B * M, D, T_prime)  # [B*M, D, T']
            embed_3d = F.interpolate(embed_3d, size=T_target, mode='linear', align_corners=False)
            return embed_3d.view(B, M, D, T_target).permute(0, 1, 3, 2)  # [B, M, T, D]

        embed_m = upsample_embed(embed_m, T)
        embed_l = upsample_embed(embed_l, T)

        # 3. 加权融合（可学习权重）
        fusion_w = F.softmax(self.fusion_w, dim=0)  # [3]
        intent_fused = fusion_w[0] * embed_s + fusion_w[1] * embed_m + fusion_w[2] * embed_l  # [B, M, T, D]

        return intent_fused 
    def forward(
        self,
        context: torch.Tensor,      # [B, N, D]
        segment: int,   # [B, 2]
        key_padding_mask=None,
        lane_mask=None,
    ):
        """
        Returns:
            trajectories: [B, M, T, 2]  # M 条未来轨迹
            confidences:  [B, M]       # 每条轨迹的未归一化置信度（logits）
        """
        B = context.shape[0]
        # self.visualize_intent_table(self.intent_bank)

        # --- 步骤 1: 初始化 K 个模态的“思考状态” ---
        mode = self.mode_queries.expand(B, -1, -1) # (B, K, D)
        
        for blk in self.query_mode:
            mode = blk(src=mode, src_kv=context, key_padding_mask=key_padding_mask)
        y_hat, pi, scal = self.predictor(mode)
        y_hat = torch.cumsum(y_hat, dim=-2)
        scal = torch.cumsum(scal, dim=-2)

        time = torch.arange(60).long().to(context.device)
        time = time * 0.1 + 0.1
        time = time.unsqueeze(-1)
        intent = self.time_embedding_mlp(time)
        intent = intent.repeat(context.size(0), 1, 1)

        for blk in self.query_intent:
            intent = blk(src=intent, src_kv=context, key_padding_mask=key_padding_mask)
        
        dense_pred = self.dense_predict(intent)
        dense_pred = torch.cumsum(dense_pred, dim=-2)

        mode_dense = mode[:, :, None] + intent[:, None, :]
        B, M, T, C = mode_dense.shape
        
        mode_dense = mode_dense.reshape(B, -1, C)
        for blk in self.intent_mode_querry:
            mode_dense = blk(src=mode_dense, src_kv=context, key_padding_mask=key_padding_mask)
        mode_dense = mode_dense.reshape(B, M, T, C)

        # Step 4: 软查询意图表 → 每个 (b,m,t) 得到意图嵌入
        # logits: [B, M, T, K]
        # logits = torch.einsum('bmtd,kd->bmtk', mode_dense, self.intent_bank)
        # weights = F.softmax(logits / self.temp, dim=-1)  # [B, M, T, K]
        # intent_seq = torch.einsum('bmtk,kd->bmtd', weights, self.intent_bank)  # [B, M, T, D]
        intent_seq = self.pyramid_intent_seq(mode_dense)
        # z = mode_dense.reshape(-1, C)          # (B*M*T, D)
        # z_q, idx, vq_loss = self.intent_bank(z)
        # idx = idx.reshape(B, M, T)
        # intent_seq = z_q.reshape(B, M, T, C)
        # intent_res = self.intent_res_mlp(intent_seq)     # (B,M,T,D)
        # intent_seq = mode_dense + intent_res              # 关键残差

        y_hat_dense, pi_dense, scal_dense = self.predictor_dense(intent_seq)  # [B, M, T, 2]

        y_hat_dense = torch.cumsum(y_hat_dense, dim=2)  # [B, M, T, 2]
        scal_dense = torch.cumsum(scal_dense, dim=2)

        return {
        # "weights": weights,
        "mode": mode, 
        "dense_pred": dense_pred,
        "y_hat": y_hat,      # [B, M, T, 2]
        "pi": pi,              # [B, M]
        "scal": scal,             # [B, M, T, 2]
        "new_y_hat": y_hat_dense, # [B, M, T, 2]
        "new_pi": pi_dense,         # [B, M]
        "scal_new": scal_dense         # [B, M, T, 2]
    }

class MultiModalIntentDecoder2(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_intents = 8,       # 意图表大小 K
        future_steps=60,      # T
        num_modes = 6,          # M (e.g., 6 for Argoverse2)
        output_dim= 2,
        num_heads = 8,
        mlp_ratio = 4.0,
        qkv_bias = False,
        drop = 0.2,
        attn_drop = 0.2,
        drop_path= 0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        query_cross_layers=2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.output_dim = output_dim

        # === 共享组件 ===
        self.intent_bank = nn.Parameter(torch.randn(num_intents, embed_dim))
        nn.init.xavier_uniform_(self.intent_bank)


        # 可学习的时间嵌入 [T, D]
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )
        self.query_mode =nn.Sequential(
            nn.Linear(1, 256), 
            nn.GELU(), 
            nn.Linear(256, self.num_modes)
        )
        self.query_intent =nn.Sequential(
            nn.Linear(1, 256), 
            nn.GELU(), 
            nn.Linear(256, self.future_steps)
        )


        self.predictor = GMMPredictor(future_steps)
        self.predictor_dense = GMMPredictor_dense(future_steps)
        # 5. 温度（可选固定或可学习）
        self.temp = 1.0  # 或设为 nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        context: torch.Tensor,      # [B, N, D]
        segment: int,   # [B, 2]
        key_padding_mask=None,
        lane_mask=None,
    ):
        """
        Returns:
            trajectories: [B, M, T, 2]  # M 条未来轨迹
            confidences:  [B, M]       # 每条轨迹的未归一化置信度（logits）
        """
        B = context.shape[0]

        # --- 步骤 1: 初始化 K 个模态的“思考状态” ---
        mode = self.query_mode(context[:,:segment,...].reshape(B, -1, segment))
        mode = mode.reshape(B, self.num_modes, -1)

        y_hat, pi, scal = self.predictor(mode)
        y_hat = torch.cumsum(y_hat, dim=-2)
        scal = torch.cumsum(scal, dim=-2)


        intent = self.query_intent(context[:,:segment,...].reshape(B, -1, segment))
        intent = intent.reshape(B, self.future_steps, -1)


        mode_dense = mode[:, :, None] + intent[:, None, :]
        
        logits = torch.einsum('bmtd,kd->bmtk', mode_dense, self.intent_bank)
        weights = F.softmax(logits / self.temp, dim=-1)  # [B, M, T, K]
        intent_seq = torch.einsum('bmtk,kd->bmtd', weights, self.intent_bank)  # [B, M, T, D]

        y_hat_dense, pi_dense, scal_dense = self.predictor_dense(intent_seq)  # [B, M, T, 2]

        # 累加得到绝对坐标
        y_hat_dense = torch.cumsum(y_hat_dense, dim=2)  # [B, M, T, 2]
        scal_dense = torch.cumsum(scal_dense, dim=2)


        return {
        "y_hat": y_hat,      # [B, M, T, 2]
        "pi": pi,              # [B, M]
        "scal": scal,             # [B, M, T, 2]
        "new_y_hat": y_hat_dense, # [B, M, T, 2]
        "new_pi": pi_dense,         # [B, M]
        "scal_new": scal_dense         # [B, M, T, 2]
    }