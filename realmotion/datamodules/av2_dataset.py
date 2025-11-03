from typing import List
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from collections import Counter
class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str = None,
        num_historical_steps: int = 50,
        split_points: List[int] = [50],
        radius: float = 150.0,
        map_radius: float = 40.0,
        n_step: int = 10,
        logger=None,
    ):
        # assert split_points[-1] == 50 and num_historical_steps <= 50
        assert split in ['train', 'val', 'test']
        super(Av2Dataset, self).__init__()
        self.data_folder = Path(data_root) / split
        self.file_list = sorted(list(self.data_folder.glob('*.pt')))
        self.num_historical_steps = num_historical_steps
        self.n_step = n_step
        self.split = split
        self.num_future_steps = 0 if split =='test' else 60
        self.radius = radius
        self.map_radius = map_radius
        self.split_points = split_points

        if logger is not None:
            logger.info(f'data root: {data_root}/{split}, total number of files: {len(self.file_list)}')

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        data = torch.load(self.file_list[index])
        data = self.process(data)
        return data
    
    def process(self, data):
        if self.split_points is not None:
            sequence_data = []
            for cur_step in self.split_points:
                ag_dict = self.process_single_agent(data,cur_step)
                sequence_data.append(ag_dict)
            return sequence_data
        else:
            sequence_data = []
            ag_dict = self.process_single_agent(data,self.num_historical_steps)
            sequence_data.append(ag_dict)
            return sequence_data


    def resample_intent_labels(self,
        intent_sequence, 
        num_segments,
        intent_priority_list=None
    ) :
        """
        将一个意图标签序列重新采样为指定数量的段。

        该函数将原始序列划分为N个大致均匀的块，并为每个块确定一个代表性标签。
        确定标签的规则是：
        1. 选择该块中出现次数最多的标签（众数）。
        2. 如果存在多个众数（出现次数相同），则根据提供的优先级列表选择优先级最高的那个。
        3. 如果没有提供优先级列表，则简单地选择第一个出现的众数。

        Args:
            intent_sequence (List[int]): 原始的意图标签序列，例如 [0, 0, 1, 1, 2, 5]。
            num_segments (int): 想要得到的新序列的长度（段数）。
            intent_priority_list (List[int], optional):
                一个定义了意图优先级的列表。列表中的标签优先级从高到低排列。
                例如：[5, 2, 3, 4, 1, 0] 表示标签5(停止)的优先级最高，0(直行)最低。
                这在处理平局时至关重要。Defaults to None.

        Returns:
            List[int]: 重新采样后的新意图标签序列，长度为 num_segments。

        Raises:
            ValueError: 如果 num_segments 大于原始序列的长度。
        """
        num_agent, original_length = intent_sequence.shape
        if num_segments > original_length:
            raise ValueError(f"目标段数 ({num_segments}) 不能大于原始序列长度 ({original_length})。")

        batch_results = []
        segment_boundaries = [round(i * original_length / num_segments) for i in range(num_segments + 1)]

        for i in range(num_agent):
            single_sequence = intent_sequence[i].tolist() # 转换为列表以便使用 Counter
            new_single_sequence = []

            for j in range(num_segments):
                start_index = segment_boundaries[j]
                end_index = segment_boundaries[j+1]
                
                # 提取当前块的子序列
                segment_labels = single_sequence[start_index:end_index]

                if not segment_labels:
                    # 处理空块的情况
                    if new_single_sequence:
                        new_single_sequence.append(new_single_sequence[-1])
                    else:
                        new_single_sequence.append(single_sequence[0])
                    continue

                # 1. 统计块内每个标签的出现次数
                counts = Counter(segment_labels)
                max_count = max(counts.values())
                
                # 2. 找到所有出现次数最多的标签（众数）
                modes = [label for label, count in counts.items() if count == max_count]

                # 3. 根据规则选择最终标签
                if len(modes) == 1:
                    chosen_label = modes[0]
                else:
                    # 处理平局
                    if intent_priority_list:
                        # 按优先级列表排序，优先级高的在前
                        modes.sort(key=lambda label: intent_priority_list.index(label))
                        chosen_label = modes[0] # 选择优先级最高的那个
                    else:
                        # 如果没有优先级，简单选择数值最小的那个以保证确定性
                        chosen_label = min(modes)
                
                new_single_sequence.append(chosen_label)
            
            batch_results.append(new_single_sequence)
            
        # 将结果列表转换为张量，并确保它在原始设备上
        return torch.tensor(batch_results, dtype=torch.long, device=intent_sequence.device)

    def process_single_agent(self, data, step):
        idx = data['focal_idx']
        cur_agent_id = data['agent_ids'][idx]
        origin = data['x_positions'][idx, step - 1]
        theta = data['x_angles'][idx, step - 1]
        rotate_mat = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ],
        )
        if self.radius > 0:
            ag_mask = torch.norm(data['x_positions'][:, step - 1] - origin, dim=-1) < self.radius
        else:
            ag_mask = torch.ones(len(data['x_positions']), dtype=torch.bool)
        ag_mask = ag_mask * data['x_valid_mask'][:, step - 1]
        ag_mask[idx] = False

        # transform agents to local
        st, ed = step - self.num_historical_steps, step + self.num_future_steps
        attr = torch.cat([data['x_attr'][[idx]], data['x_attr'][ag_mask]])
        pos = data['x_positions'][:, st: ed]
        pos = torch.cat([pos[[idx]], pos[ag_mask]])
        head = data['x_angles'][:, st: ed]
        head = torch.cat([head[[idx]], head[ag_mask]])
        vel = data['x_velocity'][:, st: ed]
        vel = torch.cat([vel[[idx]], vel[ag_mask]])
        valid_mask = data['x_valid_mask'][:, st: ed]
        valid_mask = torch.cat([valid_mask[[idx]], valid_mask[ag_mask]])
        
        intent_series = data['driving_intent_sequence'][self.n_step]
        if self.n_step is not None:
            intent = intent_series
            intent = torch.cat([intent[[idx]], intent[ag_mask]])
        else:
           intent = torch.ones(pos.shape[0], 1, dtype=torch.long) * 8
        
        pos[valid_mask] = torch.matmul(pos[valid_mask] - origin, rotate_mat)
        head[valid_mask] = (head[valid_mask] - theta + np.pi) % (2 * np.pi) - np.pi

        # transform lanes to local
        l_pos = data['lane_positions']
        l_attr = data['lane_attr']
        l_is_int = data['is_intersections']
        l_pos = torch.matmul(l_pos.reshape(-1, 2) - origin, rotate_mat).reshape(-1, l_pos.size(1), 2)

        l_ctr = l_pos[:, 9:11].mean(dim=1)
        l_head = torch.atan2(
            l_pos[:, 10, 1] - l_pos[:, 9, 1],
            l_pos[:, 10, 0] - l_pos[:, 9, 0],
        )
        if self.map_radius != 0:
            l_valid_mask = (
                (l_pos[:, :, 0] > -self.map_radius) & (l_pos[:, :, 0] < self.map_radius)
                & (l_pos[:, :, 1] > -self.map_radius) & (l_pos[:, :, 1] < self.map_radius)
            )
        else:
            l_valid_mask = torch.ones(l_pos.size(0), l_pos.size(1), dtype=torch.bool)

        l_mask = l_valid_mask.any(dim=-1)
        l_pos = l_pos[l_mask]
        l_is_int = l_is_int[l_mask]
        l_attr = l_attr[l_mask]
        l_ctr = l_ctr[l_mask]
        l_head = l_head[l_mask]
        l_valid_mask = l_valid_mask[l_mask]

        l_pos = torch.where(
            l_valid_mask[..., None], l_pos, torch.zeros_like(l_pos)
        )

        # remove outliers
        nearest_dist = torch.cdist(pos[:, self.num_historical_steps - 1, :2],
                                   l_pos.view(-1, 2)).min(dim=1).values
        ag_mask2 = nearest_dist < 5
        ag_mask2[0] = True
        pos = pos[ag_mask2]
        head = head[ag_mask2]
        vel = vel[ag_mask2]
        attr = attr[ag_mask2]
        valid_mask = valid_mask[ag_mask2]

        intent = intent[ag_mask2]

        initial_selected_indices = torch.cat([torch.tensor([idx]), torch.where(ag_mask)[0]])
        # 然后记录经过第二次筛选后的相对索引
        final_selected_mask = ag_mask2
        # 最终保留的原始索引
        original_indices = initial_selected_indices[final_selected_mask]

        # post_process
        head = head[:, :self.num_historical_steps]
        vel = vel[:, :self.num_historical_steps]
        pos_ctr = pos[:, self.num_historical_steps - 1].clone()
        if self.num_future_steps > 0:
            type_mask = attr[:, [-1]] != 3
            pos, target = pos[:, :self.num_historical_steps], pos[:, self.num_historical_steps:]
            target_mask = type_mask & valid_mask[:, [self.num_historical_steps - 1]] & valid_mask[:, self.num_historical_steps:]
            valid_mask = valid_mask[:, :self.num_historical_steps]
            target = torch.where(
                target_mask.unsqueeze(-1),
                target - pos_ctr.unsqueeze(1), torch.zeros_like(target),   
            )
        else:
            target = target_mask = None

        diff_mask = valid_mask[:, :self.num_historical_steps - 1] & valid_mask[:, 1: self.num_historical_steps]
        tmp_pos = pos.clone()
        pos_diff = pos[:, 1:self.num_historical_steps] - pos[:, :self.num_historical_steps - 1]
        pos[:, 1:self.num_historical_steps] = torch.where(
            diff_mask.unsqueeze(-1),
            pos_diff, torch.zeros(pos.size(0), self.num_historical_steps - 1, 2)
        )
        pos[:, 0] = torch.zeros(pos.size(0), 2)

        tmp_vel = vel.clone()
        vel_diff = vel[:, 1:self.num_historical_steps] - vel[:, :self.num_historical_steps - 1]
        vel[:, 1:self.num_historical_steps] = torch.where(
            diff_mask,
            vel_diff, torch.zeros(vel.size(0), self.num_historical_steps - 1)
        )
        vel[:, 0] = torch.zeros(vel.size(0))

        
        return {
            'target': target,                  #[23, 60, 2]
            'target_mask': target_mask,        #[23, 60]
            'x_positions_diff': pos,           #[23, 50, 2]
            'x_positions': tmp_pos,            #[23, 50, 2]
            'x_attr': attr,                    #[23, 3]
            'x_centers': pos_ctr,              #[23, 2]                            
            'x_angles': head,                  #[23, 50]
            'x_velocity': tmp_vel,             #[23, 50]
            'x_velocity_diff': vel,            #[23, 50]
            'x_valid_mask': valid_mask,        #[23, 50]
            'lane_positions': l_pos,           #[20, 20, 2]
            'lane_centers': l_ctr,             #[20, 2]
            'lane_angles': l_head,             #[20]
            'lane_attr': l_attr,               #[20, 3]
            'lane_valid_mask': l_valid_mask,   #[20, 20]
            'is_intersections': l_is_int,      #[20]
            'origin': origin.view(1, 2),       #[1, 2]
            'theta': theta.view(1),            #[1]
            'scenario_id': data['scenario_id'],#str
            'track_id': cur_agent_id,
            'city': data['city'],
            'timestamp': torch.Tensor([step * 0.1]),
            'driving_intent': intent,
            'agent_indices': original_indices,
        }

    
def collate_fn(seq_batch):
    """
    处理批次数据，每条数据是一个字典（而非列表）
    对具有不同维度的张量进行填充以形成批次
    """
    seq_data = []
    for i in range(len(seq_batch[0])):
        batch = [b[i] for b in seq_batch]
        data = {}
        
        # 需要使用 pad_sequence 填充的字段
        padded_fields = [
            'x_positions_diff',
            'x_attr',
            'x_positions',
            'x_centers',
            'x_angles',
            'x_velocity',
            'x_velocity_diff',
            'lane_positions',
            'lane_centers',
            'lane_angles',
            'lane_attr',
            'is_intersections',
        ]
        
        # 对需要填充的字段进行批处理
        for key in padded_fields:
            if key in batch[0]:
                data[key] = pad_sequence([b[key] for b in batch], batch_first=True)
        
        # 处理可选字段 x_scored
        if 'x_scored' in batch[0]:
            data['x_scored'] = pad_sequence(
                [b['x_scored'] for b in batch], batch_first=True
            )
        
        # 处理目标相关字段
        if batch[0]['target'] is not None:
            data['target'] = pad_sequence([b['target'] for b in batch], batch_first=True)
            data['target_mask'] = pad_sequence(
                [b['target_mask'] for b in batch], batch_first=True, padding_value=False
            )
        
        # 处理掩码字段
        mask_fields = ['x_valid_mask', 'lane_valid_mask']
        for key in mask_fields:
            if key in batch[0]:
                data[key] = pad_sequence(
                    [b[key] for b in batch], batch_first=True, padding_value=False
                )
        
        # 计算关键有效掩码
        if 'x_valid_mask' in data:
            data['x_key_valid_mask'] = data['x_valid_mask'].any(-1)
        if 'lane_valid_mask' in data:
            data['lane_key_valid_mask'] = data['lane_valid_mask'].any(-1)
        
        # 处理非张量字段
        data['scenario_id'] = [b['scenario_id'] for b in batch]
        data['track_id'] = [b['track_id'] for b in batch]
        data['city'] = [b['city'] for b in batch]
        
        # 合并张量字段
        data['origin'] = torch.cat([b['origin'] for b in batch], dim=0)
        data['theta'] = torch.cat([b['theta'] for b in batch])
        data['timestamp'] = torch.cat([b['timestamp'] for b in batch])
        
        # 处理驾驶意图
        if batch[0]['target'] is not None and 'driving_intent' in batch[0]:
            data['intent'] = pad_sequence([b['driving_intent'] for b in batch], batch_first=True)
        seq_data.append(data)
    return seq_data
