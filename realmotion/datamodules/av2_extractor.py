import math
import traceback
from pathlib import Path
from typing import List
import av2.geometry.interpolate as interp_utils
import numpy as np
import torch
from scipy.spatial import KDTree
from av2.map.map_api import ArgoverseStaticMap
from .av2_data_utils import (
    OBJECT_TYPE_MAP,
    OBJECT_TYPE_MAP_COMBINED,
    LaneTypeMap,
    load_av2_df,
    
)

from vis import generate_scenario_visualizations
from shapely.geometry import Point, Polygon

def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi] 范围"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def unwrap_angle(angle: np.ndarray) -> np.ndarray:
    """将角度数组从 [0, 2*pi] 范围解开到 [-pi, pi] 范围。

    Args:
        angle: 以弧度为单位的角度数组。

    Returns:
        解开后的角度数组。
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Av2Extractor:
    def __init__(
        self,
        radius: float = 150,
        save_path: Path = None,
        mode: str = "train",
        n_steps: List[int] = [10, 20, 30, 60],
        ignore_type: List[int] = [5, 6, 7, 8, 9],
        remove_outlier_actors: bool = True,
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.ignore_type = ignore_type
        self.turn_angle_threshold_deg = 30.0
        self.TURN_DIRECTION_MAP = {
            'STRAIGHT': 0,
            'LEFT': 1,
            'RIGHT': 2,
            'NONE': 0, # 'NONE' 通常和 'STRAIGHT' 是一个意思
            # 你可以根据需要添加更多
        }

        self.n_steps = n_steps
        self.INTENTION_LABELS = {
            'straight': 0,
            'turn_left': 1,
            'turn_right': 2,
            'lane_change_left': 3,
            'lane_change_right': 4,
            'u_turn': 5,
            'stop': 6,
            'merge': 7,
            'undefined': 8,
        }
        self.INTENTION_LABELS_reverse = {v: k for k, v in self.INTENTION_LABELS.items()}
    
    def save(self, file: Path):
        assert self.save_path is not None

        try:
            data = self.get_data(file)
        except Exception:
            print(traceback.format_exc())
            print("found error while extracting data from {}".format(file))
        save_file = self.save_path / (file.stem + ".pt")
        torch.save(data, save_file)

    def get_data(self, file: Path):
        return self.process(file)

    def process(self, raw_path: str, agent_id=None):
        df, am, scenario_id = load_av2_df(raw_path)
        city = df.city.values[0]

        timestamps = list(np.sort(df["timestep"].unique()))
        cur_df = df[df["timestep"] == timestamps[49]]
        actor_ids = list(df["track_id"].unique())
        num_nodes = len(actor_ids)

        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 3, dtype=torch.uint8)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)
        scored_idx = []
        intents = {}
        for n_step in self.n_steps:
            num_intent_steps = 60 // n_step
            driving_intent_sequence = torch.full((num_nodes, num_intent_steps),
                                                  fill_value=self.INTENTION_LABELS['undefined'],
                                                  dtype=torch.long)
            intents[n_step] = driving_intent_sequence

        for actor_id, actor_df in df.groupby("track_id"):
            node_idx = actor_ids.index(actor_id)
            node_steps = [timestamps.index(ts) for ts in actor_df["timestep"]]
            object_type = OBJECT_TYPE_MAP[actor_df["object_type"].values[0]]
            object_category = actor_df["object_category"].values[0]
            x_attr[node_idx, 0] = object_type
            x_attr[node_idx, 1] = object_category
            x_attr[node_idx, 2] = OBJECT_TYPE_MAP_COMBINED[
                actor_df["object_type"].values[0]
            ]
            if object_category == 3:
                focal_idx = node_idx
            if object_category == 2:
                scored_idx.append(node_idx)

            padding_mask[node_idx, node_steps] = False

            pos_xy = torch.from_numpy(
                np.stack(
                    [actor_df["position_x"].values, actor_df["position_y"].values],
                    axis=-1,
                )
            ).float()
            heading = torch.from_numpy(actor_df["heading"].values).float()
            velocity = torch.from_numpy(
                actor_df[["velocity_x", "velocity_y"]].values
            ).float()
            velocity_norm = torch.norm(velocity, dim=1)

            

            x[node_idx, node_steps, :2] = pos_xy
            x_heading[node_idx, node_steps] = heading
            x_velocity[node_idx, node_steps] = velocity_norm

            for n_step in self.n_steps:
                num_intent_steps = 60 // n_step
                if object_category == 3 or object_category == 2:
                    for i in range(num_intent_steps):
                        start_step = i * n_step + 50
                        end_step = start_step + n_step
                        # 提取窗口数据
                        window_positions = pos_xy[start_step:end_step, :2].numpy()
                        window_headings = heading[start_step:end_step].numpy()
                        window_velocities = velocity[start_step:end_step, :2].numpy()

                        # 调用辅助函数获取意图
                        intent = self._get_driving_intent_for_window(
                            window_positions, window_headings, window_velocities, am
                        )
                        
                        intents[n_step][node_idx, i] = intent


        # FOCAL_TRACK = driving_intent_sequence[focal_idx,:]
        # for i in range(len(FOCAL_TRACK)):
        #     print(f"{i} intent: {self.INTENTION_LABELS_reverse[FOCAL_TRACK[i].item()]}({FOCAL_TRACK[i].item()})")
        # generate_scenario_visualizations(
        #     argoverse_scenario_dir=raw_path.parent,
        #     viz_output_dir=Path("./vis_out"),
        #     num_scenarios=100,
        #     selection_criteria="first",
        #     debug=True,
        #     last=True,
        # )
            

        (
            lane_positions,
            is_intersections,
            lane_attr,
        ) = self.get_lane_features(am)

        return {
            "x_positions": x,
            "x_attr": x_attr,
            "x_angles": x_heading,
            "x_velocity": x_velocity,
            "x_valid_mask": ~padding_mask,
            "lane_positions": lane_positions,
            "lane_attr": lane_attr,
            "is_intersections": is_intersections,
            "scenario_id": scenario_id,
            "agent_ids": actor_ids,
            "focal_idx": focal_idx,
            "scored_idx": scored_idx,
            "city": city,
            "driving_intent_sequence": intents,
        }
    
    def _find_lane_id_for_point(self, point_xy: np.ndarray, map_api: ArgoverseStaticMap):
        """
        根据给定的2D坐标点，查找其所在的车道ID。
        这是必须手动实现的核心功能，因为 Argoverse 2 Map API 没有提供直接的查找方法。
        """
        # 使用 get_nearby_lane_segments 来缩小搜索范围，提高效率
        # 如果这个方法不可靠或不存在，可以回退到 get_scenario_lane_segments()
        try:
            candidate_lanes = map_api.get_nearby_lane_segments(point_xy, search_radius_m=5.0)
        except NotImplementedError:
            candidate_lanes = map_api.get_scenario_lane_segments()

        point_geom = Point(point_xy)
        
        for lane_segment in candidate_lanes:
            # 获取车道的多边形边界 (通常是 (N, 3) 的 numpy 数组)
            polygon_boundary = lane_segment.polygon_boundary
            # 转换为 Shapely Polygon 对象 (我们只关心 x, y)
            lane_polygon = Polygon(polygon_boundary[:, :2])
            
            # 检查点是否在多边形内部
            if lane_polygon.contains(point_geom):
                return lane_segment.id  # 如果找到，立即返回车道ID

        return None # 如果没有找到任何包含该点的车道，返回 None

    def _get_driving_intent_for_window(
        self,
        positions: np.ndarray,
        headings: np.ndarray,
        velocities: np.ndarray,
        map_api: ArgoverseStaticMap,
    ):
        """
        根据给定的轨迹片段（一个时间窗口）判断驾驶意图。
        【最终版：严格遵守您提供的 ArgoverseStaticMap 类定义】
        """
        # 1. 判断是否停止 (逻辑不变)
        avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
        if avg_speed < 0.5:
            return self.INTENTION_LABELS['stop']

        start_pos = positions[0]
        end_pos = positions[-1]

        # 2. 判断是否变道 (使用我们新实现的 _find_lane_id_for_point 函数)
        start_lane_id = self._find_lane_id_for_point(start_pos[:2], map_api)
        end_lane_id = self._find_lane_id_for_point(end_pos[:2], map_api)

        if start_lane_id is not None and end_lane_id is not None and start_lane_id != end_lane_id:
            # 确认是左右变道，而不是路口转向
            left_neighbor_id = map_api.get_lane_segment_left_neighbor_id(start_lane_id)
            right_neighbor_id = map_api.get_lane_segment_right_neighbor_id(start_lane_id)
            
            if left_neighbor_id == end_lane_id:
                return self.INTENTION_LABELS['lane_change_left']
            if right_neighbor_id == end_lane_id:
                return self.INTENTION_LABELS['lane_change_right']
            # 如果不是直接相邻，则可能是转向，交由后续逻辑判断
            
        # 3. 判断转向 (逻辑不变)
        start_heading = headings[0]
        end_heading = headings[-1]
        heading_change = normalize_angle(end_heading - start_heading)

        u_turn_threshold = math.radians(150)
        turn_threshold = math.radians(10)
        straight_threshold = math.radians(10)

        if abs(heading_change) > u_turn_threshold:
            return self.INTENTION_LABELS['u_turn']
        elif heading_change > turn_threshold:
            return self.INTENTION_LABELS['turn_left']
        elif heading_change < -turn_threshold:
            return self.INTENTION_LABELS['turn_right']
        
        # 4. 判断直行 (逻辑不变)
        if abs(heading_change) < straight_threshold:
            return self.INTENTION_LABELS['straight']

        # 5. 其他情况归为未定义 (逻辑不变)
        return self.INTENTION_LABELS['undefined']

    @staticmethod
    def get_lane_features(
        am: ArgoverseStaticMap,
    ):
        # lane_segments = am.get_nearby_lane_segments(query_pos.numpy(), radius)
        lane_segments = am.get_scenario_lane_segments()

        lane_positions, is_intersections, lane_attrs = [], [], []
        for segment in lane_segments:
            lane_centerline, lane_width = interp_utils.compute_midpoint_line(
                left_ln_boundary=segment.left_lane_boundary.xyz,
                right_ln_boundary=segment.right_lane_boundary.xyz,
                num_interp_pts=20,
            )
            lane_centerline = torch.from_numpy(lane_centerline[:, :2]).float()
            is_intersection = am.lane_is_in_intersection(segment.id)

            lane_positions.append(lane_centerline)
            is_intersections.append(is_intersection)

            lane_type = LaneTypeMap[segment.lane_type]
            attribute = torch.tensor(
                [lane_type, lane_width, is_intersection], dtype=torch.float
            )
            lane_attrs.append(attribute)

        lane_positions = torch.stack(lane_positions)
        is_intersections = torch.Tensor(is_intersections)
        lane_attrs = torch.stack(lane_attrs, dim=0)

        return (
            lane_positions,
            is_intersections,
            lane_attrs,
        )
