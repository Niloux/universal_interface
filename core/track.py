#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track数据处理模块

该模块负责处理轨迹相关数据。
"""

import json
import math
import pickle
from typing import Dict, Optional

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from .base import BaseProcessor
from utils import data_io, geometry, default_logger
from utils.config import Config
from utils.structures import Box3D, FrameObject, TrajectoryData


class TrackProcessor(BaseProcessor):
    """
    轨迹数据处理器

    负责处理轨迹相关数据的读取、转换和输出
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.raw_input_path = self.config.input
        self.output_root_path = self.config.output
        self.input_path = self.raw_input_path / "objects"
        self.output_path = self.output_root_path / "track"
        self.image_output_path = self.output_root_path / "images"

        self.camera_ids = [str(v) for v in self.config.camera.id_map.values()]
        self.ensure_dir(self.output_path)

        self.category_mapping = {
            "vehicle": "vehicle",
            "pedestrian": "pedestrian",
            "bicycle": "cyclist",
            "motorcycle": "cyclist",
            "sign": "sign",
            "traffic": "sign",
        }

    def _map_category(self, name: str) -> Optional[str]:
        name = name.lower()
        for key, value in self.category_mapping.items():
            if key in name:
                return value
        return None

    def process(self) -> None:
        """处理轨迹数据的主流程"""
        if not self.check_dir(self.input_path):
            default_logger.warning(f"track目录不存在: {self.input_path}, 跳过处理")
            return

        default_logger.info("第一步: 处理帧数据以生成轨迹...")
        processed_frames = self._process_frames()

        default_logger.info("第二步: 计算最终轨迹信息...")
        trajectory_info = self._calculate_final_trajectories(processed_frames["trajectory"])

        default_logger.info("第三步: 保存处理结果...")
        self._save_results(processed_frames, trajectory_info)
        default_logger.success("轨迹数据处理完成")

    def _process_frames(self) -> Dict:
        """遍历所有帧，处理每个物体并收集原始轨迹"""
        track_info, track_camera_visible, trajectory, object_ids = {}, {}, {}, {}
        track_vis_imgs = []

        extrinsics = data_io.load_extrinsics(self.output_root_path, self.camera_ids)
        intrinsics = data_io.load_intrinsics(self.output_root_path, self.camera_ids)

        frame_files = sorted(self.input_path.iterdir())
        for frame_id, f in enumerate(tqdm(frame_files, desc="处理帧")):
            with open(f, "r") as f_json:
                data = json.load(f_json)

            frame_name = f"{frame_id:06d}"
            track_info[frame_name] = {}
            track_camera_visible[frame_name] = {cam_id: [] for cam_id in self.camera_ids}

            images = data_io.load_images(self.image_output_path, frame_name, self.camera_ids)

            for obj_data in data:
                track_id = obj_data.get("track_id")
                if track_id not in object_ids:
                    object_ids[track_id] = len(object_ids)

                box_info = self._extract_box_info(obj_data, frame_id)
                track_info[frame_name][track_id] = FrameObject(lidar_box=box_info, camera_box=box_info)

                if track_id not in trajectory:
                    trajectory[track_id] = {}
                trajectory[track_id][frame_name] = box_info

                self._update_camera_visibility(
                    track_camera_visible[frame_name], images, extrinsics, intrinsics, box_info, track_id
                )

            if all(cam_id in images for cam_id in ["0", "1", "2"]):
                vis_img = np.concatenate([images["0"], images["1"], images["2"]], axis=1)
                track_vis_imgs.append(vis_img)

        return {
            "track_info": track_info,
            "track_camera_visible": track_camera_visible,
            "trajectory": trajectory,
            "object_ids": object_ids,
            "vis_images": track_vis_imgs,
        }

    def _extract_box_info(self, obj_data: Dict, frame_id: int) -> Box3D:
        """从原始对象数据中提取并格式化为Box3D对象"""
        center = obj_data.get("box3d_center")
        size = obj_data.get("box3d_size")
        return Box3D(
            height=size[2],
            width=size[1],
            length=size[0],
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
            heading=obj_data.get("box3d_heading"),
            label=self._map_category(obj_data.get("label")),
            speed=0.0,
            timestamp=data_io.load_timestamp(self.raw_input_path, frame_id),
        )

    def _update_camera_visibility(self, visibility_dict, images, extrinsics, intrinsics, box_info: Box3D, track_id):
        """检查并更新物体在各个相机中的可见性，并绘制3D框"""
        for cam_id in self.camera_ids:
            img_shape = images[cam_id].shape[:2] if cam_id in images else (1080, 1920)

            if cam_id not in extrinsics or cam_id not in intrinsics:
                continue

            visible, pts_2d = geometry.project_box_to_image(box_info, extrinsics[cam_id], intrinsics[cam_id], img_shape)

            if visible:
                visibility_dict[cam_id].append(track_id)
                if cam_id in ["0", "1", "2"] and cam_id in images:
                    self._draw_3d_box(images[cam_id], pts_2d)

    def _draw_3d_box(self, image: np.ndarray, pts_2d: np.ndarray):
        """在图像上绘制3D边界框"""
        for i, j in [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]:
            pt1, pt2 = tuple(pts_2d[i].astype(int)), tuple(pts_2d[j].astype(int))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

    def _calculate_final_trajectories(self, raw_trajectory: Dict[str, Dict[str, Box3D]]) -> Dict[str, TrajectoryData]:
        """基于原始轨迹，计算最终的轨迹信息"""
        trajectory_info = {}
        for track_id, info in raw_trajectory.items():
            if len(info) < 2:
                continue

            frames = sorted(info.keys(), key=int)
            first_box = info[frames[0]]

            poses_vehicle, poses_world, dims, timestamps, speeds = [], [], [], [], []

            for frame_name in frames:
                box = info[frame_name]
                dims.append([box.height, box.width, box.length])
                timestamps.append(box.timestamp)
                speeds.append(box.speed)

                pose_vehicle = np.eye(4)
                pose_vehicle[:3, :3] = np.array(
                    [
                        [math.cos(box.heading), -math.sin(box.heading), 0],
                        [math.sin(box.heading), math.cos(box.heading), 0],
                        [0, 0, 1],
                    ]
                )
                pose_vehicle[:3, 3] = np.array([box.center_x, box.center_y, box.center_z])
                ego_pose = data_io.load_ego_pose(self.output_root_path, int(frame_name))
                poses_vehicle.append(pose_vehicle.astype(np.float32))
                poses_world.append(np.matmul(ego_pose, pose_vehicle).astype(np.float32))

            dim = np.max(np.array(dims), axis=0)
            positions = np.array(poses_world)[:, :3, 3]
            distance = np.linalg.norm(positions[0] - positions[-1])
            dynamic = np.any(np.std(positions, axis=0) > 0.5) or distance > 2

            trajectory_info[track_id] = TrajectoryData(
                label=first_box.label,
                height=dim[0],
                width=dim[1],
                length=dim[2],
                poses_vehicle=np.array(poses_vehicle),
                timestamps=timestamps,
                frames=[int(f) for f in frames],
                speeds=speeds,
                symmetric=first_box.label != "pedestrian",
                deformable=first_box.label == "pedestrian",
                stationary=not dynamic,
            )
        return trajectory_info

    def _save_results(self, processed_frames: Dict, trajectory_info: Dict):
        """保存所有处理结果到文件"""
        if processed_frames["vis_images"]:
            imageio.mimwrite(str(self.output_path / "track_vis.mp4"), processed_frames["vis_images"], fps=24)

        with open(self.output_path / "track_info.pkl", "wb") as f:
            pickle.dump(processed_frames["track_info"], f)
        with open(self.output_path / "track_ids.json", "w") as f:
            json.dump(processed_frames["object_ids"], f, indent=4)
        with open(self.output_path / "track_camera_visible.pkl", "wb") as f:
            pickle.dump(processed_frames["track_camera_visible"], f)
        with open(self.output_path / "trajectory.pkl", "wb") as f:
            pickle.dump(trajectory_info, f)
