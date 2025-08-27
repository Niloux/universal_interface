#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track数据处理模块

该模块负责处理轨迹相关数据，包括：
track_info.pkl
track_ids.json
trajectory.pkl
track_camera_visible.pkl
track_vis.mp4

"""

import json
import math
import os
import pickle
from typing import Any, Dict, Optional

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from .base import BaseProcessor
from utils import data_io, geometry

CAMERA = ["0", "1", "2", "3", "4"]


class TrackProcessor(BaseProcessor):
    """
    轨迹数据处理器

    负责处理轨迹相关数据的读取、转换和输出
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Track处理器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.raw_input_path = self.config["input"]
        self.output_root_path = self.config["output"]
        self.input_path = os.path.join(self.raw_input_path, "objects")
        self.output_path = os.path.join(self.output_root_path, "track")
        self.image_output_path = os.path.join(self.output_root_path, "images")

        self.ensure_dir(self.output_path)

    def _map_category(self, name: str) -> Optional[str]:
        name = name.lower()
        if "vehicle" in name:
            return "vehicle"
        elif "pedestrian" in name:
            return "pedestrian"
        elif "bicycle" in name or "motorcycle" in name:
            return "cyclist"
        elif "sign" in name or "traffic" in name:
            return "sign"
        return None

    def process(self) -> None:
        """
        处理轨迹数据
        """
        if not self.check_dir(self.input_path):
            print(f"track目录不存在: {self.input_path}, 跳过处理")
            return

        track_info = {}
        track_camera_visible = {}
        trajectory = {}
        object_ids = {}
        track_vis_imgs = []

        extrinsics = data_io.load_extrinsics(self.output_root_path)
        intrinsics = data_io.load_intrinsics(self.output_root_path)

        for frame_id, f in enumerate(tqdm(sorted(os.listdir(self.input_path)))):
            object_file = os.path.join(self.input_path, f)
            with open(object_file, "r") as f:
                data = json.load(f)

            frame_name = f"{frame_id:06d}"
            track_info[frame_name] = {}
            track_camera_visible[frame_name] = {camera_id: [] for camera_id in CAMERA}

            images = data_io.load_images(self.image_output_path, frame_name)

            for j in data:
                track_id = j.get("track_id")
                center = j.get("box3d_center")
                label = self._map_category(j.get("label"))

                # 分配track_id
                if track_id not in object_ids:
                    object_ids[track_id] = len(object_ids)

                # box_info(主车坐标系下的数据)
                box_info = {
                    "height": j.get("box3d_size")[2],
                    "width": j.get("box3d_size")[1],
                    "length": j.get("box3d_size")[0],
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "heading": j.get("box3d_heading"),
                    "label": label,
                    "speed": 0.0,
                    "timestamp": data_io.load_timestamp(self.raw_input_path, frame_id),
                }

                # 保存track_info
                track_info[frame_name][track_id] = {
                    "lidar_box": box_info,
                    "camera_box": box_info,
                }

                # 保存trajectory_info
                if track_id not in trajectory:
                    trajectory[track_id] = {}
                trajectory[track_id][frame_name] = box_info

                # 检查相机可见性并绘制3D框
                for camera_id in CAMERA:
                    # 获取图像尺寸（如果图像存在）
                    img_shape = (1080, 1920)  # 默认尺寸
                    if camera_id in images:
                        img_shape = images[camera_id].shape[:2]  # (height, width)

                    if camera_id not in extrinsics or camera_id not in intrinsics:
                        visible, pts_2d = False, None
                    else:
                        visible, pts_2d = geometry.project_box_to_image(
                            box_info, extrinsics[camera_id], intrinsics[camera_id], img_shape
                        )

                    if visible:
                        track_camera_visible[frame_name][camera_id].append(track_id)

                    # 在前三个相机上绘制3D框
                    if (
                        camera_id in ["0", "1", "2"]
                        and camera_id in images
                        and pts_2d is not None
                    ):
                        # 画出外轮廓线
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
                            pt1, pt2 = pts_2d[i], pts_2d[j]
                            pt1 = (int(pt1[0]), int(pt1[1]))
                            pt2 = (int(pt2[0]), int(pt2[1]))

                            # 检查线段是否与图像区域有交集（更宽松的条件）
                            img_h, img_w = images[camera_id].shape[:2]

                            # 只要线段的任一端点在图像内，或者线段可能与图像边界相交就绘制
                            pt1_in = 0 <= pt1[0] < img_w and 0 <= pt1[1] < img_h
                            pt2_in = 0 <= pt2[0] < img_w and 0 <= pt2[1] < img_h

                            # 检查线段是否可能与图像区域相交
                            line_intersects = (
                                min(pt1[0], pt2[0]) < img_w
                                and max(pt1[0], pt2[0]) >= 0
                                and min(pt1[1], pt2[1]) < img_h
                                and max(pt1[1], pt2[1]) >= 0
                            )

                            if pt1_in or pt2_in or line_intersects:
                                cv2.line(
                                    images[camera_id],
                                    pt1,
                                    pt2,
                                    (255, 0, 0),
                                    2,
                                )
            # 生成可视化图像（前三个相机拼接）
            if all(camera_id in images for camera_id in ["0", "1", "2"]):
                track_vis_img = np.concatenate(
                    [images["0"], images["1"], images["2"]], axis=1
                )
                track_vis_imgs.append(track_vis_img)

        # 处理trajectory数据
        trajectory_info = {}
        for track_id, info in trajectory.items():
            if len(info) < 2:
                continue
            trajectory_info[track_id] = {}

            dims = []
            frames = []
            timestamps = []
            poses_vehicle = []  # 主车坐标系
            poses_world = []  # 世界坐标系
            speeds = []

            for frame_name, box in info.items():
                dims.append([box["height"], box["width"], box["length"]])
                frames.append(int(frame_name))
                timestamps.append(data_io.load_timestamp(self.raw_input_path, int(frame_name)))
                speeds.append(box["speed"])

                pose_vehicle = np.eye(4)
                pose_vehicle[:3, :3] = np.array([
                    [math.cos(box["heading"]), -math.sin(box["heading"]), 0],
                    [math.sin(box["heading"]), math.cos(box["heading"]), 0],
                    [0, 0, 1],
                ])
                pose_vehicle[:3, 3] = np.array([
                    box["center_x"],
                    box["center_y"],
                    box["center_z"],
                ])
                ego_pose = data_io.load_ego_pose(self.output_root_path, int(frame_name))
                pose_world = np.matmul(ego_pose, pose_vehicle)
                poses_vehicle.append(pose_vehicle.astype(np.float32))
                poses_world.append(pose_world.astype(np.float32))

            dims = np.array(dims).astype(np.float32)
            dim = np.max(dims, axis=0)
            poses_vehicle = np.array(poses_vehicle).astype(np.float32)
            poses_world = np.array(poses_world).astype(np.float32)

            # 计算是否动态
            positions = poses_world[:, :3, 3]
            distance = np.linalg.norm(positions[0] - positions[-1])
            dynamic = np.any(np.std(positions, axis=0) > 0.5) or distance > 2

            trajectory_info[track_id] = {
                "label": info[list(info.keys())[0]]["label"],
                "height": dim[0],
                "width": dim[1],
                "length": dim[2],
                "poses_vehicle": poses_vehicle,
                "timestamps": timestamps,
                "frames": frames,
                "speeds": speeds,
                "symmetric": info[list(info.keys())[0]]["label"] != "pedestrain",
                "deformable": info[list(info.keys())[0]]["label"] == "pedestrain",
                "stationary": not dynamic,
            }

        if track_vis_imgs:
            imageio.mimwrite(
                os.path.join(self.output_path, "track_vis.mp4"), track_vis_imgs, fps=24
            )
        with open(os.path.join(self.output_path, "track_info.pkl"), "wb") as f:
            pickle.dump(track_info, f)
        with open(os.path.join(self.output_path, "track_ids.json"), "w") as f:
            json.dump(object_ids, f, indent=4)
        with open(
            os.path.join(self.output_path, "track_camera_visible.pkl"), "wb"
        ) as f:
            pickle.dump(track_camera_visible, f)
        with open(os.path.join(self.output_path, "trajectory.pkl"), "wb") as f:
            pickle.dump(trajectory_info, f)

    
