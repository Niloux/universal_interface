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

        self.input_path = os.path.join(self.config["input"], "objects")
        self.output_path = os.path.join(self.config["output"], "track")

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

        for frame_id, f in enumerate(tqdm(sorted(os.listdir(self.input_path)))):
            object_file = os.path.join(self.input_path, f)
            with open(object_file, "r") as f:
                data = json.load(f)

            frame_name = f"{frame_id:06d}"
            track_info[frame_name] = {}
            track_camera_visible[frame_name] = {camera_id: [] for camera_id in CAMERA}

            ego_pose = self._load_ego_pose(frame_id)
            images = self._load_images(frame_id)
            extrinsics = self._load_extrinsics()
            intrinsics = self._load_intrinsics()

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
                    "timestamp": self._load_timestamp(frame_id),
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

                    visible, pts_2d = self._check_camera_visible(
                        box_info, camera_id, extrinsics, intrinsics, img_shape
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
            speeds = []

            for frame_name, box in info.items():
                dims.append([box["height"], box["width"], box["length"]])
                frames.append(int(frame_name))
                timestamps.append(self._load_timestamp(int(frame_name)))
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
                poses_vehicle.append(pose_vehicle.astype(np.float32))

            dims = np.array(dims).astype(np.float32)
            dim = np.max(dims, axis=0)
            poses_vehicle = np.array(poses_vehicle).astype(np.float32)

            # 计算是否动态
            positions = poses_vehicle[:, :3, 3]
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

    def _check_camera_visible(
        self,
        box_info: Dict[str, Any],
        camera_id: str,
        extrinsics: Dict[str, np.ndarray],
        intrinsics: Dict[str, np.ndarray],
        img_shape: tuple = (1080, 1920),
    ) -> tuple[bool, Optional[np.ndarray]]:
        """
        检查3D边界框在指定相机中是否可见，并返回2D投影点

        注意：由于外参是sensor to vehicle，直接从车辆坐标系转换到相机坐标系

        Args:
            box_info: 3D边界框信息，包含中心点、尺寸、朝向等（车辆坐标系）
            camera_id: 相机ID
            extrinsics: 相机外参字典 (sensor to vehicle)
            intrinsics: 相机内参字典
            img_shape: 图像尺寸 (height, width)，默认为 (1080, 1920)

        Returns:
            tuple: (是否可见, 2D投影点数组)
                - 是否可见: bool，表示边界框是否在相机视野内
                - 2D投影点: np.ndarray shape (8, 2)，3D边界框8个顶点的2D投影坐标
        """
        try:
            # 检查相机参数是否存在
            if camera_id not in extrinsics or camera_id not in intrinsics:
                return False, None

            # 获取3D边界框的8个顶点（主车坐标系）
            corners_3d = self._get_box_corners_3d(box_info)

            # 转换到相机坐标系
            corners_3d_camera = self._transform_to_camera(
                corners_3d, extrinsics[camera_id]
            )

            # 检查是否在相机前方（z > 0）
            z_values = corners_3d_camera[:, 2]
            if np.any(z_values <= 0):
                return False, None

            # 投影到2D图像平面
            pts_2d = self._project_to_2d(corners_3d_camera, intrinsics[camera_id])

            # 检查投影结果是否有效
            if pts_2d is None or np.any(np.isnan(pts_2d)) or np.any(np.isinf(pts_2d)):
                return False, None

            # 获取图像尺寸
            img_height, img_width = img_shape

            # 检查是否有点在图像范围内
            x_in_range = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_width)
            y_in_range = (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_height)
            points_in_image = x_in_range & y_in_range

            # 如果至少有一个点在图像内，则认为可见
            visible = bool(np.any(points_in_image))

            return visible, pts_2d.astype(int) if visible else None

        except Exception as e:
            print(f"检查相机可见性时出错: {e}")
            return False, None

    def _get_box_corners_3d(self, box_info: Dict[str, Any]) -> np.ndarray:
        """
        获取3D边界框的8个顶点坐标（主车坐标系）

        Args:
            box_info: 边界框信息

        Returns:
            np.ndarray: shape (8, 3)，8个顶点的3D坐标
        """
        # 获取边界框参数
        cx, cy, cz = box_info["center_x"], box_info["center_y"], box_info["center_z"]
        length, width, height = (
            box_info["length"],
            box_info["width"],
            box_info["height"],
        )
        heading = box_info["heading"]

        # 定义边界框的8个顶点（相对于中心点）
        # 顶点顺序：前下左、前下右、后下右、后下左、前上左、前上右、后上右、后上左
        x_corners = np.array([
            length / 2,
            length / 2,
            -length / 2,
            -length / 2,
            length / 2,
            length / 2,
            -length / 2,
            -length / 2,
        ])
        y_corners = np.array([
            -width / 2,
            width / 2,
            width / 2,
            -width / 2,
            -width / 2,
            width / 2,
            width / 2,
            -width / 2,
        ])
        z_corners = np.array([
            -height / 2,
            -height / 2,
            -height / 2,
            -height / 2,
            height / 2,
            height / 2,
            height / 2,
            height / 2,
        ])

        # 组合成顶点矩阵
        corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

        # 应用旋转（绕z轴）
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rotation_matrix = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])

        # 旋转顶点
        corners = rotation_matrix @ corners  # (3, 8)

        # 平移到实际位置
        corners[0, :] += cx
        corners[1, :] += cy
        corners[2, :] += cz

        return corners.T  # (8, 3)

    def _transform_to_camera(
        self, points: np.ndarray, extrinsics: np.ndarray
    ) -> np.ndarray:
        """
        将点从车辆坐标系转换到相机坐标系

        注意：外参是sensor to vehicle，所以需要求逆来从vehicle转到sensor

        Args:
            points: shape (N, 3)，车辆坐标系下的点
            extrinsics: shape (4, 4)，相机外参矩阵 (sensor to vehicle)

        Returns:
            np.ndarray: shape (N, 3)，相机坐标系下的点
        """
        # 转换为齐次坐标
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)

        # 应用逆变换矩阵（从车辆坐标系到相机坐标系）
        extrinsics_inv = np.linalg.inv(extrinsics)
        points_camera_homo = (extrinsics_inv @ points_homo.T).T  # (N, 4)

        # 返回3D坐标
        return points_camera_homo[:, :3]

    def _project_to_2d(
        self, points_3d: np.ndarray, intrinsics: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        将3D点投影到2D图像平面

        Args:
            points_3d: shape (N, 3)，相机坐标系下的3D点
            intrinsics: shape (3, 3)，相机内参矩阵

        Returns:
            np.ndarray: shape (N, 2)，2D图像坐标，如果投影失败返回None
        """
        try:
            # 投影到图像平面
            points_2d_homo = (intrinsics @ points_3d.T).T  # (N, 3)

            # 检查z坐标是否为零或接近零
            z_coords = points_2d_homo[:, 2]
            if np.any(np.abs(z_coords) < 1e-8):
                return None

            # 归一化得到像素坐标
            points_2d = points_2d_homo[:, :2] / z_coords.reshape(-1, 1)  # (N, 2)

            return points_2d
        except Exception as e:
            print(f"2D投影时出错: {e}")
            return None
