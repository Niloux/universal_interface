#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Mask数据处理模块

该模块负责根据动态物体的轨迹，生成每个相机视角下的2D掩码图像。
"""

import os
import pickle
from typing import Any, Dict

import cv2
import numpy as np
from tqdm import tqdm

from .base import BaseProcessor
from utils import default_logger

# 相机ID列表
CAMERA_IDS = ["0", "1", "2", "3", "4"]


class DynamicMaskProcessor(BaseProcessor):
    """
    动态物体掩码处理器

    负责根据轨迹信息，为动态物体生成并保存2D掩码。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化DynamicMaskProcessor

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.output_path = config["output"]
        self.track_output_path = os.path.join(self.output_path, "track")
        self.mask_output_path = os.path.join(self.output_path, "dynamic_mask")
        self.image_output_path = os.path.join(self.output_path, "images")
        self.ensure_dir(self.mask_output_path)

        # 默认图像尺寸，作为无法加载图像时的备用
        self.default_image_shape = (1080, 1920)  # (height, width)

    def process(self) -> bool:
        """
        处理所有帧，生成动态掩码

        Returns:
            bool: 处理是否成功
        """
        try:
            default_logger.info("开始生成动态物体掩码...")

            # 1. 加载所需数据
            trajectory_data = self._load_pickle(os.path.join(self.track_output_path, "trajectory.pkl"))
            track_info_data = self._load_pickle(os.path.join(self.track_output_path, "track_info.pkl"))

            if not trajectory_data or not track_info_data:
                default_logger.warning("缺少轨迹或跟踪信息文件，跳过动态掩码生成。")
                return False

            extrinsics = self._load_extrinsics()
            intrinsics = self._load_intrinsics()

            # 2. 识别所有动态物体的track_id
            dynamic_track_ids = {
                track_id
                for track_id, data in trajectory_data.items()
                if not data.get("stationary", True)
            }
            default_logger.info(f"共识别出 {len(dynamic_track_ids)} 个动态物体。")

            # 3. 逐帧生成掩码
            frame_names = sorted(track_info_data.keys())
            for frame_name in tqdm(frame_names, desc="生成动态掩码"):
                frame_objects = track_info_data[frame_name]
                images = self._load_images(frame_name)

                for camera_id in CAMERA_IDS:
                    # 动态获取图像尺寸
                    img_shape = self.default_image_shape
                    if camera_id in images:
                        img_shape = images[camera_id].shape[:2]

                    # 创建空白掩码图像
                    mask_image = np.zeros(img_shape, dtype=np.uint8)

                    # 筛选出当前帧可见的动态物体
                    for track_id, box_info_dict in frame_objects.items():
                        if track_id in dynamic_track_ids:
                            box_info = box_info_dict.get("lidar_box")
                            if not box_info:
                                continue

                            # 投影3D框到2D图像
                            _, pts_2d = self._project_box_to_2d(
                                box_info, camera_id, extrinsics, intrinsics, img_shape
                            )

                            # 如果投影点有效，则在掩码上绘制填充多边形
                            if pts_2d is not None:
                                # 使用凸包来获得一个封闭的多边形
                                hull = cv2.convexHull(pts_2d)
                                cv2.drawContours(mask_image, [hull], -1, (255), thickness=cv2.FILLED)

                    # 保存掩码图像
                    output_filename = f"{frame_name}_{camera_id}.png"
                    output_filepath = os.path.join(self.mask_output_path, output_filename)
                    cv2.imwrite(output_filepath, mask_image)

            default_logger.success("动态物体掩码生成完成。")
            return True

        except Exception as e:
            default_logger.error(f"生成动态掩码时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _project_box_to_2d(
        self,
        box_info: Dict[str, Any],
        camera_id: str,
        extrinsics: Dict[str, np.ndarray],
        intrinsics: Dict[str, np.ndarray],
        img_shape: tuple,
    ) -> tuple[bool, Any]:
        """
        将单个3D边界框投影到指定相机的2D图像平面。
        (此函数逻辑复用自 core/track.py)
        """
        if camera_id not in extrinsics or camera_id not in intrinsics:
            return False, None

        corners_3d = self._get_box_corners_3d(box_info)

        # 转换到相机坐标系
        extrinsics_inv = np.linalg.inv(extrinsics[camera_id])
        points_homo = np.hstack([corners_3d, np.ones((corners_3d.shape[0], 1))])
        corners_3d_camera = (extrinsics_inv @ points_homo.T).T[:, :3]

        # 检查是否在相机前方
        if np.any(corners_3d_camera[:, 2] <= 0):
            return False, None

        # 投影到2D图像平面
        intrinsics_matrix = intrinsics[camera_id]
        points_2d_homo = (intrinsics_matrix @ corners_3d_camera.T).T

        z_coords = points_2d_homo[:, 2]
        if np.any(np.abs(z_coords) < 1e-8):
            return None, None

        pts_2d = points_2d_homo[:, :2] / z_coords.reshape(-1, 1)

        # 检查是否有任何部分在图像内
        img_height, img_width = img_shape
        x_in_range = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_width)
        y_in_range = (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_height)

        # 只要有任何一个角点在图像内，就认为可见
        if not np.any(x_in_range & y_in_range):
            return False, None

        return True, pts_2d.astype(int)

    def _get_box_corners_3d(self, box_info: Dict[str, Any]) -> np.ndarray:
        """
        获取3D边界框的8个顶点坐标（主车坐标系）。
        (此函数逻辑复用自 core/track.py)
        """
        cx, cy, cz = box_info["center_x"], box_info["center_y"], box_info["center_z"]
        length, width, height = box_info["length"], box_info["width"], box_info["height"]
        heading = box_info["heading"]

        x_corners = np.array([l / 2 for l in [length, length, -length, -length, length, length, -length, -length]])
        y_corners = np.array([w / 2 for w in [-width, width, width, -width, -width, width, width, -width]])
        z_corners = np.array([h / 2 for h in [-height, -height, -height, -height, height, height, height, height]])

        corners = np.vstack([x_corners, y_corners, z_corners])

        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rotation_matrix = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])

        corners = rotation_matrix @ corners
        corners[0, :] += cx
        corners[1, :] += cy
        corners[2, :] += cz

        return corners.T

    def _load_images(self, frame_name: str) -> Dict[str, np.ndarray]:
        """加载指定帧的所有相机图像"""
        images = {}
        for camera_id in CAMERA_IDS:
            img_path = os.path.join(self.image_output_path, f"{frame_name}_{camera_id}.png")
            if os.path.exists(img_path):
                images[camera_id] = cv2.imread(img_path)
        return images

    def _load_pickle(self, file_path: str) -> Any:
        """加载pickle文件"""
        if not os.path.exists(file_path):
            default_logger.error(f"文件不存在: {file_path}")
            return None
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def _load_extrinsics(self) -> Dict[str, np.ndarray]:
        """加载所有相机的外参"""
        extrinsics = {}
        for camera_id in CAMERA_IDS:
            ext_path = os.path.join(self.output_path, "extrinsics", f"{camera_id}.txt")
            if os.path.exists(ext_path):
                extrinsics[camera_id] = np.loadtxt(ext_path)
        return extrinsics

    def _load_intrinsics(self) -> Dict[str, np.ndarray]:
        """加载所有相机的内参"""
        intrinsics = {}
        for camera_id in CAMERA_IDS:
            int_path = os.path.join(self.output_path, "intrinsics", f"{camera_id}.txt")
            if os.path.exists(int_path):
                params = np.loadtxt(int_path)
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                intrinsics[camera_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return intrinsics
