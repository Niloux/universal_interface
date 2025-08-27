#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point Cloud数据处理模块

该模块负责提取点云的深度信息以及分离动态和静态的点云
"""

import pathlib
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict
from plyfile import PlyData

from .base import BaseProcessor
from utils import default_logger, data_io
from utils.config import Config
from utils.structures import TrajectoryData


class PointCloudProcessor(BaseProcessor):
    """
    Lidar点云处理器

    提取点云的深度信息以及分离动态和静态的点云
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_path = self.config.output
        self.lidar_output_path = self.output_path / "lidar"
        self.actor_output_path = self.lidar_output_path / "actor"
        self.background_output_path = self.lidar_output_path / "background"
        self.depth_output_path = self.lidar_output_path / "depth"
        self.ensure_dir(self.actor_output_path)
        self.ensure_dir(self.background_output_path)
        self.ensure_dir(self.depth_output_path)

        self.camera_ids = [str(v) for v in self.config.camera.id_map.values()]

    def process(self) -> bool:
        try:
            default_logger.info("开始生成点云数据...")

            # 1. 加载所需数据
            extrinsics = data_io.load_extrinsics(self.output_path, self.camera_ids)
            intrinsics = data_io.load_intrinsics(self.output_path, self.camera_ids)

            trajectory: Dict[str, TrajectoryData] = data_io.load_pickle(self.output_path / "track" / "trajectory.pkl")

            # 为动态物体创建目录
            pointcloud_actor = {}
            for track_id, traj in trajectory.items():
                dynamic = not traj.stationary
                if dynamic and traj.label != "sign":
                    self.ensure_dir(self.actor_output_path / track_id)
                    pointcloud_actor[track_id] = {"xyz": [], "rgb": [], "mask": []}
            default_logger.info(f"加载了{len(pointcloud_actor)}个actor")

            # 2. 处理lidar数据
            point_clouds_path = self.config.input / "point_clouds"
            if not self.check_dir(point_clouds_path):
                default_logger.error("点云输入目录为空")
                return False

            lidar_names = [f.name for f in point_clouds_path.iterdir() if f.is_dir()]
            default_logger.info(f"开始处理点云数据，发现{len(lidar_names)}个激光雷达: {lidar_names}")

            first_lidar = lidar_names[0]
            first_lidar_path = point_clouds_path / first_lidar
            frame_files = sorted([f.stem for f in first_lidar_path.iterdir() if f.suffix == ".ply"])
            default_logger.info(f"总共需要处理 {len(frame_files)} 帧")

            for frame_name in tqdm(frame_files):
                # 收集当前帧的所有点云数据
                all_xyzs = []
                for lidar_name in lidar_names:
                    lidar_points_path = point_clouds_path / lidar_name
                    points_file_path = lidar_points_path / f"{frame_name}.ply"

                    if not point_clouds_path.exists:
                        continue
                    # 读取点云数据
                    plydata = PlyData.read(points_file_path)
                    vertices = plydata["vertex"].data
                    points_3d = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
                    all_xyzs.append(points_3d)

                if not all_xyzs:
                    continue

                # 合并所有lidar的数据
                xyzs = np.concatenate(all_xyzs, axis=0)
                xyzs_for_depth = xyzs  # 用于深度图生成的点云数据
                # 初始化RGB为零（黑色，用于actor/background分离）
                rgbs = np.zeros((xyzs.shape[0], 3), dtype=np.uint8)
                masks = np.ones(xyzs.shape[0], dtype=bool)  # 所有点都有效

                # 为每个相机处理深度图和RGB颜色
                images = data_io.load_images(self.output_path / "images", frame_name, self.camera_ids)
                for i in self.camera_ids:
                    # === 深度图 ===
                    vehicle_to_cam = np.linalg.inv(extrinsics[i])
                    image_height, image_width = images[i].shape[:2]

                    depth = self._generate_depth_map(
                        xyzs_for_depth, vehicle_to_cam, intrinsics[i], image_width, image_height
                    )

                    # 保存深度图
                    self._save_depth_map(depth, self.depth_output_path, frame_name, i, image_width, image_height)

                    # 可视化深度图（仅对第一个相机）
                    if i == "0":
                        self._generate_depth_visualization(
                            depth, self.depth_output_path, frame_name, i, image_width, image_height, images[i]
                        )

                    # === RGB颜色获取部分（用于actor/background分离） ===
                    self._assign_rgb_colors(
                        xyzs, rgbs, vehicle_to_cam, intrinsics[i], image_width, image_height, images[i]
                    )

                # === Actor/Background分离部分 ===
                self._separate_actors_and_background(
                    xyzs,
                    rgbs,
                    masks,
                    trajectory,
                    pointcloud_actor,
                    self.actor_output_path,
                    self.background_output_path,
                    frame_name,
                )

        except Exception as e:
            default_logger.error(f"生成点云时发生错误：{e}")
            import traceback

            traceback.print_exc()
            return False

    def _generate_depth_map(
        self,
        xyzs: np.ndarray,
        vehicle_to_cam: np.ndarray,
        intrinsic: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """生成深度图

        Args:
            xyzs: 点云坐标
            vehicle_to_cam: 车辆坐标系到相机坐标系的变换矩阵
            intrinsic: 相机内参矩阵
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            深度图数组
        """
        # 初始化深度图
        depth = (np.ones((image_height, image_width)) * np.finfo(np.float32).max).reshape(-1)

        # 转换为齐次坐标
        xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)

        # 将点云从vehicle坐标系转换到相机坐标系
        xyzs_cam = xyzs_homo @ vehicle_to_cam.T

        # 获取相机坐标系下的深度（Z轴坐标）
        xyzs_depth = xyzs_cam[..., 2]

        # 过滤掉深度为负或过小的点
        valid_depth_mask = xyzs_depth > 1e-1
        if valid_depth_mask.sum() > 0:
            # 只使用有效深度的点
            valid_xyzs_cam = xyzs_cam[valid_depth_mask]
            valid_depths = xyzs_depth[valid_depth_mask]
            # 移除深度上限限制，保留所有有效深度值
            valid_depths = np.clip(valid_depths, a_min=1e-1, a_max=None)

            # 投影到像素坐标
            pixel_coords = valid_xyzs_cam[:, :3] @ intrinsic.T
            pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2:3]

            # 转换为整数像素坐标
            u_depth = pixel_coords[:, 0].astype(np.int32)
            v_depth = pixel_coords[:, 1].astype(np.int32)

            # 确保坐标在图像范围内
            valid_pixel_mask = (u_depth >= 0) & (u_depth < image_width) & (v_depth >= 0) & (v_depth < image_height)

            if valid_pixel_mask.sum() > 0:
                # 只使用在图像范围内的点
                final_u = u_depth[valid_pixel_mask]
                final_v = v_depth[valid_pixel_mask]
                final_depths = valid_depths[valid_pixel_mask]

                # 计算扁平化索引
                indices = final_v * image_width + final_u

                # 使用最小深度值填充（waymo标准做法）
                np.minimum.at(depth, indices, final_depths)

        return depth

    def _save_depth_map(
        self,
        depth: np.ndarray,
        depth_output_path: pathlib.Path,
        frame_name: str,
        camera_id: str,
        image_width: int,
        image_height: int,
    ):
        """保存深度图

        Args:
            depth: 深度图数组
            depth_output_path: 深度图输出路径
            frame_name: 帧编号
            camera_id: 相机编号
            image_width: 图像宽度
            image_height: 图像高度
        """
        # 处理深度图
        depth[depth >= np.finfo(np.float32).max - 1e-5] = 0
        valid_depth_pixel = depth != 0
        valid_depth_value = depth[valid_depth_pixel].astype(np.float32)
        valid_depth_pixel = valid_depth_pixel.reshape(image_height, image_width).astype(np.bool_)

        # 保存深度图
        depth_filename = depth_output_path / f"{frame_name}_{camera_id}.npz"
        np.savez_compressed(depth_filename, mask=valid_depth_pixel, value=valid_depth_value)

    def _generate_depth_visualization(
        self,
        depth: np.ndarray,
        depth_vis_output_path: pathlib.Path,
        frame_name: str,
        camera_id: str,
        image_width: int,
        image_height: int,
        image: np.ndarray,
    ):
        """生成深度图可视化

        Args:
            depth: 深度图数组
            depth_vis_output_path: 深度图可视化输出路径
            frame_name: 帧编号
            camera_id: 相机编号
            image_width: 图像宽度
            image_height: 图像高度
            image: 图像内容
        """

        def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
            """
            depth: (H, W)
            """
            x = np.nan_to_num(depth)  # change nan to 0
            if minmax is None:
                mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
                ma = np.max(x)
            else:
                mi, ma = minmax
            x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
            x = (255 * x).astype(np.uint8)
            x_ = cv2.applyColorMap(x, cmap)
            return x_, [mi, ma]

        try:
            depth_2d = depth.reshape(image_height, image_width).astype(np.float32)
            depth_vis, _ = visualize_depth_numpy(depth_2d)

            # 图像和深度图应该已经是相同尺寸
            depth_on_img = image.copy()
            valid_mask = depth_2d > 0
            depth_on_img[valid_mask] = depth_vis[valid_mask]

            depth_vis_filename = depth_vis_output_path / f"{frame_name}_{camera_id}.png"
            cv2.imwrite(depth_vis_filename, depth_on_img[..., [2, 1, 0]])

        except Exception as e:
            default_logger.error(f"深度图可视化失败 {frame_name}_{camera_id}: {e}")

    def _assign_rgb_colors(
        self,
        xyzs: np.ndarray,
        rgbs: np.ndarray,
        vehicle_to_cam: np.ndarray,
        intrinsic: np.ndarray,
        image_width: int,
        image_height: int,
        image: np.ndarray,
    ):
        """为点云分配RGB颜色

        Args:
            xyzs: 点云坐标
            rgbs: RGB颜色数组（会被修改）
            vehicle_to_cam: 车辆坐标系到相机坐标系的变换矩阵
            intrinsic: 相机内参矩阵
            image_width: 图像宽度
            image_height: 图像高度
            image: 图像内容
        """
        # 将原始点云投影到相机图像
        xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
        xyzs_cam = xyzs_homo @ vehicle_to_cam.T

        # 过滤有效深度的点
        valid_depth_mask = xyzs_cam[:, 2] > 1e-1
        if valid_depth_mask.sum() > 0:
            valid_xyzs_cam = xyzs_cam[valid_depth_mask]

            # 投影到像素坐标
            pixel_coords = valid_xyzs_cam[:, :3] @ intrinsic.T
            pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2:3]

            u_rgb = pixel_coords[:, 0].astype(np.int32)
            v_rgb = pixel_coords[:, 1].astype(np.int32)

            # 确保坐标在图像范围内
            valid_pixel_mask = (u_rgb >= 0) & (u_rgb < image_width) & (v_rgb >= 0) & (v_rgb < image_height)

            if valid_pixel_mask.sum() > 0:
                # 获取有效像素的RGB值
                valid_u = u_rgb[valid_pixel_mask]
                valid_v = v_rgb[valid_pixel_mask]
                valid_indices = np.where(valid_depth_mask)[0][valid_pixel_mask]

                rgb_colors = image[valid_v, valid_u]
                rgbs[valid_indices] = rgb_colors

    def _separate_actors_and_background(
        self,
        xyzs: np.ndarray,
        rgbs: np.ndarray,
        masks: np.ndarray,
        trajectory: Dict[str, TrajectoryData],
        pointcloud_actor: Dict,
        lidar_dir_actor: pathlib.Path,
        lidar_dir_background: pathlib.Path,
        frame_name: str,
    ):
        """分离Actor和Background点云

        Args:
            xyzs: 点云坐标(主车坐标系)
            rgbs: RGB颜色数组
            masks: 掩码数组
            trajectory: 轨迹信息
            pointcloud_actor: Actor点云数据
            lidar_dir_actor: Actor输出目录
            lidar_dir_background: Background输出目录
            frame_name: 帧编号
        """
        # 创建actor mask
        actor_mask = np.zeros(xyzs.shape[0], dtype=bool)

        # 对每个track处理
        for track_id, traj_info in trajectory.items():
            if track_id not in pointcloud_actor:
                continue

            # 检查当前帧是否有该track
            frames = traj_info.frames
            if int(frame_name) not in frames:
                continue

            frame_idx_in_traj = frames.index(int(frame_name))
            poses_vehicle = np.array(traj_info.poses_vehicle)

            if frame_idx_in_traj >= len(poses_vehicle):
                continue

            pose_vehicle = poses_vehicle[frame_idx_in_traj]  # 主车坐标系下的actor位姿

            # 获取边界框尺寸
            length = traj_info.length or 1.0
            width = traj_info.width or 1.0
            height = traj_info.height or 1.0

            # 将点云转换到物体坐标系
            xyzs_homo = np.concatenate([xyzs, np.ones_like(xyzs[..., :1])], axis=-1)
            xyzs_actor = xyzs_homo @ np.linalg.inv(pose_vehicle).T
            xyzs_actor = xyzs_actor[..., :3]

            # 创建3D边界框
            bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5

            # 判断点是否在边界框内
            inbbox_mask = (
                (xyzs_actor[:, 0] >= bbox[0, 0])
                & (xyzs_actor[:, 0] <= bbox[1, 0])
                & (xyzs_actor[:, 1] >= bbox[0, 1])
                & (xyzs_actor[:, 1] <= bbox[1, 1])
                & (xyzs_actor[:, 2] >= bbox[0, 2])
                & (xyzs_actor[:, 2] <= bbox[1, 2])
            )

            actor_mask = np.logical_or(actor_mask, inbbox_mask)

            # 保存该track的点云
            if inbbox_mask.sum() > 0:
                xyzs_inbbox = xyzs_actor[inbbox_mask]
                rgbs_inbbox = rgbs[inbbox_mask]
                masks_inbbox = masks[inbbox_mask]

                pointcloud_actor[track_id]["xyz"].append(xyzs_inbbox)
                pointcloud_actor[track_id]["rgb"].append(rgbs_inbbox)
                pointcloud_actor[track_id]["mask"].append(masks_inbbox)

                # 保存单帧actor点云
                masks_inbbox_expanded = masks_inbbox[..., None]
                ply_actor_path = lidar_dir_actor / track_id / f"{frame_name}.ply"
                try:
                    data_io.storePly(
                        ply_actor_path,
                        xyzs_inbbox,
                        rgbs_inbbox,
                        masks_inbbox_expanded,
                    )
                except Exception as e:
                    default_logger.error(f"保存actor点云失败 {track_id}/{frame_name}: {e}")

        # 保存background点云
        xyzs_background = xyzs[~actor_mask]
        rgbs_background = rgbs[~actor_mask]
        masks_background = masks[~actor_mask]
        masks_background_expanded = masks_background[..., None]

        ply_background_path = lidar_dir_background / f"{frame_name}.ply"
        try:
            data_io.storePly(
                ply_background_path,
                xyzs_background,
                rgbs_background,
                masks_background_expanded,
            )
        except Exception as e:
            default_logger.error(f"保存background点云失败 {frame_name}: {e}")
