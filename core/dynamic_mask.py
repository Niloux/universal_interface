#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Mask数据处理模块

该模块负责根据动态物体的轨迹，生成每个相机视角下的2D掩码图像。
"""

import cv2
import numpy as np
from tqdm import tqdm

from .base import BaseProcessor
from utils import default_logger, data_io, geometry
from utils.config import Config


class DynamicMaskProcessor(BaseProcessor):
    """
    动态物体掩码处理器

    负责根据轨迹信息，为动态物体生成并保存2D掩码。
    """

    def __init__(self, config: Config):
        """
        初始化DynamicMaskProcessor

        Args:
            config: 配置管理对象
        """
        super().__init__(config)
        self.output_path = self.config.output
        self.track_output_path = self.output_path / "track"
        self.mask_output_path = self.output_path / "dynamic_mask"
        self.image_output_path = self.output_path / "images"
        self.ensure_dir(self.mask_output_path)

        self.camera_ids = [str(v) for v in self.config.camera.id_map.values()]

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
            trajectory_data = data_io.load_pickle(self.track_output_path / "trajectory.pkl")
            track_info_data = data_io.load_pickle(self.track_output_path / "track_info.pkl")

            if not trajectory_data or not track_info_data:
                default_logger.warning("缺少轨迹或跟踪信息文件，跳过动态掩码生成。")
                return False

            extrinsics = data_io.load_extrinsics(self.output_path, self.camera_ids)
            intrinsics = data_io.load_intrinsics(self.output_path, self.camera_ids)

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
                images = data_io.load_images(self.image_output_path, frame_name, self.camera_ids)

                for camera_id in self.camera_ids:
                    # 动态获取图像尺寸
                    img_shape = self.default_image_shape
                    if camera_id in images:
                        img_shape = images[camera_id].shape[:2]

                    # 如果相机参数不存在，则跳过
                    if camera_id not in extrinsics or camera_id not in intrinsics:
                        continue

                    # 创建空白掩码图像
                    mask_image = np.zeros(img_shape, dtype=np.uint8)

                    # 筛选出当前帧可见的动态物体
                    for track_id, box_info_dict in frame_objects.items():
                        if track_id in dynamic_track_ids:
                            box_info = box_info_dict.get("lidar_box")
                            if not box_info:
                                continue

                            # 投影3D框到2D图像
                            visible, pts_2d = geometry.project_box_to_image(
                                box_info, extrinsics[camera_id], intrinsics[camera_id], img_shape
                            )

                            # 如果投影点有效，则在掩码上绘制填充多边形
                            if visible and pts_2d is not None:
                                # 使用凸包来获得一个封闭的多边形
                                hull = cv2.convexHull(pts_2d)
                                cv2.drawContours(mask_image, [hull], -1, (255), thickness=cv2.FILLED)

                    # 保存掩码图像
                    output_filename = f"{frame_name}_{camera_id}.png"
                    output_filepath = self.mask_output_path / output_filename
                    cv2.imwrite(str(output_filepath), mask_image)

            default_logger.success("动态物体掩码生成完成。")
            return True

        except Exception as e:
            default_logger.error(f"生成动态掩码时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False