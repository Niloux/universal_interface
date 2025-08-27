#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera数据处理模块

该模块负责处理相机相关数据，包括：
1. 读取和转换相机内外参数据
2. 处理和转换图像数据

"""

import json
import shutil
from typing import Any, Dict, List
from pathlib import Path

from tqdm import tqdm

from utils import default_logger
from utils.config import Config
from .base import BaseProcessor


class CameraProcessor(BaseProcessor):
    """
    相机数据处理器

    负责处理相机内外参和图像数据的读取、转换和输出
    """

    def __init__(self, config: Config):
        """
        初始化相机处理器

        Args:
            config: 配置管理对象
        """
        super().__init__(config)
        self.camera_positions = self.config.camera.positions
        self.camera_id_map = self.config.camera.id_map

        self.input_path = self.config.input
        self.output_path = self.config.output

        # 定义输入路径
        self.images_path = self.input_path / "images"
        self.metadata_path = self.input_path / "images_metadata"

        default_logger.info(
            f"初始化相机处理器，支持 {len(self.camera_positions)} 个相机位置"
        )

    def process(self) -> bool:
        """
        处理所有相机数据

        Returns:
            bool: 处理是否成功
        """
        try:
            default_logger.info(f"开始处理 {len(self.camera_positions)} 个相机的数据")

            # 创建输出目录
            images_output_dir = self.output_path / "images"
            intrinsics_output_dir = self.output_path / "intrinsics"
            extrinsics_output_dir = self.output_path / "extrinsics"

            self.ensure_dir(images_output_dir)
            self.ensure_dir(intrinsics_output_dir)
            self.ensure_dir(extrinsics_output_dir)

            # 首先生成相机参数文件（每个相机一个文件）
            self._generate_camera_params(intrinsics_output_dir, extrinsics_output_dir)

            # 然后处理图像文件
            self._process_images(images_output_dir)

            default_logger.success("所有相机数据处理完成")
            return True

        except Exception as e:
            default_logger.error(f"相机数据处理失败: {e}")
            return False

    def _generate_camera_params(
        self, intrinsics_output_dir: Path, extrinsics_output_dir: Path
    ):
        """
        生成相机参数文件

        Args:
            intrinsics_output_dir: 内参输出目录
            extrinsics_output_dir: 外参输出目录
        """
        for camera_position in self.camera_positions:
            camera_id = self.camera_id_map[camera_position]

            # 获取第一个元数据文件作为参考（假设相机参数在整个序列中是固定的）
            metadata_files = self._get_metadata_files(camera_position)
            if not metadata_files:
                continue

            # 读取第一个元数据文件
            with open(metadata_files[0], "r") as f:
                metadata = json.load(f)

            # 生成内参文件
            intrinsics_file = intrinsics_output_dir / f"{camera_id}.txt"
            self._write_intrinsics_file(metadata, intrinsics_file)

            # 生成外参文件
            extrinsics_file = extrinsics_output_dir / f"{camera_id}.txt"
            self._write_extrinsics_file(metadata, extrinsics_file)

        default_logger.info("相机参数文件生成完成")

    def _get_image_files(self, camera_position: str) -> List[Path]:
        """
        获取图像文件列表

        Args:
            camera_position: 相机位置

        Returns:
            图像文件路径列表
        """
        images_dir = self.images_path / camera_position
        if not images_dir.exists():
            return []

        image_files = []
        for item in images_dir.iterdir():
            if item.is_file() and item.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_files.append(item)

        return sorted(image_files)

    def _get_metadata_files(self, camera_position: str) -> List[Path]:
        """
        获取元数据文件列表

        Args:
            camera_position: 相机位置

        Returns:
            元数据文件路径列表
        """
        metadata_dir = self.metadata_path / camera_position
        if not metadata_dir.exists():
            return []

        metadata_files = []
        for item in metadata_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".json":
                metadata_files.append(item)

        return sorted(metadata_files)

    def _process_images(self, images_output_dir: Path):
        """
        处理图像文件，按照帧号_相机ID.png的格式命名

        Args:
            images_output_dir: 图像输出目录
        """
        # 获取所有帧的数量（以第一个相机为准）
        first_camera = self.camera_positions[0]
        image_files = self._get_image_files(first_camera)
        total_frames = len(image_files)

        default_logger.info(f"开始处理图像文件，共 {total_frames} 帧")

        for frame_idx in tqdm(range(total_frames), desc="处理图像"):
            frame_name = f"{frame_idx:06d}"

            for camera_position in self.camera_positions:
                camera_id = self.camera_id_map[camera_position]

                # 源图像文件
                input_dir = self.images_path / camera_position
                source_file = input_dir / f"{frame_name}.jpg"

                if source_file.exists():
                    # 目标文件名：帧号_相机ID.png
                    output_file = images_output_dir / f"{frame_name}_{camera_id}.png"

                    try:
                        # 复制并转换格式（如果需要）
                        shutil.copy2(source_file, output_file)
                    except Exception as e:
                        default_logger.error(f"处理图像文件 {source_file} 时出错: {e}")

    def _write_intrinsics_file(self, metadata: Dict[str, Any], output_file: Path):
        """
        写入内参文件

        Args:
            metadata: 元数据字典
            output_file: 输出文件路径
        """
        intrinsics = metadata.get("intrinsics", [])

        # 如果是列表格式 [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        if isinstance(intrinsics, list) and len(intrinsics) >= 9:
            fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics[:9]
        else:
            # 如果是字典格式，提取参数
            fx = intrinsics.get("focal_length", {}).get("fx", 0.0)
            fy = intrinsics.get("focal_length", {}).get("fy", 0.0)
            cx = intrinsics.get("principal_point", {}).get("cx", 0.0)
            cy = intrinsics.get("principal_point", {}).get("cy", 0.0)

            distortion = intrinsics.get("distortion_coefficients", {})
            k1 = distortion.get("k1", 0.0)
            k2 = distortion.get("k2", 0.0)
            p1 = distortion.get("p1", 0.0)
            p2 = distortion.get("p2", 0.0)
            k3 = distortion.get("k3", 0.0)

        # 写入文件，每行一个参数
        with open(output_file, "w") as f:
            f.write(f"{fx:.18e}\n")
            f.write(f"{fy:.18e}\n")
            f.write(f"{cx:.18e}\n")
            f.write(f"{cy:.18e}\n")
            f.write(f"{k1:.18e}\n")
            f.write(f"{k2:.18e}\n")
            f.write(f"{p1:.18e}\n")
            f.write(f"{p2:.18e}\n")
            f.write(f"{k3:.18e}\n")

    def _write_extrinsics_file(self, metadata: Dict[str, Any], output_file: Path):
        """
        写入外参文件

        Args:
            metadata: 元数据字典
            output_file: 输出文件路径
        """
        extrinsics = metadata.get("extrinsics", [])

        # 如果是列表格式（4x4矩阵）
        if isinstance(extrinsics, list) and len(extrinsics) == 4:
            transform_matrix = extrinsics
        else:
            # 如果是字典格式，提取变换矩阵
            transform_matrix = extrinsics.get("transformation_matrix", [])

        # 写入文件，4x4矩阵格式
        with open(output_file, "w") as f:
            for row in transform_matrix:
                row_str = " ".join([f"{val:.18e}" for val in row])
                f.write(f"{row_str}\n")