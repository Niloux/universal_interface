#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera数据处理模块

该模块负责处理相机相关数据，包括：
1. 读取和转换相机内外参数据
2. 处理和转换图像数据

"""

import shutil
from pathlib import Path
from typing import List

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
        self.extrinsics_path = self.input_path / "extrinsics"
        self.intrinsics_path = self.input_path / "intrinsics"

        default_logger.info(f"初始化相机处理器，支持 {len(self.camera_positions)} 个相机位置")

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

    def _generate_camera_params(self, intrinsics_output_dir: Path, extrinsics_output_dir: Path):
        """
        生成相机参数文件

        Args:
            intrinsics_output_dir: 内参输出目录
            extrinsics_output_dir: 外参输出目录
        """
        for camera_position in self.camera_positions:
            camera_id = self.camera_id_map[camera_position]

            with open(self.extrinsics_path / f"{camera_id}.txt", "r") as f:
                extrinsics = f.read()
            with open(self.intrinsics_path / f"{camera_id}.txt", "r") as f:
                intrinsics = f.read()

            # 生成内参文件
            intrinsics_file = intrinsics_output_dir / f"{camera_id}.txt"
            self._write_intrinsics_file(intrinsics, intrinsics_file)

            # 生成外参文件
            extrinsics_file = extrinsics_output_dir / f"{camera_id}.txt"
            self._write_extrinsics_file(extrinsics, extrinsics_file)

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

    def _write_intrinsics_file(self, intrinsics: str, output_file: Path):
        """
        写入内参文件

        Args:
            intrinsics: 相机内参数据（一行空格分隔）
            output_file: 输出文件路径
        """
        intrinsics = [float(x) for x in intrinsics.strip().split()]

        fx, _, cx, _, fy, cy, _, _, _ = intrinsics[:9]
        _x = 0

        # 写入文件，每行一个参数
        with open(output_file, "w") as f:
            f.write(f"{fx:.18e}\n")
            f.write(f"{fy:.18e}\n")
            f.write(f"{cx:.18e}\n")
            f.write(f"{cy:.18e}\n")
            f.write(f"{_x:.18e}\n")
            f.write(f"{_x:.18e}\n")
            f.write(f"{_x:.18e}\n")
            f.write(f"{_x:.18e}\n")
            f.write(f"{_x:.18e}\n")

    def _write_extrinsics_file(self, extrinsics: str, output_file: Path):
        """
        写入外参文件

        Args:
            extrinsics: 相机外参数据（一行空格分隔）
            output_file: 输出文件路径
        """
        extrinsics = [float(x) for x in extrinsics.strip().split()]

        # 确保有16个数字可以构成4x4矩阵
        if len(extrinsics) != 16:
            raise ValueError(f"输入数据必须包含16个数字以构成4x4矩阵，当前数量: {len(extrinsics)}")

        # 构建4x4矩阵字符串
        matrix_str = ""
        for i in range(0, 16, 4):
            # 每行4个数字，保留6位小数，用空格分隔
            row = " ".join(f"{num:.6f}" for num in extrinsics[i : i + 4])
            matrix_str += row + "\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(matrix_str.strip())
