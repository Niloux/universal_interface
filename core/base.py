"""处理器基类模块

提供数据处理器的抽象基类和通用功能。
所有具体的数据处理器都应该继承BaseProcessor类。
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import cv2
import numpy as np

from utils.config import Config


class BaseProcessor(ABC):
    """数据处理器抽象基类

    定义了数据处理器的通用接口和工具方法。
    所有具体的处理器都必须继承此类并实现process方法。

    Attributes:
        config: 配置管理对象，包含所有配置信息
    """

    def __init__(self, config: Config) -> None:
        """初始化处理器

        Args:
            config: 配置管理对象
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def process(self) -> None:
        """处理数据的抽象方法

        子类必须实现此方法来定义具体的数据处理逻辑。
        """
        pass

    def get_folders(self, path: str) -> List[str]:
        """获取指定路径下的所有文件夹名称

        Args:
            path: 要扫描的目录路径

        Returns:
            文件夹名称列表

        Raises:
            OSError: 当路径不存在或无法访问时
        """
        if not os.path.exists(path):
            raise OSError(f"路径不存在: {path}")

        folders = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                folders.append(item)
        return folders

    def ensure_dir(self, path: str) -> None:
        """确保目录存在，如果不存在则创建

        Args:
            path: 要创建的目录路径

        Raises:
            OSError: 当无法创建目录时
        """
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                raise OSError(f"无法创建目录 {path}: {e}")

    def check_dir(self, path: str) -> bool:
        """检查目录是否存在

        Args:
            path: 要检查的目录路径

        Returns:
            目录存在返回True，否则返回False
        """
        return os.path.exists(path) and os.path.isdir(path)

    def _load_ego_pose(self, frame_id: int) -> np.array:
        """
        加载指定帧的ego pose数据
        """
        path = os.path.join(self.config["output"], "ego_pose", f"{frame_id:06d}.txt")
        return np.loadtxt(path)

    def _load_extrinsics(self) -> np.array:
        """
        加载相机外参
        """
        camera_names = []
        path = os.path.join(self.config["output"], "extrinsics")
        for f in os.listdir(path):
            if f.endswith(".txt"):
                camera_names.append(f.split(".")[0])
        extrinsics = {}
        for camera_name in camera_names:
            extrinsics[camera_name] = np.loadtxt(
                os.path.join(path, f"{camera_name}.txt")
            )
        return extrinsics

    def _load_intrinsics(self) -> np.array:
        """
        加载相机内参
        """
        camera_names = []
        path = os.path.join(self.config["output"], "intrinsics")
        for f in os.listdir(path):
            if f.endswith(".txt"):
                camera_names.append(f.split(".")[0])
        intrinsics = {}
        for camera_name in camera_names:
            intrinsics_file = os.path.join(path, f"{camera_name}.txt")
            with open(intrinsics_file, "r") as f:
                values = [float(line.strip()) for line in f.readlines()]
            fx, fy, cx, cy = values[:4]
            intrinsics[camera_name] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ])
        return intrinsics

    def _load_images(self, frame_id: int) -> Dict[str, Any]:
        """
        加载指定帧的图像数据
        """
        path = os.path.join(self.config["output"], "images")
        images = {}
        for i in os.listdir(path):
            if i.endswith(".png") and int(i.split(".")[0].split("_")[0]) == frame_id:
                images[i.split(".")[0].split("_")[-1]] = cv2.imread(
                    os.path.join(path, i)
                )
        return images

    def _load_timestamp(self, frame_id: int) -> str:
        """
        加载指定帧的时间戳
        """
        path = os.path.join(self.config["input"], "sequence_info.json")
        with open(path, "r") as f:
            data = json.load(f)
        for i in data:
            if int(i.get("frame_id").strip("_")[-1]) == frame_id:
                return i.get("timestamp")
        return None
