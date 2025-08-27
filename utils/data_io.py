"""数据输入输出工具模块

提供统一的、原子化的数据加载和保存功能，供所有处理器调用。
"""

import os
import pickle
import json
from typing import Any, Dict, List

import cv2
import numpy as np

from utils.logger import default_logger

CAMERA_IDS = ["0", "1", "2", "3", "4"]


def load_pickle(file_path: str) -> Any:
    """加载pickle文件"""
    if not os.path.exists(file_path):
        default_logger.error(f"Pickle文件不存在: {file_path}")
        return None
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        default_logger.error(f"加载pickle文件失败 {file_path}: {e}")
        return None


def load_extrinsics(output_path: str) -> Dict[str, np.ndarray]:
    """加载所有相机的外参"""
    extrinsics = {}
    ext_path_dir = os.path.join(output_path, "extrinsics")
    if not os.path.exists(ext_path_dir):
        return extrinsics
        
    for camera_id in CAMERA_IDS:
        ext_path = os.path.join(ext_path_dir, f"{camera_id}.txt")
        if os.path.exists(ext_path):
            extrinsics[camera_id] = np.loadtxt(ext_path)
    return extrinsics


def load_intrinsics(output_path: str) -> Dict[str, np.ndarray]:
    """加载所有相机的内参，并构建3x3矩阵"""
    intrinsics = {}
    int_path_dir = os.path.join(output_path, "intrinsics")
    if not os.path.exists(int_path_dir):
        return intrinsics

    for camera_id in CAMERA_IDS:
        int_path = os.path.join(int_path_dir, f"{camera_id}.txt")
        if os.path.exists(int_path):
            params = np.loadtxt(int_path)
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            intrinsics[camera_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsics


def load_images(image_path: str, frame_name: str) -> Dict[str, np.ndarray]:
    """加载指定帧的所有相机图像"""
    images = {}
    for camera_id in CAMERA_IDS:
        img_path = os.path.join(image_path, f"{frame_name}_{camera_id}.png")
        if os.path.exists(img_path):
            images[camera_id] = cv2.imread(img_path)
    return images


def load_ego_pose(output_path: str, frame_id: int) -> np.ndarray:
    """
    加载指定帧的ego pose数据
    """
    path = os.path.join(output_path, "ego_pose", f"{frame_id:06d}.txt")
    if os.path.exists(path):
        return np.loadtxt(path)
    return np.eye(4) # 返回默认单位矩阵


def load_timestamp(input_path: str, frame_id: int) -> int:
    """
    加载指定帧的时间戳
    """
    path = os.path.join(input_path, "ego_pose", f"{frame_id:06d}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("timestamp", 0)
    return 0
