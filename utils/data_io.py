"""数据输入输出工具模块

提供统一的、原子化的数据加载和保存功能，供所有处理器调用。
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from utils.logger import default_logger


def save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, mask: np.ndarray):
    """将点云数据保存为PLY文件

    Args:
        path: PLY文件的保存路径
        xyz: 点云坐标 (N, 3)
        rgb: 点云颜色 (N, 3)
        mask: 点云掩码 (N, 1)
    """
    # set rgb to 0 - 255
    if rgb.max() <= 1.0 and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0.0, 255.0)

    # set mask to bool data type
    mask = mask.astype(np.bool_)

    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("mask", "?"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, rgb, mask), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(str(path))


def load_pickle(file_path: Path) -> Any:
    """加载pickle文件"""
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle文件不存在: {file_path}")
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        default_logger.error(f"加载pickle文件失败 {file_path}: {e}")
        raise


def load_extrinsics(output_path: Path, camera_ids: List[str]) -> Dict[str, np.ndarray]:
    """加载所有相机的外参"""
    extrinsics = {}
    ext_path_dir = output_path / "extrinsics"
    if not ext_path_dir.exists():
        return extrinsics

    for camera_id in camera_ids:
        ext_path = ext_path_dir / f"{camera_id}.txt"
        if ext_path.exists():
            extrinsics[camera_id] = np.loadtxt(ext_path)
    return extrinsics


def load_intrinsics(output_path: Path, camera_ids: List[str]) -> Dict[str, np.ndarray]:
    """加载所有相机的内参，并构建3x3矩阵"""
    intrinsics = {}
    int_path_dir = output_path / "intrinsics"
    if not int_path_dir.exists():
        return intrinsics

    for camera_id in camera_ids:
        int_path = int_path_dir / f"{camera_id}.txt"
        if int_path.exists():
            params = np.loadtxt(int_path)
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            intrinsics[camera_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsics


def load_images(image_path: Path, frame_name: str, camera_ids: List[str]) -> Dict[str, np.ndarray]:
    """加载指定帧的所有相机图像"""
    images = {}
    for camera_id in camera_ids:
        img_path = image_path / f"{frame_name}_{camera_id}.png"
        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            images[camera_id] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR->RGB，保持连续
    return images


def load_ego_pose(output_path: Path, frame_id: int) -> np.ndarray:
    """
    加载指定帧的ego pose数据
    """
    path = output_path / "ego_pose" / f"{frame_id:06d}.txt"
    if path.exists():
        return np.loadtxt(path)
    return np.eye(4)  # 返回默认单位矩阵


def load_timestamp(input_path: Path, frame_id: int) -> int:
    """
    加载指定帧的时间戳
    """
    path = input_path / "ego_pose" / f"{frame_id:06d}.json"
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("timestamp", 0)
    return 0
