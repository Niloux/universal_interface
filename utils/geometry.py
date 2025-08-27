"""几何变换工具模块

提供统一的3D到2D坐标变换、投影等功能。
"""

import math
from typing import Any, Dict, Optional

import numpy as np

def get_box_corners_3d(box_info: Dict[str, Any]) -> np.ndarray:
    """
    获取3D边界框的8个顶点坐标（主车坐标系）

    Args:
        box_info: 边界框信息

    Returns:
        np.ndarray: shape (8, 3)，8个顶点的3D坐标
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

def transform_to_camera(points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """
    将点从车辆坐标系转换到相机坐标系

    注意：外参是sensor to vehicle，所以需要求逆来从vehicle转到sensor

    Args:
        points: shape (N, 3)，车辆坐标系下的点
        extrinsics: shape (4, 4)，相机外参矩阵 (sensor to vehicle)

    Returns:
        np.ndarray: shape (N, 3)，相机坐标系下的点
    """
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    extrinsics_inv = np.linalg.inv(extrinsics)
    points_camera_homo = (extrinsics_inv @ points_homo.T).T
    return points_camera_homo[:, :3]

def project_to_2d(points_3d: np.ndarray, intrinsics: np.ndarray) -> Optional[np.ndarray]:
    """
    将3D点投影到2D图像平面

    Args:
        points_3d: shape (N, 3)，相机坐标系下的3D点
        intrinsics: shape (3, 3)，相机内参矩阵

    Returns:
        np.ndarray: shape (N, 2)，2D图像坐标，如果投影失败返回None
    """
    try:
        points_2d_homo = (intrinsics @ points_3d.T).T
        z_coords = points_2d_homo[:, 2]
        if np.any(np.abs(z_coords) < 1e-8):
            return None
        points_2d = points_2d_homo[:, :2] / z_coords.reshape(-1, 1)
        return points_2d
    except Exception:
        return None

def project_box_to_image(
    box_info: Dict[str, Any],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    img_shape: tuple,
) -> tuple[bool, Optional[np.ndarray]]:
    """
    将3D边界框投影到图像，并检查可见性。
    这是一个集成了多个步骤的便捷函数。

    Args:
        box_info: 3D边界框信息
        extrinsics: 对应相机的外参 (4x4)
        intrinsics: 对应相机的内参 (3x3)
        img_shape: 图像尺寸 (height, width)

    Returns:
        tuple: (是否可见, 2D投影点数组)
    """
    try:
        corners_3d_vehicle = get_box_corners_3d(box_info)
        corners_3d_camera = transform_to_camera(corners_3d_vehicle, extrinsics)

        if np.any(corners_3d_camera[:, 2] <= 0):
            return False, None

        pts_2d = project_to_2d(corners_3d_camera, intrinsics)

        if pts_2d is None or np.any(np.isnan(pts_2d)) or np.any(np.isinf(pts_2d)):
            return False, None

        img_height, img_width = img_shape
        x_in_range = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_width)
        y_in_range = (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_height)
        
        visible = bool(np.any(x_in_range & y_in_range))

        return visible, pts_2d.astype(int) if visible else None

    except Exception:
        return False, None
