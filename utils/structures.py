"""数据结构模块

使用dataclasses定义项目中使用的核心数据结构，以增强类型安全和代码清晰度。
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Box3D:
    """代表一个3D边界框的数据结构"""

    height: float
    width: float
    length: float
    center_x: float
    center_y: float
    center_z: float
    heading: float
    label: Optional[str]
    speed: float
    timestamp: int


@dataclass
class TrajectoryData:
    """代表一条完整轨迹的数据结构"""

    label: Optional[str]
    height: float
    width: float
    length: float
    poses_vehicle: np.ndarray
    timestamps: List[int]
    frames: List[int]
    speeds: List[float]
    symmetric: bool
    deformable: bool
    stationary: bool


@dataclass
class FrameObject:
    """代表某一帧中的一个被跟踪的物体"""

    lidar_box: Box3D
    camera_box: Box3D  # 在这个项目中，两者是相同的
