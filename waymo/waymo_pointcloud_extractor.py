#!/usr/bin/env python3
"""
Waymo点云和激光雷达外参提取脚本

功能：
1. 从Waymo数据集提取点云数据（包含强度信息）
2. 提取激光雷达外参（雷达坐标系到车辆坐标系的变换矩阵）
3. 将所有激光雷达点云从车辆坐标系统一转换到0号(TOP)雷达坐标系并合并
4. 保存为PLY格式（包含x,y,z,intensity）
5. 保存外参为TXT格式（4x4变换矩阵）

使用方法：
python waymo_pointcloud_extractor.py --input_path /path/to/waymo/tfrecords --output_path /path/to/output
"""  # noqa: E501

import argparse
import os
from typing import Dict

import numpy as np
import tensorflow as tf
from plyfile import PlyData, PlyElement

try:
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0" to install the official devkit.'  # noqa: E501
    )

# Waymo激光雷达名称映射
WAYMO_LIDAR_ENUM_TO_STR_MAP = {
    dataset_pb2.LaserName.TOP: "TOP",
    dataset_pb2.LaserName.FRONT: "FRONT",
    dataset_pb2.LaserName.SIDE_LEFT: "SIDE_LEFT",
    dataset_pb2.LaserName.SIDE_RIGHT: "SIDE_RIGHT",
    dataset_pb2.LaserName.REAR: "REAR",
}

# 激光雷达编号映射（TOP=0, FRONT=1, SIDE_LEFT=2, SIDE_RIGHT=3, REAR=4）
LIDAR_NAME_TO_ID = {
    "TOP": 0,
    "FRONT": 1,
    "SIDE_LEFT": 2,
    "SIDE_RIGHT": 3,
    "REAR": 4,
}

# Waymo标准激光雷达的有序列表（与frame_utils返回顺序一致）
ORDERED_LIDAR_NAME_ENUMS = [
    dataset_pb2.LaserName.TOP,
    dataset_pb2.LaserName.FRONT,
    dataset_pb2.LaserName.SIDE_LEFT,
    dataset_pb2.LaserName.SIDE_RIGHT,
    dataset_pb2.LaserName.REAR,
]


class WaymoPointCloudExtractor:
    """Waymo点云和外参提取器"""

    def __init__(self, input_path: str, output_path: str):
        """
        初始化提取器

        Args:
            input_path: Waymo tfrecord文件路径
            output_path: 输出目录路径
        """
        self.input_path = input_path
        self.output_path = output_path

        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        self.pointcloud_dir = os.path.join(output_path, "pointclouds")
        self.calib_dir = os.path.join(output_path, "calibrations")
        os.makedirs(self.pointcloud_dir, exist_ok=True)
        os.makedirs(self.calib_dir, exist_ok=True)

        print(f"输入路径: {input_path}")
        print(f"输出路径: {output_path}")
        print(f"点云保存目录: {self.pointcloud_dir}")
        print(f"外参保存目录: {self.calib_dir}")

    def _parse_laser_calibration(self, context_pb2) -> Dict[str, Dict]:
        """
        解析激光雷达标定参数

        Args:
            context_pb2: Waymo context protobuf对象

        Returns:
            激光雷达标定参数字典
        """
        laser_calibrations = {}

        for laser_calib in context_pb2.laser_calibrations:
            lidar_name = WAYMO_LIDAR_ENUM_TO_STR_MAP.get(laser_calib.name)
            if lidar_name is None:
                continue

            # 提取外参矩阵（激光雷达坐标系到车辆坐标系的变换）
            extrinsic_matrix = np.array(laser_calib.extrinsic.transform).reshape(4, 4)

            # 提取光束倾角范围
            beam_inclination_min = laser_calib.beam_inclination_min
            beam_inclination_max = laser_calib.beam_inclination_max

            laser_calibrations[lidar_name] = {
                "extrinsic_matrix": extrinsic_matrix,  # 激光雷达到车辆坐标系
                "beam_inclination_min": beam_inclination_min,
                "beam_inclination_max": beam_inclination_max,
                "lidar_id": LIDAR_NAME_TO_ID[lidar_name],
            }

        return laser_calibrations

    def _transform_points_to_lidar_frame(
        self, points_vehicle: np.ndarray, extrinsic_matrix: np.ndarray
    ) -> np.ndarray:
        """
        将点云从车辆坐标系转换到激光雷达坐标系

        Args:
            points_vehicle: 车辆坐标系下的点云 [N, 3] 或 [N, 6]
                          当keep_polar_features=True时，格式为[range, intensity, elongation, x, y, z]
            extrinsic_matrix: 激光雷达到车辆坐标系的4x4变换矩阵

        Returns:
            激光雷达坐标系下的点云
        """  # noqa: E501
        if points_vehicle.size == 0:
            return points_vehicle

        # 根据点云维度确定xyz坐标的位置
        if points_vehicle.shape[1] == 6:
            # keep_polar_features=True: [range, intensity, elongation, x, y, z]
            xyz_vehicle = points_vehicle[:, 3:6]  # 提取后3个维度作为xyz坐标
        else:
            # keep_polar_features=False: [x, y, z]
            xyz_vehicle = points_vehicle[:, :3]

        # 转换为齐次坐标
        xyz_homo = np.concatenate(
            [xyz_vehicle, np.ones((xyz_vehicle.shape[0], 1))], axis=1
        )

        # 车辆坐标系到激光雷达坐标系的变换（外参的逆）
        vehicle_to_lidar = np.linalg.inv(extrinsic_matrix)

        # 执行坐标变换
        xyz_lidar_homo = xyz_homo @ vehicle_to_lidar.T
        xyz_lidar = xyz_lidar_homo[:, :3]

        # 如果原始点云包含其他特征，保留它们
        if points_vehicle.shape[1] == 6:
            # keep_polar_features=True: 保留前3个极坐标特征[range, intensity, elongation]  # noqa: E501
            polar_features = points_vehicle[:, :3]
            points_lidar = np.concatenate([polar_features, xyz_lidar], axis=1)
        else:
            # keep_polar_features=False: 只有xyz坐标
            points_lidar = xyz_lidar

        return points_lidar

    def _normalize_intensity_to_uint8(self, intensity: np.ndarray) -> np.ndarray:
        """
        将Waymo点云的intensity归一化到0~255并转换为uint8

        说明:
            - 若intensity已在[0,1]内，则直接线性映射到[0,255]
            - 否则使用1%~99%分位数做线性归一化并裁剪到[0,255]，以降低极端值影响

        Args:
            intensity: 原始intensity数组，形状为[N,]

        Returns:
            归一化后的uint8数组，形状为[N,]
        """
        intensity = np.asarray(intensity, dtype=np.float32)
        if intensity.size == 0:
            return intensity.astype(np.uint8)

        finite_mask = np.isfinite(intensity)
        if not np.any(finite_mask):
            return np.zeros_like(intensity, dtype=np.uint8)

        finite_vals = intensity[finite_mask]
        min_val = float(np.min(finite_vals))
        max_val = float(np.max(finite_vals))

        if min_val >= 0.0 and max_val <= 1.0:
            scaled = intensity * 255.0
        else:
            lo = float(np.percentile(finite_vals, 1.0))
            hi = float(np.percentile(finite_vals, 99.0))
            if hi <= lo + 1e-6:
                scaled = np.zeros_like(intensity, dtype=np.float32)
            else:
                scaled = (intensity - lo) / (hi - lo) * 255.0

        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        scaled = np.clip(scaled, 0.0, 255.0)
        return scaled.astype(np.uint8)

    def _save_pointcloud_as_ply(self, points: np.ndarray, filename: str):
        """
        保存点云为PLY格式

        Args:
            points: 点云数据 [N, 3] 或 [N, 6] (x,y,z) 或 (range,intensity,elongation,x,y,z)
            filename: 输出文件名
        """  # noqa: E501
        if points.size == 0:
            print(f"警告: 点云为空，跳过保存 {filename}")
            return

        # 根据点云维度确定数据格式
        if points.shape[1] == 3:
            # 只有xyz坐标
            vertex_data = [
                (points[i, 0], points[i, 1], points[i, 2])
                for i in range(points.shape[0])
            ]
            vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

        elif points.shape[1] == 6:
            # 包含range, intensity, elongation, x, y, z
            intensity_u8 = self._normalize_intensity_to_uint8(points[:, 1])
            vertex_data = [
                (
                    points[i, 3],
                    points[i, 4],
                    points[i, 5],
                    int(intensity_u8[i]),
                )
                for i in range(points.shape[0])
            ]
            vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "u1")]

        else:
            raise ValueError(f"不支持的点云维度: {points.shape[1]}")

        # 创建PLY元素
        vertex_element = PlyElement.describe(
            np.array(vertex_data, dtype=vertex_dtype), "vertex"
        )

        # 保存PLY文件
        PlyData([vertex_element]).write(filename)
        print(f"保存点云: {filename} ({points.shape[0]} 个点)")

    def _save_calibration_as_txt(self, extrinsic_matrix: np.ndarray, filename: str):
        """
        保存外参矩阵为TXT格式

        Args:
            extrinsic_matrix: 4x4外参矩阵
            filename: 输出文件名
        """
        # 将4x4矩阵展平为一行，用空格分隔
        matrix_flat = extrinsic_matrix.flatten()

        with open(filename, "w") as f:
            f.write(" ".join([f"{val:.6f}" for val in matrix_flat]))

        print(f"保存外参: {filename}")

    def extract_sequence(self, tfrecord_path: str) -> bool:  # noqa: C901
        """
        提取单个序列的点云和外参

        Args:
            tfrecord_path: tfrecord文件路径

        Returns:
            是否成功提取
        """
        try:
            # 读取tfrecord数据集
            tf_dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")

            # 获取序列名称
            sequence_name = os.path.splitext(os.path.basename(tfrecord_path))[0]
            print(f"\n处理序列: {sequence_name}")

            # 解析第一帧获取标定信息
            first_frame_data = next(iter(tf_dataset), None)
            if first_frame_data is None:
                print(f"错误: 序列 {sequence_name} 为空")
                return False

            # 解析context信息
            context_frame_pb2 = dataset_pb2.Frame()
            context_frame_pb2.ParseFromString(bytearray(first_frame_data.numpy()))
            context_pb2 = context_frame_pb2.context

            # 解析激光雷达标定参数
            laser_calibrations = self._parse_laser_calibration(context_pb2)

            # 保存外参文件（每个激光雷达一个文件）
            for lidar_name, calib_data in laser_calibrations.items():
                lidar_id = calib_data["lidar_id"]
                extrinsic_matrix = calib_data["extrinsic_matrix"]

                calib_filename = os.path.join(self.calib_dir, f"{lidar_id}.txt")
                self._save_calibration_as_txt(extrinsic_matrix, calib_filename)

            # 逐帧处理点云数据
            frame_count = 0
            for frame_idx, data in enumerate(tf_dataset):
                frame_pb2 = dataset_pb2.Frame()
                frame_pb2.ParseFromString(bytearray(data.numpy()))

                # 检查是否有激光雷达数据
                if not frame_pb2.lasers:
                    continue

                # 解析range image和相机投影
                (
                    range_images,
                    camera_projections,
                    segmentation_labels,
                    range_image_top_pose,
                ) = frame_utils.parse_range_image_and_camera_projection(frame_pb2)

                # 转换range image为点云（包含强度信息）
                points_with_features, _ = (
                    frame_utils.convert_range_image_to_point_cloud(
                        frame_pb2,
                        range_images,
                        camera_projections,
                        range_image_top_pose,
                        ri_index=0,
                        keep_polar_features=True,  # 保留强度等特征
                    )
                )

                # 将所有激光雷达点云统一变换到0号(TOP)雷达坐标系并合并
                top_extrinsic_matrix = laser_calibrations.get("TOP", {}).get(
                    "extrinsic_matrix"
                )
                if top_extrinsic_matrix is None:
                    print("警告: 未找到 TOP 的标定参数，跳过该帧")
                    continue

                merged_points_top = []
                for i, lidar_enum in enumerate(ORDERED_LIDAR_NAME_ENUMS):
                    _lidar_name = WAYMO_LIDAR_ENUM_TO_STR_MAP[lidar_enum]

                    # 获取该激光雷达的点云（车辆坐标系）
                    points_vehicle = points_with_features[i]
                    if points_vehicle.size == 0:
                        continue

                    points_top = self._transform_points_to_lidar_frame(
                        points_vehicle, top_extrinsic_matrix
                    )
                    merged_points_top.append(points_top)

                if merged_points_top:
                    merged_points_top = np.concatenate(merged_points_top, axis=0)
                else:
                    merged_points_top = np.empty((0, 6), dtype=np.float32)

                # 保存合并后的PLY文件（frameIdx_0.ply）
                ply_filename = os.path.join(
                    self.pointcloud_dir, f"{frame_idx:06d}_0.ply"
                )
                self._save_pointcloud_as_ply(merged_points_top, ply_filename)

                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"已处理 {frame_count} 帧")

            print(f"序列 {sequence_name} 处理完成，共 {frame_count} 帧")
            return True

        except Exception as e:
            print(f"处理序列 {tfrecord_path} 时出错: {e}")
            return False

    def extract_all_sequences(self):
        """
        提取指定路径下所有tfrecord文件的点云和外参
        """
        if os.path.isfile(self.input_path):
            # 单个文件
            if self.input_path.endswith(".tfrecord"):
                self.extract_sequence(self.input_path)
            else:
                print(f"错误: {self.input_path} 不是tfrecord文件")
        elif os.path.isdir(self.input_path):
            # 目录中的所有tfrecord文件
            tfrecord_files = sorted([
                f for f in os.listdir(self.input_path) if f.endswith(".tfrecord")
            ])

            if not tfrecord_files:
                print(f"错误: 在 {self.input_path} 中未找到tfrecord文件")
                return

            print(f"找到 {len(tfrecord_files)} 个tfrecord文件")

            success_count = 0
            for tfrecord_file in tfrecord_files:
                tfrecord_path = os.path.join(self.input_path, tfrecord_file)
                if self.extract_sequence(tfrecord_path):
                    success_count += 1

            print(f"\n提取完成: {success_count}/{len(tfrecord_files)} 个序列成功处理")
        else:
            print(f"错误: 路径 {self.input_path} 不存在")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从Waymo数据集提取点云和激光雷达外参",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
输出格式说明:
1. 点云文件: XXXXXX_0.ply
   - XXXXXX: 6位帧编号 (000000, 000001, ...)
   - 坐标系: 0号(TOP)激光雷达坐标系
   - 内容: 合并了TOP/FRONT/SIDE_LEFT/SIDE_RIGHT/REAR五个激光雷达点云
   - 包含: x, y, z, intensity

2. 外参文件: Y.txt
   - Y: 激光雷达编号 (0=TOP, 1=FRONT, 2=SIDE_LEFT, 3=SIDE_RIGHT, 4=REAR)
   - 格式: 4x4变换矩阵的16个数字，空格分隔，一行
   - 含义: 激光雷达坐标系到车辆坐标系的变换矩阵

示例:
python waymo_pointcloud_extractor.py --input_path /data/waymo/segment-xxx.tfrecord --output_path /output/waymo_extracted
python waymo_pointcloud_extractor.py --input_path /data/waymo/ --output_path /output/waymo_extracted
        """,  # noqa: E501
    )

    parser.add_argument(
        "--input_path",
        required=True,
        help="输入路径（tfrecord文件或包含tfrecord文件的目录）",
    )
    parser.add_argument("--output_path", required=True, help="输出目录路径")

    args = parser.parse_args()

    # 创建提取器并执行提取
    extractor = WaymoPointCloudExtractor(args.input_path, args.output_path)
    extractor.extract_all_sequences()


if __name__ == "__main__":
    main()
