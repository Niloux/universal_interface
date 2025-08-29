#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云数据读取脚本

用于读取PLY格式的点云数据并显示基本信息
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData


def read_ply_file(file_path: str) -> tuple[np.ndarray, dict]:
    """
    读取PLY文件并返回点云数据

    Args:
        file_path: PLY文件路径

    Returns:
        tuple: (点云坐标数组, 属性字典)
    """
    try:
        # 读取PLY文件
        ply_data = PlyData.read(file_path)

        # 获取顶点数据
        vertex = ply_data["vertex"]

        # 提取坐标信息，使用更安全的方式
        x_coords = np.array(vertex["x"], dtype=np.float64)
        y_coords = np.array(vertex["y"], dtype=np.float64)
        z_coords = np.array(vertex["z"], dtype=np.float64)

        # 检查并清理无效值
        valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords) & np.isfinite(z_coords)

        if not np.any(valid_mask):
            print("警告: 所有点云坐标都包含无效值")
            return None, None

        invalid_count = np.sum(~valid_mask)
        if invalid_count > 0:
            print(f"警告: 发现 {invalid_count} 个无效点，已自动过滤")

        # 只保留有效的点
        points = np.column_stack([x_coords[valid_mask], y_coords[valid_mask], z_coords[valid_mask]])

        # 提取其他属性
        attributes = {}
        for prop in vertex.properties:
            if prop.name not in ["x", "y", "z"]:
                attr_data = np.array(vertex[prop.name])
                # 同样过滤属性数据
                if len(attr_data) == len(valid_mask):
                    attributes[prop.name] = attr_data[valid_mask]
                else:
                    attributes[prop.name] = attr_data

        return points, attributes

    except Exception as e:
        print(f"读取PLY文件失败: {e}")
        return None, None


def analyze_pointcloud(points: np.ndarray, attributes: dict) -> None:
    """
    分析点云数据并打印统计信息

    Args:
        points: 点云坐标数组
        attributes: 点云属性字典
    """
    if points is None:
        return

    print("\n=== 点云数据分析 ===")
    print(f"点云总数: {len(points):,}")
    print(f"坐标维度: {points.shape[1]}")

    # 坐标范围统计
    print("\n坐标范围:")
    for i, axis in enumerate(["X", "Y", "Z"]):
        if i < points.shape[1]:
            min_val = np.min(points[:, i])
            max_val = np.max(points[:, i])
            mean_val = np.mean(points[:, i])
            print(f"  {axis}轴: [{min_val:.3f}, {max_val:.3f}], 均值: {mean_val:.3f}")

    # 属性信息
    if attributes:
        print("\n点云属性:")
        for attr_name, attr_data in attributes.items():
            if attr_data.dtype in [np.uint8, np.int32, np.float32, np.float64]:
                print(f"  {attr_name}: {attr_data.dtype}, 范围: [{np.min(attr_data)}, {np.max(attr_data)}]")
            else:
                print(f"  {attr_name}: {attr_data.dtype}")
    else:
        print("\n无额外属性")


def save_sample_points(points: np.ndarray, output_path: str, sample_size: int = 1000) -> None:
    """
    保存采样点云数据到文本文件

    Args:
        points: 点云坐标数组
        output_path: 输出文件路径
        sample_size: 采样点数
    """
    if points is None or len(points) == 0:
        return

    # 随机采样
    if len(points) > sample_size:
        indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[indices]
    else:
        sample_points = points

    # 保存到文件
    try:
        np.savetxt(output_path, sample_points, fmt="%.6f", delimiter=",", header="x,y,z", comments="")
        print(f"\n已保存 {len(sample_points)} 个采样点到: {output_path}")
    except Exception as e:
        print(f"保存采样点失败: {e}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="读取PLY格式点云数据")
    parser.add_argument("file_path", help="PLY文件路径")
    parser.add_argument("--sample", "-s", type=int, default=1000, help="保存采样点数 (默认: 1000)")
    parser.add_argument("--output", "-o", type=str, help="采样点输出文件路径 (可选)")

    args = parser.parse_args()

    # 检查文件是否存在
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"错误: 文件不存在 - {file_path}")
        sys.exit(1)

    if not file_path.suffix.lower() == ".ply":
        print(f"警告: 文件扩展名不是.ply - {file_path}")

    print(f"正在读取点云文件: {file_path}")

    # 读取点云数据
    points, attributes = read_ply_file(str(file_path))

    if points is not None:
        # 分析点云
        analyze_pointcloud(points, attributes)

        # 保存采样点 (如果指定了输出路径)
        if args.output:
            save_sample_points(points, args.output, args.sample)

        print("\n读取完成!")
    else:
        print("读取失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
