import shutil
from pathlib import Path

import numpy as np


def process_files(input_dir, output_dir, pattern, processor):
    """通用文件处理函数"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"目录不存在: {input_dir}")
        return

    files = list(input_dir.glob(pattern))
    if not files:
        print(f"未找到匹配文件: {input_dir}/{pattern}")
        return

    print(f"处理 {len(files)} 个文件...")

    for file in files:
        try:
            processor(file, output_dir)
        except Exception as e:
            print(f"处理 {file.name} 失败: {e}")


def pose_processor(file, output_dir):
    """处理pose文件：16个数字转4x4矩阵"""
    numbers = [float(x) for x in file.read_text().strip().split()]
    if len(numbers) != 16:
        raise ValueError(f"期望16个数字，实际{len(numbers)}个")

    matrix = np.array(numbers).reshape(4, 4)
    output_file = output_dir / file.name

    with open(output_file, "w") as f:
        for row in matrix:
            f.write(" ".join(f"{x:.15e}" for x in row) + "\n")


def image_processor(file, output_dir):
    """处理图像文件：XXXXXX_Y.png -> camera_name/XXXXXX.jpg"""
    cameras = {"0": "FRONT", "1": "FRONT_LEFT", "2": "FRONT_RIGHT", "3": "SIDE_LEFT", "4": "SIDE_RIGHT"}

    parts = file.stem.split("_")
    if len(parts) != 2:
        raise ValueError("文件名格式错误，期望XXXXXX_Y.png")

    frame_id, camera_id = parts
    if camera_id not in cameras:
        raise ValueError(f"未知相机ID: {camera_id}")

    camera_dir = output_dir / cameras[camera_id]
    camera_dir.mkdir(exist_ok=True)

    shutil.copy2(file, camera_dir / f"{frame_id}.jpg")


def copy_processor(file, output_dir):
    """直接复制文件到输出目录"""
    shutil.copy2(file, output_dir / file.name)


def labels_processor(file, output_dir):
    """将标签txt文件转换为按帧分组的JSON格式的处理器"""
    import json
    from collections import defaultdict

    # 按帧ID分组存储对象
    frames = defaultdict(list)

    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            if len(parts) >= 9:
                frame_id, obj_id, obj_type, x, y, z, l, w, h, heading = parts[:10]  # noqa: E741

                obj = {
                    "track_id": obj_id,
                    "label": f"type_{obj_type}",
                    "box3d_center": [float(x), float(y), float(z)],
                    "box3d_size": [float(l), float(w), float(h)],
                    "box3d_heading": float(heading),
                }
                frames[frame_id].append(obj)

    # 为每个帧生成单独的JSON文件
    for frame_id, objects in frames.items():
        output_file = output_dir / f"{frame_id}.json"
        with open(output_file, "w") as f:
            json.dump(objects, f, indent=2)


def pointcloud_processor(input_file, output_dir):
    """处理点云数据：从二进制PLY转换为ASCII PLY，并进行坐标转换"""
    import os
    import struct

    try:
        # 从文件名解析帧号和雷达ID
        filename = Path(input_file).stem  # 例如: "000000_0"
        frame_id, lidar_id = filename.split("_")

        # 雷达ID到目录名的映射 - 根据用户提供的正确映射
        lidar_to_dir = {
            "0": "TOP",  # TOP是0
            "1": "FRONT",  # FRONT是1
            "2": "SIDE_LEFT",  # side_left是2
            "3": "SIDE_RIGHT",  # side_right是3
            "4": "REAR",  # rear是4
        }

        if lidar_id not in lidar_to_dir:
            print(f"未知的雷达ID: {lidar_id}")
            return

        # 读取对应的lidar2ego转换矩阵
        extrinsics_file = f"final_data/extrinsics_lidar/{lidar_id}.txt"
        if not os.path.exists(extrinsics_file):
            print(f"外参文件不存在: {extrinsics_file}")
            return

        # 读取4x4转换矩阵 (16个数字排成一行)
        matrix_data = np.loadtxt(extrinsics_file)
        lidar2ego = matrix_data.reshape(4, 4)

        # 读取二进制PLY文件
        with open(input_file, "rb") as f:
            # 跳过PLY头部
            line = f.readline().decode("ascii").strip()
            while line != "end_header":
                line = f.readline().decode("ascii").strip()

            # 读取点云数据 (x, y, z, intensity)
            # 每个点13字节: 3个float(x,y,z) + 1个uchar(intensity)
            points_data = []
            while True:
                data = f.read(13)  # 3*4 + 1 = 13字节
                if len(data) < 13:
                    break
                x, y, z = struct.unpack("<fff", data[:12])
                intensity = struct.unpack("<B", data[12:13])[0]
                points_data.append([x, y, z])

        if not points_data:
            print(f"文件 {input_file} 中没有点云数据")
            return

        # 转换为numpy数组并进行坐标变换
        points = np.array(points_data)
        # 添加齐次坐标
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])

        # 应用lidar2ego变换
        points_ego = (lidar2ego @ points_homo.T).T
        # 只保留x, y, z坐标
        points_ego = points_ego[:, :3]

        # 创建输出目录
        sensor_dir = os.path.join(output_dir, lidar_to_dir[lidar_id])
        os.makedirs(sensor_dir, exist_ok=True)

        # 写入ASCII格式的PLY文件
        output_file = os.path.join(sensor_dir, f"{frame_id}.ply")
        with open(output_file, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_ego)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            for point in points_ego:
                f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f}\n")

    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")


if __name__ == "__main__":
    base_input = Path("final_data")
    base_output = Path("required_data")

    process_files(base_input / "poses", base_output / "ego_pose", "*.txt", pose_processor)
    process_files(base_input / "images", base_output / "images", "*.png", image_processor)
    process_files(base_input / "extrinsics_camera", base_output / "extrinsics", "*.txt", copy_processor)
    process_files(base_input / "intrinsics_camera", base_output / "intrinsics", "*.txt", copy_processor)
    process_files(base_input / "labels", base_output / "objects", "*.txt", labels_processor)
    process_files(base_input / "pointclouds", base_output / "pointclouds", "*.ply", pointcloud_processor)
