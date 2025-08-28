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


if __name__ == "__main__":
    base_input = Path("final_data")
    base_output = Path("required_data")

    process_files(base_input / "poses", base_output / "ego_pose", "*.txt", pose_processor)
    process_files(base_input / "images", base_output / "images", "*.png", image_processor)
    process_files(base_input / "extrinsics_camera", base_output / "extrinsics", "*.txt", copy_processor)
    process_files(base_input / "intrinsics_camera", base_output / "intrinsics", "*.txt", copy_processor)
    process_files(base_input / "labels", base_output / "objects", "*.txt", labels_processor)
