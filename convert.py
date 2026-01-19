import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


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
    """处理pose文件：将数字序列转换为4x4矩阵并写出。

    - 若文件包含16个数字，则直接按行填充为4x4矩阵；
    - 若文件包含12个数字，则在末尾补齐 `0, 0, 0, 1` 形成16个数字，再转换为4x4矩阵；
    - 其他长度的数字则视为格式错误并报错。
    """
    tokens = file.read_text().strip().split()
    numbers = [float(x) for x in tokens]

    if len(numbers) == 16:
        pass  # 直接使用
    elif len(numbers) == 12:
        # 在末尾补齐 0, 0, 0, 1，得到 4x4 齐次变换矩阵
        numbers += [0.0, 0.0, 0.0, 1.0]
    else:
        raise ValueError(f"期望12或16个数字，实际{len(numbers)}个")

    matrix = np.array(numbers).reshape(4, 4)
    output_file = output_dir / file.name

    with open(output_file, "w") as f:
        for row in matrix:
            f.write(" ".join(f"{x:.15e}" for x in row) + "\n")


def image_processor(file, output_dir):
    """处理图像文件：{frame}_{cam}.(jpg|png) -> images/{cam}/{frame}.(jpg|png)。

    说明：
    - 该处理器用于“平铺式 images 目录”输入：所有图像都在同一层级，通过文件名携带相机信息；
    - 若图像已按相机分目录组织（images/<cam>/<frame>.jpg），请使用 `convert_images(...)`。
    """
    parts = file.stem.split("_")
    if len(parts) != 2:
        raise ValueError("文件名格式错误，期望{frame}_{cam}.(jpg|png)")

    frame_id, camera_id = parts
    camera_dir = output_dir / str(camera_id)
    camera_dir.mkdir(exist_ok=True)

    shutil.copy2(file, camera_dir / f"{frame_id}{file.suffix.lower()}")


def convert_images(images_input_dir: Path, images_output_dir: Path) -> None:
    """转换图像目录到标准化结构：images/<cam>/<frame>.(jpg|png)。

    支持两种输入：
    1) 分目录式：images/<cam>/<frame>.(jpg|png)
    2) 平铺式：images/{frame}_{cam}.(jpg|png)

    Args:
        images_input_dir: 原始 images 目录路径。
        images_output_dir: 输出 images 目录路径。
    """
    images_output_dir.mkdir(parents=True, exist_ok=True)

    if not images_input_dir.exists():
        print(f"目录不存在: {images_input_dir}")
        return

    subdirs = [p for p in images_input_dir.iterdir() if p.is_dir()]
    if subdirs:
        for cam_dir in sorted(subdirs, key=lambda p: p.name):
            out_cam_dir = images_output_dir / cam_dir.name
            out_cam_dir.mkdir(parents=True, exist_ok=True)
            for img in sorted(cam_dir.iterdir(), key=lambda p: p.name):
                if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    shutil.copy2(img, out_cam_dir / f"{img.stem}{img.suffix.lower()}")
        return

    for img in sorted(images_input_dir.iterdir(), key=lambda p: p.name):
        if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                image_processor(img, images_output_dir)
            except Exception as e:
                print(f"处理 {img.name} 失败: {e}")


def copy_processor(file, output_dir):
    """直接复制文件到输出目录"""
    shutil.copy2(file, output_dir / file.name)


def labels_processor(file, output_dir):
    """将标签txt文件转换为按帧分组的JSON格式。

    解析规则：
    - 支持以逗号分隔（",")或以空白分隔（空格/制表符）的行；
    - 每行期望字段顺序为：frame_id, track_id, obj_type, x, y, z, l, w, h, heading；
    - 输出文件以帧ID命名为 {frame_id}.json；若输入文件名为纯数字（如 000000.txt），
      则对帧ID进行同宽零填充，以保持与输入文件名一致（如将 0 写为 000000.json）。
    """
    import json
    from collections import defaultdict

    # 按帧ID分组存储对象
    frames = defaultdict(list)

    # 若输入文件名为纯数字，则记录其宽度用于输出文件名零填充
    stem = file.stem
    pad_width = len(stem) if stem.isdigit() else None

    with open(file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 兼容两种分隔符：优先尝试逗号分隔，否则按任意空白分隔
            parts = (
                [p.strip() for p in line.split(",")] if "," in line else line.split()
            )

            # 至少需要10个字段（frame_id, track_id, obj_type, x, y, z, l, w, h, heading）
            if len(parts) < 10:
                # 跳过格式不正确的行
                continue

            frame_id, obj_id, obj_type, x, y, z, l, w, h, heading = parts[:10]  # noqa: E741

            try:
                obj = {
                    "track_id": obj_id,
                    "label": f"type_{obj_type}",
                    "box3d_center": [float(x), float(y), float(z)],
                    "box3d_size": [float(l), float(w), float(h)],
                    "box3d_heading": float(heading),
                }
            except ValueError:
                # 数值解析失败则跳过该行
                continue

            frames[frame_id].append(obj)

    # 为每个帧生成单独的JSON文件
    for fid, objects in frames.items():
        # 若需零填充且帧ID为数字，则按输入文件名宽度填充
        out_name = fid.zfill(pad_width) if pad_width and fid.isdigit() else fid
        output_file = output_dir / f"{out_name}.json"
        with open(output_file, "w") as f:
            json.dump(objects, f, indent=2)


def pointcloud_processor(input_file, output_dir):
    """处理点云为 PLY：从输入 PLY 读取并保留 x/y/z，以及可选的
    intensity 和 dropout 属性；按帧与雷达ID组织输出。"""
    try:
        filename = Path(input_file).stem
        frame_id, lidar_id = filename.split("_")

        lidar_to_dir = {
            "0": "TOP",
            "1": "FRONT",
            "2": "SIDE_LEFT",
            "3": "SIDE_RIGHT",
            "4": "REAR",
        }
        if lidar_id not in lidar_to_dir:
            print(f"未知的雷达ID: {lidar_id}")
            return

        ply = PlyData.read(str(input_file))
        if "vertex" not in ply:
            print(f"PLY缺少vertex元素: {input_file}")
            return
        vertices = ply["vertex"].data
        if not {"x", "y", "z"}.issubset(vertices.dtype.names):
            print(f"vertex缺少x/y/z属性: {input_file}")
            return

        base_input = Path(input_file).parent.parent
        extrinsics_path = base_input / "extrinsics_lidar" / f"{lidar_id}.txt"
        if extrinsics_path.exists():
            print("Lidar_T_Vehicle转换")
            E = np.loadtxt(str(extrinsics_path))
            if E.ndim == 1:
                if E.size == 12:
                    E = np.append(E, [0.0, 0.0, 0.0, 1.0])
                E = E.reshape(4, 4)
            elif E.shape == (3, 4):
                E = np.vstack([E, np.array([0.0, 0.0, 0.0, 1.0])])
            elif E.shape != (4, 4):
                raise ValueError(f"外参矩阵形状不合法: {E.shape}")

            points_lidar = np.stack(
                [vertices["x"], vertices["y"], vertices["z"]], axis=1
            ).astype(np.float32)
            ones = np.ones((points_lidar.shape[0], 1), dtype=np.float32)
            points_homo = np.hstack([points_lidar, ones])
            points_ego_homo = (E @ points_homo.T).T
            points_ego = points_ego_homo[:, :3].astype(np.float32)
        else:
            points_ego = np.stack(
                [vertices["x"], vertices["y"], vertices["z"]], axis=1
            ).astype(np.float32)

        has_intensity = "intensity" in vertices.dtype.names
        has_dropout = "dropout" in vertices.dtype.names
        intensity = (
            np.asarray(vertices["intensity"], dtype=np.float32)
            if has_intensity
            else np.ones(points_ego.shape[0], dtype=np.float32)
        )
        if intensity.max() > 1.0:
            intensity = np.clip(intensity, 0.0, 255.0) / 255.0
        else:
            intensity = np.clip(intensity, 0.0, 1.0)
        dropout = (
            np.asarray(vertices["dropout"], dtype=np.bool_) if has_dropout else None
        )

        sensor_dir = Path(output_dir) / lidar_to_dir[lidar_id]
        sensor_dir.mkdir(parents=True, exist_ok=True)
        output_file = sensor_dir / f"{frame_id}.ply"

        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")]
        if has_dropout:
            dtype.append(("dropout", "?"))
        vertex = np.empty(points_ego.shape[0], dtype=dtype)
        vertex["x"] = points_ego[:, 0].astype(np.float32)
        vertex["y"] = points_ego[:, 1].astype(np.float32)
        vertex["z"] = points_ego[:, 2].astype(np.float32)
        vertex["intensity"] = intensity
        if has_dropout:
            vertex["dropout"] = dropout

        ply_out = PlyData(
            [PlyElement.describe(vertex, "vertex")], text=False
        )  # text=True -> ASCII
        ply_out.write(str(output_file))
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {e}")


def convert_dataset(input_root: Path, required_data_root: Path) -> bool:
    """将原始数据集目录转换为流水线所需的标准化结构（required_data）。

    Args:
        input_root: 原始输入数据根目录（包含 poses/images/extrinsics_camera/...）。
        required_data_root: 标准化数据输出根目录（即 required_data）。

    Returns:
        bool: 转换是否成功（遇到关键目录缺失/异常会返回False）。
    """
    try:
        input_root = Path(input_root)
        required_data_root = Path(required_data_root)

        process_files(
            input_root / "poses",
            required_data_root / "ego_pose",
            "*.txt",
            pose_processor,
        )
        convert_images(input_root / "images", required_data_root / "images")
        process_files(
            input_root / "extrinsics_camera",
            required_data_root / "extrinsics",
            "*.txt",
            copy_processor,
        )
        process_files(
            input_root / "intrinsics_camera",
            required_data_root / "intrinsics",
            "*.txt",
            copy_processor,
        )
        process_files(
            input_root / "labels",
            required_data_root / "objects",
            "*.txt",
            labels_processor,
        )
        process_files(
            input_root / "pointclouds",
            required_data_root / "pointclouds",
            "*.ply",
            pointcloud_processor,
        )
        return True
    except Exception as e:
        print(f"转换数据集失败: {e}")
        return False


if __name__ == "__main__":
    ok = convert_dataset(Path("input"), Path("required_data"))
    exit(0 if ok else 1)
