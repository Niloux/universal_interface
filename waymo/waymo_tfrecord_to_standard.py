#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from plyfile import PlyData, PlyElement

try:
    from waymo_open_dataset import dataset_pb2, label_pb2
    from waymo_open_dataset.utils import frame_utils
except ImportError as exc:
    raise ImportError(
        "请先安装 Waymo 官方 devkit，例如: pip install waymo-open-dataset-tf-2-6-0"
    ) from exc


WAYMO_CAMERA_ENUM_TO_ID = {
    dataset_pb2.CameraName.FRONT: 0,
    dataset_pb2.CameraName.FRONT_LEFT: 1,
    dataset_pb2.CameraName.FRONT_RIGHT: 2,
    dataset_pb2.CameraName.SIDE_LEFT: 3,
    dataset_pb2.CameraName.SIDE_RIGHT: 4,
}

WAYMO_LIDAR_ENUM_TO_NAME = {
    dataset_pb2.LaserName.TOP: "TOP",
    dataset_pb2.LaserName.FRONT: "FRONT",
    dataset_pb2.LaserName.SIDE_LEFT: "SIDE_LEFT",
    dataset_pb2.LaserName.SIDE_RIGHT: "SIDE_RIGHT",
    dataset_pb2.LaserName.REAR: "REAR",
}

WAYMO_LABEL_TYPE_TO_NAME = {
    label_pb2.Label.TYPE_UNKNOWN: "unknown",
    label_pb2.Label.TYPE_VEHICLE: "vehicle",
    label_pb2.Label.TYPE_PEDESTRIAN: "pedestrian",
    label_pb2.Label.TYPE_SIGN: "sign",
    label_pb2.Label.TYPE_CYCLIST: "cyclist",
}

ORDERED_LIDARS = [
    dataset_pb2.LaserName.TOP,
    dataset_pb2.LaserName.FRONT,
    dataset_pb2.LaserName.SIDE_LEFT,
    dataset_pb2.LaserName.SIDE_RIGHT,
    dataset_pb2.LaserName.REAR,
]


def ensure_dirs(root: Path) -> Dict[str, Path]:
    """创建标准输出目录结构并返回各目录路径。"""
    dirs = {
        "images": root / "images",
        "pointclouds": root / "pointclouds",
        "labels": root / "labels",
        "intrinsics_camera": root / "intrinsics_camera",
        "intrinsics_lidar": root / "intrinsics_lidar",
        "extrinsics_camera": root / "extrinsics_camera",
        "extrinsics_lidar": root / "extrinsics_lidar",
        "poses": root / "poses",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def flatten_3x4_line(mat4x4: np.ndarray) -> str:
    """将4x4矩阵取前三行并按3x4平铺为一行字符串。"""
    mat = np.asarray(mat4x4, dtype=np.float64).reshape(4, 4)[:3, :]
    return " ".join(f"{v:.15f}" for v in mat.reshape(-1))


def get_camera_resolution(frame: dataset_pb2.Frame, calib) -> Tuple[int, int]:
    """获取相机分辨率(width,height)。

    优先使用 Waymo camera_calibration 中的宽高字段；若缺失或为0，
    则回退到首帧 frame.images 中对应相机图像的宽高字段；仍缺失则返回(0,0)。
    """
    w = int(getattr(calib, "width", 0) or getattr(calib, "image_width", 0) or 0)
    h = int(getattr(calib, "height", 0) or getattr(calib, "image_height", 0) or 0)

    if w > 0 and h > 0:
        return w, h

    cam_name = getattr(calib, "name", None)
    if cam_name is None:
        return 0, 0

    for img in getattr(frame, "images", []):
        if getattr(img, "name", None) != cam_name:
            continue
        w2 = int(getattr(img, "width", 0) or getattr(img, "image_width", 0) or 0)
        h2 = int(getattr(img, "height", 0) or getattr(img, "image_height", 0) or 0)
        if w2 > 0 and h2 > 0:
            return w2, h2

    return 0, 0


def _convert_waymo_camera_extrinsic_to_opencv(
    extrinsic_cam_to_vehicle: np.ndarray,
) -> np.ndarray:  # noqa: E501
    """将 Waymo 相机外参(cam->vehicle)转换为 OpenCV 相机坐标系定义的 cam->vehicle。

    背景：
        - Waymo 相机坐标系：x 沿光轴向前，z 向上（官方说明）。
        - 本项目投影/深度实现：假设相机坐标系 z 为前方深度（OpenCV 风格）。

    目标：把 Waymo 的 cam 坐标轴重排成 OpenCV 风格（z前, x右, y下），
    使得后续的 `project_to_2d()` / 深度图生成逻辑能够正确使用 Z 作为深度。

    坐标轴映射（点坐标）：
        X_cv = -Y_waymo
        Y_cv = -Z_waymo
        Z_cv =  X_waymo

    对 cam->vehicle 外参的处理：
        设 p_waymo = R_waymo_from_cv * p_cv
        则 E_cv = E_waymo @ T_waymo_from_cv

    Args:
        extrinsic_cam_to_vehicle: Waymo 定义的 cam->vehicle 外参，shape (4,4)。

    Returns:
        OpenCV 定义的 cam->vehicle 外参，shape (4,4)。
    """
    e = np.asarray(extrinsic_cam_to_vehicle, dtype=np.float64).reshape(4, 4)

    r_waymo_from_cv = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
    )
    t_waymo_from_cv = np.eye(4, dtype=np.float64)
    t_waymo_from_cv[:3, :3] = r_waymo_from_cv

    return e @ t_waymo_from_cv


def save_camera_calibrations(
    frame: dataset_pb2.Frame, out_dirs: Dict[str, Path]
) -> None:
    """从首帧写出相机内外参到标准目录。"""
    for calib in frame.context.camera_calibrations:
        if calib.name not in WAYMO_CAMERA_ENUM_TO_ID:
            continue
        cam_id = WAYMO_CAMERA_ENUM_TO_ID[calib.name]

        intr = list(calib.intrinsic)
        if len(intr) < 4:
            continue
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
        width, height = get_camera_resolution(frame, calib)

        k = [
            float(width),
            float(height),
            fx,
            0.0,
            cx,
            0.0,
            fy,
            cy,
            0.0,
            0.0,
            1.0,
        ]
        (out_dirs["intrinsics_camera"] / f"{cam_id}.txt").write_text(
            " ".join(f"{v:.15f}" for v in k), encoding="utf-8"
        )

        ext_waymo = np.array(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        ext = _convert_waymo_camera_extrinsic_to_opencv(ext_waymo)
        (out_dirs["extrinsics_camera"] / f"{cam_id}.txt").write_text(
            flatten_3x4_line(ext), encoding="utf-8"
        )


def parse_top_lidar_calib(
    frame: dataset_pb2.Frame,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """提取 TOP 雷达外参及用于 intrinsics_lidar/0.txt 的参数。"""
    for lc in frame.context.laser_calibrations:
        if lc.name != dataset_pb2.LaserName.TOP:
            continue

        ext = np.array(lc.extrinsic.transform, dtype=np.float64).reshape(4, 4)

        if len(lc.beam_inclinations) > 0:
            incl = np.asarray(lc.beam_inclinations, dtype=np.float64)
            v_up = float(np.rad2deg(np.max(incl)))
            v_down = float(np.rad2deg(np.min(incl)))
            num_lines = int(incl.size)
        else:
            v_up = float(np.rad2deg(lc.beam_inclination_max))
            v_down = float(np.rad2deg(lc.beam_inclination_min))
            num_lines = 64

        meta = {
            "min_range": 0.0,
            "max_range": 80.0,
            "h_fov_deg": 360.0,
            "v_up_deg": v_up,
            "v_down_deg": v_down,
            "num_lines": float(num_lines),
            "num_cols": 2650.0,
        }
        return ext, meta

    raise ValueError("未找到 TOP 雷达标定信息")


def save_lidar_files(
    top_ext: np.ndarray, lidar_meta: Dict[str, float], out_dirs: Dict[str, Path]
) -> None:
    """写出0号雷达外参和内参文件。"""
    (out_dirs["extrinsics_lidar"] / "0.txt").write_text(
        flatten_3x4_line(top_ext), encoding="utf-8"
    )
    vals = [
        lidar_meta["min_range"],
        lidar_meta["max_range"],
        lidar_meta["h_fov_deg"],
        lidar_meta["v_up_deg"],
        lidar_meta["v_down_deg"],
        lidar_meta["num_lines"],
        lidar_meta["num_cols"],
    ]
    (out_dirs["intrinsics_lidar"] / "0.txt").write_text(
        " ".join(f"{v:.15f}" for v in vals), encoding="utf-8"
    )


def transform_vehicle_to_lidar(
    points_vehicle: np.ndarray, lidar_to_vehicle_4x4: np.ndarray
) -> np.ndarray:
    """将车辆坐标系点云转换到指定雷达坐标系，输入格式为[N,6]。"""
    if points_vehicle.size == 0:
        return points_vehicle
    xyz_vehicle = points_vehicle[:, 3:6]
    xyz_h = np.concatenate(
        [xyz_vehicle, np.ones((xyz_vehicle.shape[0], 1), dtype=np.float32)], axis=1
    )
    vehicle_to_lidar = np.linalg.inv(lidar_to_vehicle_4x4).astype(np.float32)
    xyz_lidar = (xyz_h @ vehicle_to_lidar.T)[:, :3]
    return np.concatenate([points_vehicle[:, :3], xyz_lidar], axis=1)


def intensity_to_u8(intensity: np.ndarray) -> np.ndarray:
    """将点云强度归一化到0~255的uint8。"""
    intensity = np.asarray(intensity, dtype=np.float32)
    if intensity.size == 0:
        return intensity.astype(np.uint8)
    finite = np.isfinite(intensity)
    if not np.any(finite):
        return np.zeros_like(intensity, dtype=np.uint8)

    vals = intensity[finite]
    min_v, max_v = float(np.min(vals)), float(np.max(vals))
    if min_v >= 0.0 and max_v <= 1.0:
        scaled = intensity * 255.0
    else:
        lo, hi = float(np.percentile(vals, 1.0)), float(np.percentile(vals, 99.0))
        if hi <= lo + 1e-6:
            scaled = np.zeros_like(intensity, dtype=np.float32)
        else:
            scaled = (intensity - lo) / (hi - lo) * 255.0
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def save_merged_top_ply(points_top: np.ndarray, ply_path: Path) -> None:
    """保存合并后的0号雷达点云为PLY（x,y,z,intensity）。"""
    if points_top.size == 0:
        vertex = np.empty(
            0, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "u1")]
        )
    else:
        intensity = intensity_to_u8(points_top[:, 1])
        vertex = np.empty(
            points_top.shape[0],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "u1")],
        )
        vertex["x"] = points_top[:, 3].astype(np.float32)
        vertex["y"] = points_top[:, 4].astype(np.float32)
        vertex["z"] = points_top[:, 5].astype(np.float32)
        vertex["intensity"] = intensity

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(ply_path))


def write_pose(frame: dataset_pb2.Frame, pose_path: Path) -> None:
    """写出车辆到世界坐标的pose为3x4一行格式。"""
    pose4 = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    pose_path.write_text(flatten_3x4_line(pose4), encoding="utf-8")


def dump_images(
    frame: dataset_pb2.Frame, frame_idx: int, out_dirs: Dict[str, Path]
) -> None:
    """写出当前帧所有相机图像，命名为{frame}_{cam}.jpg。"""
    for img in frame.images:
        if img.name not in WAYMO_CAMERA_ENUM_TO_ID:
            continue
        cam_id = WAYMO_CAMERA_ENUM_TO_ID[img.name]
        out_path = out_dirs["images"] / f"{frame_idx:06d}_{cam_id}.jpg"
        out_path.write_bytes(img.image)


def collect_label_lines(frame: dataset_pb2.Frame, frame_idx: int) -> List[str]:
    """提取当前帧3D框并转换为标准label文本行。"""
    lines: List[str] = []
    for lb in frame.laser_labels:
        # NOTE: 过滤掉交通灯TYPE_SIGN类型的标签
        if lb.type == label_pb2.Label.TYPE_SIGN:
            continue
        t = WAYMO_LABEL_TYPE_TO_NAME.get(lb.type, "unknown")
        b = lb.box
        line = (
            f"{frame_idx} {lb.id} {t} "
            f"{b.center_x:.6f} {b.center_y:.6f} {b.center_z:.6f} "
            f"{b.length:.6f} {b.width:.6f} {b.height:.6f} {b.heading:.6f}"
        )
        lines.append(line)
    return lines


def process_one_tfrecord(tfrecord_path: Path, output_root: Path) -> None:
    """处理单个tfrecord并导出标准目录数据。"""
    out_dirs = ensure_dirs(output_root)

    first_raw = next(
        iter(tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")), None
    )
    if first_raw is None:
        raise ValueError(f"空tfrecord: {tfrecord_path}")

    first_frame = dataset_pb2.Frame()
    first_frame.ParseFromString(bytearray(first_raw.numpy()))

    save_camera_calibrations(first_frame, out_dirs)
    top_ext, lidar_meta = parse_top_lidar_calib(first_frame)
    save_lidar_files(top_ext, lidar_meta, out_dirs)

    label_lines: List[str] = []

    ds = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
    for frame_idx, raw in enumerate(ds):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(raw.numpy()))

        dump_images(frame, frame_idx, out_dirs)
        write_pose(frame, out_dirs["poses"] / f"{frame_idx:06d}.txt")
        label_lines.extend(collect_label_lines(frame, frame_idx))

        if not frame.lasers:
            continue

        range_images, camera_projections, _, range_image_top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )
        points_with_features, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0,
            keep_polar_features=True,
        )

        merged = []
        for i, lidar_enum in enumerate(ORDERED_LIDARS):
            _ = WAYMO_LIDAR_ENUM_TO_NAME[lidar_enum]
            pts_vehicle = points_with_features[i]
            if pts_vehicle.size == 0:
                continue
            pts_top = transform_vehicle_to_lidar(pts_vehicle, top_ext)
            merged.append(pts_top)

        if merged:
            merged_top = np.concatenate(merged, axis=0)
        else:
            merged_top = np.empty((0, 6), dtype=np.float32)

        save_merged_top_ply(
            merged_top, out_dirs["pointclouds"] / f"{frame_idx:06d}_0.ply"
        )

    (out_dirs["labels"] / "000000.txt").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8"
    )


def list_tfrecords(input_path: Path) -> List[Path]:
    """解析输入路径并返回待处理tfrecord文件列表。"""
    if input_path.is_file() and input_path.suffix == ".tfrecord":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.tfrecord"))
    return []


def main() -> None:
    """命令行入口：支持单文件或目录批处理。"""
    parser = argparse.ArgumentParser(description="Waymo TFRecord 转标准数据集脚本")
    parser.add_argument("--input_path", required=True, help="Waymo tfrecord文件或目录")
    parser.add_argument("--output_path", required=True, help="标准数据输出目录")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    tfrecords = list_tfrecords(input_path)
    if not tfrecords:
        raise FileNotFoundError(f"未找到tfrecord: {input_path}")

    if len(tfrecords) == 1 and input_path.is_file():
        process_one_tfrecord(tfrecords[0], output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        for tfr in tfrecords:
            seq_name = tfr.stem
            process_one_tfrecord(tfr, output_path / seq_name)


if __name__ == "__main__":
    main()
