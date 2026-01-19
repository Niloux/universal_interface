from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2
from plyfile import PlyData, PlyElement

from api import camera_tf


def _write_pose_txt(path: Path) -> None:
    """写入一个12数字的pose文本（convert会补齐为4x4）。"""
    numbers = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    path.write_text(" ".join(str(x) for x in numbers), encoding="utf-8")


def _write_extrinsics_txt(path: Path) -> None:
    """写入一个4x4单位矩阵外参（16数字）。"""
    mat = np.eye(4, dtype=np.float32).reshape(-1)
    path.write_text(" ".join(f"{x:.6f}" for x in mat.tolist()), encoding="utf-8")


def _write_intrinsics_txt(path: Path) -> None:
    """写入一个3x3相机内参（9数字，按行）。"""
    fx, fy, cx, cy = 1000.0, 1000.0, 640.0, 360.0
    k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    path.write_text(" ".join(f"{x:.6f}" for x in k), encoding="utf-8")


def _write_dummy_image(path: Path, seed: int) -> None:
    """写入一张小尺寸JPEG图像。"""
    rng = np.random.default_rng(seed)
    img = (rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"写图像失败: {path}")


def _write_pointcloud_ply(path: Path, seed: int) -> None:
    """写入一个最小的PLY点云文件（vertex包含x/y/z/intensity）。"""
    rng = np.random.default_rng(seed)
    n = 50
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    intensity = rng.uniform(0, 1, size=(n,)).astype(np.float32)
    vertex = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["intensity"] = intensity
    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(path))


class TestCameraTF(unittest.TestCase):
    def test_camera_tf_minimal_dataset(self) -> None:
        """构造最小输入数据，验证camera_tf端到端返回成功并产出关键文件。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_root = root / "input"
            output_root = root / "output"

            poses_dir = input_root / "poses"
            images_dir = input_root / "images"
            extr_dir = input_root / "extrinsics_camera"
            intr_dir = input_root / "intrinsics_camera"
            labels_dir = input_root / "labels"
            pc_dir = input_root / "pointclouds"

            poses_dir.mkdir(parents=True, exist_ok=True)
            extr_dir.mkdir(parents=True, exist_ok=True)
            intr_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            pc_dir.mkdir(parents=True, exist_ok=True)

            for frame in (0, 1):
                _write_pose_txt(poses_dir / f"{frame:06d}.txt")

            for cam in ("0", "1"):
                _write_extrinsics_txt(extr_dir / f"{cam}.txt")
                _write_intrinsics_txt(intr_dir / f"{cam}.txt")
                for frame in (0, 1):
                    _write_dummy_image(images_dir / cam / f"{frame:06d}.jpg", seed=frame + int(cam) * 10)

            label_lines = [
                "0 1 vehicle 0 0 10 4 2 2 0",
                "1 1 vehicle 20 0 10 4 2 2 0",
            ]
            (labels_dir / "labels.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

            for frame in (0, 1):
                _write_pointcloud_ply(pc_dir / f"{frame:06d}_0.ply", seed=frame)

            ret = camera_tf(input_root, output_root)
            self.assertEqual(ret, 1)

            self.assertTrue((output_root / "timestamps.json").exists())
            self.assertTrue((output_root / "images" / "000000_0.png").exists())
            self.assertTrue((output_root / "intrinsics" / "0.txt").exists())
            self.assertTrue((output_root / "extrinsics" / "0.txt").exists())
            self.assertTrue((output_root / "track" / "trajectory.pkl").exists())
            self.assertTrue((output_root / "dynamic_mask" / "000000_0.png").exists())
            self.assertTrue((output_root / "lidar" / "depth" / "000000_0.npz").exists())

