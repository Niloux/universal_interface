from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from convert import convert_dataset
from core.camera import CameraProcessor
from core.dynamic_mask import DynamicMaskProcessor
from core.ego_pose import EgoPoseProcessor
from core.lidar import PointCloudProcessor
from core.sky_mask import SkyMaskProcessor
from core.track import TrackProcessor
from utils.camera_discovery import discover_cameras
from utils.logger import default_logger
from utils.runtime_config import RuntimeConfig
from utils.timestamp import generate_timestamps


def _load_optional_sam3_checkpoint(project_root: Path) -> Optional[str]:
    """尝试从项目根目录的 config.yaml 读取 sam3_checkpoint（若存在）。"""
    config_path = Path(project_root) / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ckpt = cfg.get("sam3_checkpoint")
        return str(ckpt) if ckpt else None
    except Exception:
        return None


def camera_tf(data_path: str | Path, data_save_path: str | Path) -> int:
    """对外上层接口：将 input 目录结构一键转换为 3DGS 训练所需输出。

    调用约定：
        info = camera_tf(data_path, data_save_path)

    Args:
        data_path: 原始输入数据根目录（包含 poses/images/extrinsics_camera/...）。
        data_save_path: 输出根目录（将生成 output 内容，同时生成 required_data 工作目录）。

    Returns:
        int: 1 表示处理成功；0 表示处理失败。
    """  # noqa: E501
    t0 = time.time()
    data_path = Path(data_path)
    data_save_path = Path(data_save_path)
    required_data_path = data_save_path / "required_data"
    run_info: Dict[str, Any] = {
        "ok": False,
        "stages": {},
        "paths": {
            "required_data": str(required_data_path),
            "output": str(data_save_path),
        },
    }

    try:
        data_save_path.mkdir(parents=True, exist_ok=True)

        default_logger.info(f"开始转换: input={data_path} -> output={data_save_path}")

        s0 = time.time()
        ok = convert_dataset(data_path, required_data_path)
        run_info["stages"]["convert"] = {"ok": bool(ok), "seconds": time.time() - s0}
        if not ok:
            run_info["error"] = "convert_failed"
            return 0

        discovery = discover_cameras(required_data_path)
        run_info["camera_ids"] = list(discovery.camera_ids)

        sam3_checkpoint = _load_optional_sam3_checkpoint(Path(__file__).parent)
        if sam3_checkpoint is not None:
            ckpt_path = Path(sam3_checkpoint)
            if not ckpt_path.is_absolute():
                ckpt_path = Path(__file__).parent / ckpt_path
            sam3_checkpoint = str(ckpt_path) if ckpt_path.exists() else None

        config = RuntimeConfig(
            input_path=required_data_path,
            output_path=data_save_path,
            camera_ids=discovery.camera_ids,
            sam3_checkpoint=sam3_checkpoint,
            image_dir_by_id=discovery.image_dir_by_id,
        )

        s0 = time.time()
        generate_timestamps(config)
        run_info["stages"]["timestamps"] = {"ok": True, "seconds": time.time() - s0}

        processors = [
            ("ego_pose", EgoPoseProcessor),
            ("camera", CameraProcessor),
            ("track", TrackProcessor),
            ("dynamic_mask", DynamicMaskProcessor),
            ("lidar", PointCloudProcessor),
        ]
        if sam3_checkpoint:
            processors.append(("sky_mask", SkyMaskProcessor))
        else:
            run_info.setdefault("skipped", []).append("sky_mask")

        for stage_name, processor_cls in processors:
            s0 = time.time()
            try:
                processor = processor_cls(config)
                result = processor.process()
                ok_stage = result is not False
            except Exception as e:
                default_logger.error(f"{stage_name} 阶段异常: {e}")
                ok_stage = False

            run_info["stages"][stage_name] = {
                "ok": bool(ok_stage),
                "seconds": time.time() - s0,
            }
            if not ok_stage:
                run_info["error"] = f"{stage_name}_failed"
                return 0

        run_info["ok"] = True
        return 1
    except Exception as e:
        default_logger.error(f"camera_tf 执行失败: {e}")
        run_info["error"] = "unhandled_exception"
        return 0
    finally:
        run_info["seconds"] = time.time() - t0
        try:
            with open(data_save_path / "run_info.json", "w", encoding="utf-8") as f:
                json.dump(run_info, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python api.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    info = camera_tf(input_path, output_path)
    print(info)
