import json

from utils.config import Config
from utils.logger import default_logger


def generate_timestamps(config: Config) -> None:
    """生成时间戳文件timestamps.json，步进为0.1秒

    行为说明：
    - 优先从输入目录`input/ego_pose`扫描帧号，确定输出的帧数量与编号；
    - 使用0作为起始时间；
    - 以0.1秒为固定步进为每一帧生成时间戳，键为零填充6位的字符串（如`000000`）。

    输出位置：
    - 在output目录生成`timestamps.json`，包含一个顶层键`FRAME`，其值为帧到时间戳的映射。
    """
    try:
        input_root = config.input
        output_root = config.output
        output_root.mkdir(parents=True, exist_ok=True)

        frame_ids = []

        ego_dir = input_root / "ego_pose"
        if ego_dir.exists():
            try:
                stems = [p.stem for p in ego_dir.iterdir() if p.is_file() and p.suffix == ".txt" and p.stem.isdigit()]
                frame_ids = sorted(stems, key=lambda s: int(s))
            except Exception:
                default_logger.error("从ego_pose目录读取帧号失败，抛出错误")
                raise

        num_frames = len(frame_ids)
        step = 0.1
        start_time = 0.0

        frame_keys = [f"{i:06d}" for i in range(num_frames)]
        timestamps = {k: start_time + i * step for i, k in enumerate(frame_keys)}

        out_data = {"FRAME": timestamps}

        try:
            cam_ids = config.camera.id_map.values()
        except Exception:
            default_logger.error("无法从配置中读取相机ID,抛出错误")
            raise

        for cam_id in cam_ids:
            out_data[str(cam_id)] = dict(timestamps)

        out_path = output_root / "timestamps.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

        default_logger.success(f"时间戳文件生成成功: {out_path}")
    except Exception as e:
        default_logger.error(f"生成时间戳文件失败: {e}")
