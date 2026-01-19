from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class CameraDiscovery:
    camera_ids: List[str]
    image_dir_by_id: Dict[str, str]


def discover_cameras(required_data_root: Path) -> CameraDiscovery:
    """从标准化数据目录中自动发现相机集合与图像子目录映射。

    发现优先级：
    1) 从 `intrinsics/*.txt` 与 `extrinsics/*.txt` 的数字文件名推断 camera_id；
    2) 否则从 `images/` 下的子目录推断：
       - 子目录名为数字：目录名即 camera_id；
       - 子目录名非数字：按字典序排序后映射为 0..N-1。

    Args:
        required_data_root: `required_data` 根目录路径。

    Returns:
        CameraDiscovery: 包含 `camera_ids` 以及 `image_dir_by_id`（camera_id -> images子目录名）。

    Raises:
        FileNotFoundError: 当无法从 intr/extr/images 推断任何相机时抛出。
    """
    required_data_root = Path(required_data_root)

    numeric_ids: set[str] = set()
    for sub in ("intrinsics", "extrinsics"):
        d = required_data_root / sub
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() == ".txt" and p.stem.isdigit():
                numeric_ids.add(p.stem)

    if numeric_ids:
        ids = sorted(numeric_ids, key=lambda s: int(s))
        return CameraDiscovery(camera_ids=ids, image_dir_by_id={cid: cid for cid in ids})

    images_dir = required_data_root / "images"
    if images_dir.exists():
        subdirs = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])
        if subdirs:
            if all(name.isdigit() for name in subdirs):
                ids = sorted(subdirs, key=lambda s: int(s))
                return CameraDiscovery(camera_ids=ids, image_dir_by_id={cid: cid for cid in ids})

            ids = [str(i) for i in range(len(subdirs))]
            return CameraDiscovery(camera_ids=ids, image_dir_by_id={cid: name for cid, name in zip(ids, subdirs)})

    raise FileNotFoundError(f"无法在 {required_data_root} 中自动发现相机（缺少 intrinsics/extrinsics/images）")

