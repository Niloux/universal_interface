from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CameraCompatConfig:
    positions: List[str]
    id_map: Dict[str, int]


class RuntimeConfig:
    """运行时配置对象，用于替代必须依赖 YAML 的 Config。

    该对象提供与现有 Processor 兼容的属性：
    - input: Path，指向 required_data 根目录
    - output: Path，指向最终输出根目录
    - camera_ids: List[str]，本次数据的相机ID集合（字符串）
    - camera: CameraCompatConfig，兼容旧逻辑（positions/id_map）
    - sam3_checkpoint: Optional[str]，天空掩码权重路径（可选）
    """

    def __init__(
        self,
        *,
        input_path: Path,
        output_path: Path,
        camera_ids: List[str],
        sam3_checkpoint: Optional[str] = None,
        image_dir_by_id: Optional[Dict[str, str]] = None,
    ) -> None:
        """构造运行时配置。

        Args:
            input_path: required_data 根目录。
            output_path: output 根目录。
            camera_ids: 相机ID列表（字符串）。
            sam3_checkpoint: 可选的SAM3权重路径。
            image_dir_by_id: 可选的 `camera_id -> images子目录名` 映射，用于非数字相机目录兼容。
        """
        self.input = Path(input_path)
        self.output = Path(output_path)
        self.camera_ids = list(camera_ids)
        self.sam3_checkpoint = sam3_checkpoint
        self.image_dir_by_id = dict(image_dir_by_id or {cid: cid for cid in self.camera_ids})

        id_map = {cid: int(cid) if str(cid).isdigit() else i for i, cid in enumerate(self.camera_ids)}
        self.camera = CameraCompatConfig(positions=list(self.image_dir_by_id.values()), id_map=id_map)

