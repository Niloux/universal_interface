"""配置管理模块

提供一个经过验证的、支持属性式访问的配置类。
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


class CameraConfig:
    """封装相机相关的配置"""

    def __init__(self, camera_dict: Dict[str, Any]):
        self.positions: List[str] = camera_dict.get("positions", [])
        self.id_map: Dict[str, int] = camera_dict.get("id_map", {})

        if not self.positions or not self.id_map:
            raise ValueError("配置文件中缺少相机 'positions' 或 'id_map'")


class Config:
    """
    配置类，提供加载、验证和属性式访问功能。

    示例:
        config = Config()
        input_path = config.input  # 属性式访问
        camera_positions = config.camera.positions
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        """
        初始化并加载配置。

        Args:
            path: 配置文件路径。如果为None，则默认加载项目根目录下的 'config.yaml'。

        Raises:
            FileNotFoundError: 如果配置文件不存在。
            ValueError: 如果缺少必要的配置项。
        """
        if path is None:
            path = Path(__file__).parent.parent / "config.yaml"

        self._config = self._load_and_validate(path)

        # 将顶层键作为属性动态添加到实例上
        for key, value in self._config.items():
            if key == "camera":
                setattr(self, key, CameraConfig(value))
            elif key in ["input", "output"]:
                # 将输入输出路径转换为Path对象
                setattr(self, key, Path(value))
            else:
                setattr(self, key, value)

    def _load_and_validate(self, path: Path) -> Dict[str, Any]:
        """加载并验证配置文件"""
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"配置文件格式错误: {e}")

        if not isinstance(config, dict):
            raise TypeError("配置文件根节点必须是字典类型")

        # 验证必要的键是否存在
        required_keys = ["input", "output", "camera"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置文件中缺少必要的键: '{key}'")

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        提供类似字典的get方法以安全地访问配置项。

        Args:
            key: 配置项的键。
            default: 如果键不存在时返回的默认值。

        Returns:
            配置项的值或默认值。
        """
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config(keys={list(self._config.keys())})"
