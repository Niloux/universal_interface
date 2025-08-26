from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """配置类"""

    def __init__(self, path: Optional[str] = None) -> None:
        """初始化"""
        if path is None:
            root = Path(__file__).parent.parent
            path = root / "config.yaml"
        self._config = self._load_config(path)

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"配置文件格式错误: {e}")

    @property
    def config(self) -> Dict[str, Any]:
        """获取配置"""
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """获取配置项"""
        return self._config[key]
