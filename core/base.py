"""处理器基类模块

提供数据处理器的抽象基类和通用功能。
所有具体的数据处理器都应该继承BaseProcessor类。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from utils.config import Config


class BaseProcessor(ABC):
    """数据处理器抽象基类

    定义了数据处理器的通用接口和工具方法。
    所有具体的处理器都必须继承此类并实现process方法。

    Attributes:
        config: 配置管理对象，包含所有配置信息
    """

    def __init__(self, config: Config) -> None:
        """初始化处理器

        Args:
            config: 配置管理对象
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def process(self) -> None:
        """处理数据的抽象方法

        子类必须实现此方法来定义具体的数据处理逻辑。
        """
        pass

    def get_folders(self, path: Path) -> List[Path]:
        """获取指定路径下的所有文件夹名称

        Args:
            path: 要扫描的目录路径

        Returns:
            文件夹路径列表

        Raises:
            OSError: 当路径不存在或无法访问时
        """
        if not path.exists():
            raise OSError(f"路径不存在: {path}")

        folders = [item for item in path.iterdir() if item.is_dir()]
        return folders

    def ensure_dir(self, path: Path) -> None:
        """确保目录存在，如果不存在则创建

        Args:
            path: 要创建的目录路径

        Raises:
            OSError: 当无法创建目录时
        """
        path.mkdir(parents=True, exist_ok=True)

    def check_dir(self, path: Path) -> bool:
        """检查目录是否存在

        Args:
            path: 要检查的目录路径

        Returns:
            目录存在返回True，否则返回False
        """
        return path.exists() and path.is_dir()
