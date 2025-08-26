"""处理器基类
提供通用功能和接口
"""

import os
from abc import ABC, abstractmethod
from typing import List

from utils.config import Config


class BaseProcessor(ABC):
    """处理器基类"""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def process(self) -> None:
        """处理数据"""
        pass

    def get_folders(self, path: str) -> List[str]:
        """获取所有文件夹的名称"""
        segments = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                segments.append(item)
        return segments

    def ensure_dir(self, path: str) -> None:
        """确保目录存在"""
        if not os.path.exists(path):
            os.makedirs(path)
