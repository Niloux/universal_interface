"""处理器基类
提供通用功能和接口
"""

from abc import ABC


class BaseProcessor(ABC):
    """处理器基类"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
