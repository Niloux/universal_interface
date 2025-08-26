"""处理器工厂模块

提供处理器的创建和管理功能，支持根据配置动态创建处理器实例。
"""

from typing import Dict, Type, List

from utils.config import Config
from .base import BaseProcessor
from .ego_pose import EgoPoseProcessor


class ProcessorFactory:
    """处理器工厂类
    
    负责根据配置创建和管理不同类型的数据处理器。
    支持动态注册新的处理器类型。
    """
    
    # 注册的处理器类型映射
    _processors: Dict[str, Type[BaseProcessor]] = {
        "ego_pose": EgoPoseProcessor,
        # 可以在这里添加更多处理器类型
        # "camera": CameraProcessor,
        # "lidar": LidarProcessor,
    }
    
    @classmethod
    def register_processor(cls, name: str, processor_class: Type[BaseProcessor]) -> None:
        """注册新的处理器类型
        
        Args:
            name: 处理器名称
            processor_class: 处理器类
            
        Raises:
            ValueError: 当处理器名称已存在时
        """
        if name in cls._processors:
            raise ValueError(f"处理器 '{name}' 已经注册")
        
        if not issubclass(processor_class, BaseProcessor):
            raise ValueError(f"处理器类必须继承自 BaseProcessor")
            
        cls._processors[name] = processor_class
    
    @classmethod
    def create_processor(cls, name: str, config: Config) -> BaseProcessor:
        """创建指定类型的处理器
        
        Args:
            name: 处理器名称
            config: 配置对象
            
        Returns:
            处理器实例
            
        Raises:
            ValueError: 当处理器类型不存在时
        """
        if name not in cls._processors:
            raise ValueError(f"未知的处理器类型: {name}")
            
        processor_class = cls._processors[name]
        return processor_class(config)
    
    @classmethod
    def get_available_processors(cls) -> List[str]:
        """获取所有可用的处理器类型
        
        Returns:
            处理器名称列表
        """
        return list(cls._processors.keys())
    
    @classmethod
    def create_enabled_processors(cls, config: Config) -> List[BaseProcessor]:
        """根据配置创建所有启用的处理器
        
        Args:
            config: 配置对象
            
        Returns:
            启用的处理器实例列表
        """
        processors = []
        
        for processor_name in cls._processors.keys():
            if config.is_enable(processor_name):
                try:
                    processor = cls.create_processor(processor_name, config)
                    processors.append(processor)
                    print(f"✓ 已启用处理器: {processor_name}")
                except Exception as e:
                    print(f"✗ 创建处理器 {processor_name} 失败: {e}")
        
        return processors