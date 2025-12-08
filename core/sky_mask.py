"""
SkyMask数据处理模块

该模块负责处理天空区域的掩码图像。必须在图像处理模块之后运行。
"""

from utils.config import Config
from utils.logger import default_logger

from .base import BaseProcessor


class SkyMaskProcessor(BaseProcessor):
    """
    天空掩码处理器

    负责根据图像和深度信息，生成并保存天空区域的掩码图像。
    """

    def __init__(self, config: Config):
        """
        初始化SkyMaskProcessor

        Args:
            config: 配置管理对象
        """
        super().__init__(config)
        self.output_path = self.config.output

        # 定义输入输出路径
        self.images_path = self.output_path / "images"
        self.mask_output_path = self.output_path / "sky_mask"

        default_logger.info(f"天空掩码输出路径: {self.mask_output_path}")

    def process(self):
        """
        处理天空掩码
        """
        try:
            # TODO: 要用python3.12来装环境，后面再看
            default_logger.info("开始处理天空掩码")
            self.ensure_dir(self.mask_output_path)
        except Exception as e:
            default_logger.error(f"处理天空掩码时出错: {e}")
            return False
