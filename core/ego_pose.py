"""EgoPose数据处理器模块

专门处理ego_pose类型的数据，实现数据读取、转换和输出功能。
"""

import os
from typing import List

from tqdm import tqdm

from utils.config import Config

from .base import BaseProcessor


class EgoPoseProcessor(BaseProcessor):
    """EgoPose数据处理器

    继承BaseProcessor，专门处理ego_pose类型的数据。
    负责将输入的ego_pose数据转换为3DGS训练所需的格式。

    Attributes:
        input_path: ego_pose数据输入路径
        output_path: 处理后数据输出路径
    """

    def __init__(self, config: Config) -> None:
        """初始化EgoPose处理器

        Args:
            config: 配置管理对象

        Raises:
            KeyError: 当配置文件缺少必要配置项时
        """
        super().__init__(config)

        # 确保输出根目录存在
        self.ensure_dir(self.config["output"])

        # 设置输入输出路径
        self.input_path = os.path.join(self.config["input"], "ego_pose")
        self.output_path = os.path.join(self.config["output"], "ego_pose")

    def process(self) -> None:
        """处理ego_pose数据

        读取输入目录中的所有文件，进行格式转换后输出到目标目录。
        如果输入目录不存在，则跳过处理。

        Raises:
            OSError: 当文件读写操作失败时
        """
        # 检查输入目录是否存在
        if not self.check_dir(self.input_path):
            print(f"ego pose目录不存在: {self.input_path}, 跳过处理")
            return

        print(f"ego pose目录存在: {self.input_path}, 开始处理")

        # 确保输出目录存在
        self.ensure_dir(self.output_path)

        # 获取所有需要处理的文件
        input_files = self._get_input_files()

        if not input_files:
            print("没有找到需要处理的文件")
            return

        # 处理每个文件
        for filename in tqdm(input_files, desc="处理ego_pose文件"):
            try:
                self._process_single_file(filename)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                continue

        print(f"ego pose数据处理完成, 输出目录: {self.output_path}")

    def _get_input_files(self) -> List[str]:
        """获取输入目录中的所有文件

        Returns:
            文件名列表，只包含文件，不包含子目录
        """
        try:
            all_items = os.listdir(self.input_path)
            # 只返回文件，过滤掉目录
            files = [
                item
                for item in all_items
                if os.path.isfile(os.path.join(self.input_path, item))
            ]
            return files
        except OSError as e:
            print(f"读取输入目录失败: {e}")
            return []

    def _process_single_file(self, filename: str) -> None:
        """处理单个文件

        Args:
            filename: 要处理的文件名

        Raises:
            OSError: 当文件读写操作失败时
        """
        input_file_path = os.path.join(self.input_path, filename)
        output_file_path = os.path.join(self.output_path, filename)

        # 读取输入文件
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = f.read()

        # 在这里可以添加数据转换逻辑
        processed_data = self._transform_data(data)

        # 写入输出文件
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(processed_data)

    def _transform_data(self, data: str) -> str:
        """转换数据格式

        Args:
            data: 原始数据内容

        Returns:
            转换后的数据内容
        """
        return data
