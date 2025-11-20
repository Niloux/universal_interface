"""EgoPose数据处理器模块

专门处理ego_pose类型的数据，实现数据读取、转换和输出功能。
"""

from pathlib import Path
from typing import List

from tqdm import tqdm

from utils.config import Config
from utils.logger import default_logger

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
        """
        super().__init__(config)

        # 确保输出根目录存在
        self.ensure_dir(self.config.output)

        # 设置输入输出路径
        self.input_path = self.config.input / "ego_pose"
        self.output_path = self.config.output / "ego_pose"

    def process(self) -> None:
        """处理ego_pose数据

        读取输入目录中的所有文件，进行格式转换后输出到目标目录。
        如果输入目录不存在，则跳过处理。
        """
        # 检查输入目录是否存在
        if not self.check_dir(self.input_path):
            default_logger.warning(f"ego pose目录不存在: {self.input_path}, 跳过处理")
            return

        default_logger.info(f"ego pose目录存在: {self.input_path}, 开始处理")

        # 确保输出目录存在
        self.ensure_dir(self.output_path)

        # 获取所有需要处理的文件
        input_files = self._get_input_files()

        if not input_files:
            default_logger.warning("没有找到需要处理的ego_pose文件")
            return

        # 处理每个文件
        for filename in tqdm(input_files, desc="处理ego_pose文件"):
            try:
                self._process_single_file(filename)
            except Exception as e:
                default_logger.error(f"处理文件 {filename} 时出错: {e}")
                continue

        default_logger.info(f"ego pose数据处理完成, 输出目录: {self.output_path}")

    def _get_input_files(self) -> List[Path]:
        """获取输入目录中的所有文件

        Returns:
            文件名列表，只包含文件，不包含子目录
        """
        try:
            # 只返回文件，过滤掉目录
            files = [item for item in self.input_path.iterdir() if item.is_file()]
            return files
        except OSError as e:
            default_logger.error(f"读取输入目录失败: {e}")
            return []

    def _process_single_file(self, file_path: Path) -> None:
        """处理单个文件

        Args:
            file_path: 要处理的文件路径
        """
        output_file_path = self.output_path / file_path.name

        # 读取输入文件
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()

        # 在这里可以添加数据转换逻辑
        processed_data = self._transform_data(data)

        # 写入输出文件
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(processed_data)

        # 生成带相机编号的副本文件
        self._write_duplicates(file_path.stem, processed_data)

    def _transform_data(self, data: str) -> str:
        """转换数据格式

        Args:
            data: 原始数据内容

        Returns:
            转换后的数据内容
        """
        return data

    def _write_duplicates(self, frame_stem: str, content: str) -> None:
        """为指定帧生成副本文件

        将 `frame_stem.txt` 的内容复制到 `frame_stem_{i}.txt`，其中 i 来自相机id映射，
        默认为配置中的 `camera.id_map` 值集合。

        Args:
            frame_stem: 帧文件名的前缀（不含扩展名）
            content: 要写入副本的文本内容
        """
        try:
            ids = sorted({int(v) for v in self.config.camera.id_map.values()})
        except Exception:
            ids = list(range(7))

        for cam_id in ids:
            dup_path = self.output_path / f"{frame_stem}_{cam_id}.txt"
            try:
                with open(dup_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as e:
                default_logger.error(f"写入副本文件失败: {dup_path} -> {e}")

    def replicate_output_files(self) -> None:
        """为现有输出目录中的所有ego_pose帧生成副本文件

        遍历 `output/ego_pose` 下的 `XXXXXX.txt` 文件，为每个文件生成
        `XXXXXX_{i}.txt` 的副本，`i` 取自 `camera.id_map` 的值集合或 0..6。
        """
        if not self.check_dir(self.output_path):
            default_logger.warning(f"输出目录不存在: {self.output_path}")
            return

        try:
            ids = sorted({int(v) for v in self.config.camera.id_map.values()})
        except Exception:
            ids = list(range(7))

        files = [p for p in self.output_path.iterdir() if p.is_file() and p.suffix == ".txt" and "_" not in p.stem]
        if not files:
            default_logger.info("没有找到需要复制的ego_pose基础文件")
            return

        for base_file in files:
            try:
                with open(base_file, "r", encoding="utf-8") as f:
                    content = f.read()
                frame_stem = base_file.stem
                for cam_id in ids:
                    dup_path = self.output_path / f"{frame_stem}_{cam_id}.txt"
                    with open(dup_path, "w", encoding="utf-8") as f:
                        f.write(content)
            except Exception as e:
                default_logger.error(f"复制 {base_file.name} 失败: {e}")
                continue

        default_logger.success("ego_pose副本文件生成完成")
