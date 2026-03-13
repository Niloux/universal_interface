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

    负责根据图像信息，生成并保存天空/地面区域的掩码图像。
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
        self.ground_mask_output_path = self.output_path / "ground_mask"

        default_logger.info(f"天空掩码输出路径: {self.mask_output_path}")
        default_logger.info(f"地面掩码输出路径: {self.ground_mask_output_path}")

    def process(self):
        """
        处理天空/地面掩码
        """
        try:
            from pathlib import Path

            import cv2
            import numpy as np
            import torch
            from PIL import Image
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
            from tqdm import tqdm

            default_logger.info("开始处理天空/地面掩码")
            self.ensure_dir(self.mask_output_path)
            self.ensure_dir(self.ground_mask_output_path)

            if not self.images_path.exists():
                default_logger.warning(
                    f"图像目录不存在，跳过天空/地面掩码生成: {self.images_path}"
                )
                return False

            image_files = sorted(self.images_path.glob("*.png"))
            if not image_files:
                default_logger.warning(
                    f"图像目录为空，跳过天空/地面掩码生成: {self.images_path}"
                )
                return False

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 手动指定 bpe_path，避免 pkg_resources 在某些环境下无法找到资源的问题
            project_root = Path(__file__).resolve().parent.parent
            bpe_path = project_root / "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

            model = build_sam3_image_model(
                device=device,
                checkpoint_path=self.config.sam3_checkpoint,
                load_from_HF=False,
                bpe_path=str(bpe_path),
            )
            processor = Sam3Processor(model, device=device)

            def _masks_to_u8(masks, height: int, width: int) -> np.ndarray:
                if masks is None or masks.numel() == 0:
                    return np.zeros((height, width), dtype=np.uint8)
                if masks.ndim == 4:
                    masks_3d = masks.squeeze(1)
                else:
                    masks_3d = masks
                merged = masks_3d.any(dim=0)
                return merged.to(dtype=torch.uint8).cpu().numpy() * 255

            for image_path in tqdm(image_files, desc="生成天空/地面掩码"):
                try:
                    image = Image.open(image_path).convert("RGB")
                    state = processor.set_image(image)
                    processor.set_text_prompt(state=state, prompt="sky")
                    sky_mask_img = _masks_to_u8(
                        state.get("masks"), image.height, image.width
                    )

                    processor.set_text_prompt(state=state, prompt="road")
                    ground_mask_img = _masks_to_u8(
                        state.get("masks"), image.height, image.width
                    )

                    sky_out_path = self.mask_output_path / image_path.name
                    cv2.imwrite(str(sky_out_path), sky_mask_img)

                    ground_out_path = self.ground_mask_output_path / image_path.name
                    cv2.imwrite(str(ground_out_path), ground_mask_img)
                except Exception as inner_e:
                    default_logger.warning(
                        f"生成天空/地面掩码失败，已跳过: {image_path.name}, err={inner_e}"
                    )

            default_logger.success("天空/地面掩码生成完成。")
            return True
        except Exception as e:
            default_logger.error(f"处理天空/地面掩码时出错: {e}")
            return False
