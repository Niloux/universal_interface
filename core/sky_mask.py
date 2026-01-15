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
            import cv2
            import numpy as np
            import torch
            from PIL import Image
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
            from tqdm import tqdm

            default_logger.info("开始处理天空掩码")
            self.ensure_dir(self.mask_output_path)

            if not self.images_path.exists():
                default_logger.warning(
                    f"图像目录不存在，跳过天空掩码生成: {self.images_path}"
                )
                return False

            image_files = sorted(self.images_path.glob("*.png"))
            if not image_files:
                default_logger.warning(
                    f"图像目录为空，跳过天空掩码生成: {self.images_path}"
                )
                return False

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_sam3_image_model(
                device=device,
                checkpoint_path=self.config.sam3_checkpoint,
                load_from_HF=False,
            )
            processor = Sam3Processor(model, device=device)

            for image_path in tqdm(image_files, desc="生成天空掩码"):
                try:
                    image = Image.open(image_path).convert("RGB")
                    state = processor.set_image(image)
                    output = processor.set_text_prompt(state=state, prompt="sky")

                    masks = output.get("masks")
                    if masks is None or masks.numel() == 0:
                        mask_img = np.zeros((image.height, image.width), dtype=np.uint8)
                    else:
                        if masks.ndim == 4:
                            masks_3d = masks.squeeze(1)
                        else:
                            masks_3d = masks
                        merged = masks_3d.any(dim=0)
                        mask_img = merged.to(dtype=torch.uint8).cpu().numpy() * 255

                    out_path = self.mask_output_path / image_path.name
                    cv2.imwrite(str(out_path), mask_img)
                except Exception as inner_e:
                    default_logger.warning(
                        f"生成天空掩码失败，已跳过: {image_path.name}, err={inner_e}"
                    )

            default_logger.success("天空掩码生成完成。")
            return True
        except Exception as e:
            default_logger.error(f"处理天空掩码时出错: {e}")
            return False
