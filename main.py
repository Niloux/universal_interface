"""Universal Interface 主程序入口

3DGS训练通用数据接口，将自定义数据集转换成3DGS训练数据格式。
"""

from core.camera import CameraProcessor
from core.ego_pose import EgoPoseProcessor
from core.lidar import PointCloudProcessor
from core.track import TrackProcessor
from utils.config import Config
from utils.logger import default_logger


def main() -> int:
    """主函数：按顺序执行所有数据处理任务"""
    try:
        default_logger.info("Universal Interface 启动")

        # 加载配置
        config = Config()
        default_logger.success("配置加载成功")

        # -- 按顺序显式执行数据处理流水线 --
        processors = [
            (EgoPoseProcessor, "Ego-Pose数据处理"),
            (CameraProcessor, "相机内外参及图像处理"),
            (TrackProcessor, "轨迹和动态物体处理"),
            (PointCloudProcessor, "激光雷达数据处理"),
            # (DynamicMaskProcessor, "动态物体掩码生成"),
        ]

        for processor_class, name in processors:
            default_logger.info(f"--- 开始运行: {name} ---")
            processor = processor_class(config)
            processor.process()
            default_logger.success(f"--- {name} 运行成功 ---")

        default_logger.success("所有数据处理任务完成！")
        return 0

    except Exception as e:
        default_logger.error(f"程序执行失败: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
