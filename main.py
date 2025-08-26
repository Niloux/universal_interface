"""Universal Interface 主程序入口

3DGS训练通用数据接口，将自定义数据集转换成3DGS训练数据格式。
"""

from core import ProcessorFactory
from utils import Config, logger


def main() -> None:
    """主函数：根据配置文件动态执行数据处理任务"""
    try:
        logger.info("Universal Interface 启动")

        # 加载配置
        config = Config()
        logger.success("配置加载成功")

        # 显示可用的处理器
        available_processors = ProcessorFactory.get_available_processors()
        logger.info(f"可用的处理器: {', '.join(available_processors)}")

        # 创建所有启用的处理器
        processors = ProcessorFactory.create_enabled_processors(config)

        if not processors:
            logger.warning("没有启用任何处理器，请检查配置文件")
            return 1

        logger.info(f"开始处理数据，共启用 {len(processors)} 个处理器")

        # 执行所有处理器
        success_count = 0
        for processor in processors:
            processor_name = processor.__class__.__name__
            logger.info(f"正在执行: {processor_name}")
            try:
                processor.process()
                logger.success(f"{processor_name} 执行完成")
                success_count += 1
            except Exception as e:
                logger.error(f"{processor_name} 执行失败: {e}")

        if success_count == len(processors):
            logger.success("所有数据处理任务完成！")
        else:
            logger.warning(f"部分任务失败，成功: {success_count}/{len(processors)}")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
