"""日志工具模块

提供简单的日志记录功能，支持不同级别的日志输出。
"""

import sys
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class Logger:
    """简单的日志记录器

    提供带时间戳和级别标识的日志输出功能。
    """

    # 日志级别对应的颜色代码
    _COLORS = {
        LogLevel.DEBUG: "\033[36m",  # 青色
        LogLevel.INFO: "\033[37m",  # 白色
        LogLevel.WARNING: "\033[33m",  # 黄色
        LogLevel.ERROR: "\033[31m",  # 红色
        LogLevel.SUCCESS: "\033[32m",  # 绿色
    }

    _RESET = "\033[0m"  # 重置颜色

    def __init__(self, name: str = "UniversalInterface", enable_color: bool = True):
        """初始化日志记录器

        Args:
            name: 日志记录器名称
            enable_color: 是否启用颜色输出
        """
        self.name = name
        self.enable_color = enable_color

    def _format_message(self, level: LogLevel, message: str) -> str:
        """格式化日志消息

        Args:
            level: 日志级别
            message: 日志消息

        Returns:
            格式化后的日志消息
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_str = level.value

        if self.enable_color:
            color = self._COLORS.get(level, "")
            formatted = f"{color}[{timestamp}] {level_str:>7}: {message}{self._RESET}"
        else:
            formatted = f"[{timestamp}] {level_str:>7}: {message}"

        return formatted

    def debug(self, message: str) -> None:
        """输出调试日志"""
        print(self._format_message(LogLevel.DEBUG, message))

    def info(self, message: str) -> None:
        """输出信息日志"""
        print(self._format_message(LogLevel.INFO, message))

    def warning(self, message: str) -> None:
        """输出警告日志"""
        print(self._format_message(LogLevel.WARNING, message), file=sys.stderr)

    def error(self, message: str) -> None:
        """输出错误日志"""
        print(self._format_message(LogLevel.ERROR, message), file=sys.stderr)

    def success(self, message: str) -> None:
        """输出成功日志"""
        print(self._format_message(LogLevel.SUCCESS, message))

    def log(self, level: LogLevel, message: str) -> None:
        """输出指定级别的日志

        Args:
            level: 日志级别
            message: 日志消息
        """
        if level in [LogLevel.WARNING, LogLevel.ERROR]:
            print(self._format_message(level, message), file=sys.stderr)
        else:
            print(self._format_message(level, message))


# 创建默认的日志记录器实例
default_logger = Logger()


# 提供便捷的模块级函数
def debug(message: str) -> None:
    """输出调试日志"""
    default_logger.debug(message)


def info(message: str) -> None:
    """输出信息日志"""
    default_logger.info(message)


def warning(message: str) -> None:
    """输出警告日志"""
    default_logger.warning(message)


def error(message: str) -> None:
    """输出错误日志"""
    default_logger.error(message)


def success(message: str) -> None:
    """输出成功日志"""
    default_logger.success(message)
