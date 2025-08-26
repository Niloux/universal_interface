#!/usr/bin/env python3
"""
将track_info.pkl文件转换为JSON格式的脚本
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，确保JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def pkl_to_json(pkl_path, json_path=None):
    """
    将pickle文件转换为JSON格式

    Args:
        pkl_path (str): pickle文件路径
        json_path (str): 输出JSON文件路径，如果为None则自动生成
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"找不到文件: {pkl_path}")

        print(f"正在加载pickle文件: {pkl_path}")

        # 加载pickle文件
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        print(f"数据加载成功，类型: {type(data)}")

        if isinstance(data, dict):
            print(f"数据包含 {len(data)} 个键")
            if len(data) > 0:
                first_key = list(data.keys())[0]
                first_value = data[first_key]
                print(f"第一个键: {first_key}, 类型: {type(first_key)}")
                print(f"第一个值类型: {type(first_value)}")
                if isinstance(first_value, dict):
                    print(f"第一个值包含 {len(first_value)} 个键")

        # 转换numpy类型
        print("正在转换数据类型...")
        converted_data = convert_numpy_types(data)

        # 确定输出文件路径
        if json_path is None:
            pkl_file = Path(pkl_path)
            json_path = pkl_file.with_suffix(".json")

        # 保存为JSON
        print(f"正在保存到: {json_path}")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

        print(f"转换完成！输出文件: {json_path}")

        # 显示文件大小信息
        pkl_size = os.path.getsize(pkl_path)
        json_size = os.path.getsize(json_path)
        print(f"文件大小: {pkl_size} bytes -> {json_size} bytes")

        return True

    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="将pickle文件转换为JSON格式")
    parser.add_argument("input", help="输入的pickle文件路径")
    parser.add_argument("-o", "--output", help="输出的JSON文件路径（可选）")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    if args.verbose:
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")

    # 执行转换
    success = pkl_to_json(args.input, args.output)

    if success:
        print("✅ 转换成功完成")
    else:
        print("❌ 转换失败")
        exit(1)


if __name__ == "__main__":
    main()
