#!/usr/bin/env python3
"""
姿态文件转换脚本
将final_data/poses目录下的txt文件从4行格式转换为1行格式
"""

import os
import glob
from pathlib import Path


def convert_pose_file(input_file: str, output_file: str) -> None:
    """
    转换单个姿态文件从4行格式到1行格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 移除每行末尾的换行符并合并所有数值
    all_values = []
    for line in lines:
        values = line.strip().split()
        all_values.extend(values)
    
    # 将所有数值写入一行，用空格分隔
    with open(output_file, 'w') as f:
        f.write(' '.join(all_values) + '\n')


def convert_all_poses(input_dir: str, output_dir: str) -> None:
    """
    转换目录下所有的姿态文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))
    txt_files.sort()  # 确保按文件名排序
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    for input_file in txt_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        convert_pose_file(input_file, output_file)
        print(f"已转换: {filename}")
    
    print(f"所有文件已转换完成，输出目录: {output_dir}")


def main():
    """
    主函数
    """
    input_dir = "final_data/poses"
    output_dir = "final_data/poses_converted"
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    print(f"开始转换姿态文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    convert_all_poses(input_dir, output_dir)


if __name__ == "__main__":
    main()