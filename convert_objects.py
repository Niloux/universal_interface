#!/usr/bin/env python3
"""
对象标签转换脚本
将input/objects目录下的所有JSON文件合并转换为final_data/labels/000000.txt文件
"""

import os
import json
import glob
from pathlib import Path


def convert_all_objects_to_single_file(input_dir: str, output_file: str) -> None:
    """
    转换所有对象文件到单个CSV文件
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出文件路径
    """
    # 创建输出目录
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    json_files.sort()  # 确保按文件名排序
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 准备所有CSV行数据
    all_csv_lines = []
    
    for input_file in json_files:
        # 提取文件名（不含扩展名）作为frame_id
        filename = os.path.basename(input_file)
        frame_id = os.path.splitext(filename)[0]
        
        print(f"正在处理: {filename}")
        
        with open(input_file, 'r') as f:
            objects_data = json.load(f)
        
        for obj in objects_data:
            # 提取所需字段
            obj_id = obj.get('track_id', '')
            obj_type = obj.get('label', '').replace('type_', '')  # 移除type_前缀
            
            # 3D边界框中心坐标
            center = obj.get('box3d_center', [0, 0, 0])
            x, y, z = center[0], center[1], center[2]
            
            # 3D边界框尺寸
            size = obj.get('box3d_size', [0, 0, 0])
            l, w, h = size[0], size[1], size[2]  # length, width, height
            
            # 朝向角
            heading = obj.get('box3d_heading', 0)
            
            # 构建CSV行
            csv_line = f"{frame_id}, {obj_id}, {obj_type}, {x}, {y}, {z}, {l}, {w}, {h}, {heading}"
            all_csv_lines.append(csv_line)
    
    # 写入单个CSV文件（不包含表头）
    with open(output_file, 'w') as f:
        for line in all_csv_lines:
            f.write(line + '\n')
    
    print(f"所有对象数据已合并到: {output_file}")
    print(f"总共写入 {len(all_csv_lines)} 行数据")


def main():
    """
    主函数
    """
    input_dir = "input/objects"
    output_file = "final_data/labels/000000.txt"
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    print(f"开始转换对象标签文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_file}")
    
    convert_all_objects_to_single_file(input_dir, output_file)


if __name__ == "__main__":
    main()