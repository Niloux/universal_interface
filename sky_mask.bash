#!/bin/bash

# 1.激活transfer的虚拟环境
# source /home/saimo/wuyou/transfer/.conda/bin/activate

# 2. 读取路径
read -p "请输入输入数据的目录路径: " input_dir
read -p "请输入输出数据的目录路径: " output_dir

# 检查输入是否为空
if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
    echo "错误：输入或输出路径不能为空"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$input_dir" ]; then
    echo "错误：输入目录不存在: $input_dir"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 3. 调用python脚本
cd /home/saimo/wuyou/transfer
/home/saimo/wuyou/transfer/.conda/bin/python /home/saimo/wuyou/transfer/simple_sky_mask.py "$input_dir" "$output_dir"