# Universal Interface

## 项目简介

3DGS训练通用数据接口，将自定义数据集转换成3DGS训练数据格式。

## 项目架构

```
universal_interface/
├── config.yaml          # 配置文件
├── main.py              # 程序入口
├── core/                # 核心处理模块
│   ├── __init__.py      # 模块导出
│   ├── base.py          # 基础处理器抽象类
│   └── ego_pose.py      # EgoPose数据处理器
├── utils/               # 工具模块
│   ├── __init__.py      # 模块导出
│   └── config.py        # 配置管理类
└── test/                # 输出目录
```

## 核心组件

### 1. BaseProcessor (基础处理器)
- 抽象基类，定义了数据处理器的通用接口
- 提供目录操作的通用方法
- 所有具体处理器都继承此类

### 2. EgoPoseProcessor (EgoPose处理器)
- 继承BaseProcessor
- 专门处理ego_pose类型的数据
- 实现数据读取、转换和输出功能

### 3. Config (配置管理)
- 负责加载和管理YAML配置文件
- 提供配置项访问接口
- 支持配置项启用状态检查

## 使用方法

1. 配置config.yaml文件
2. 将数据放入input目录对应子目录
3. 运行程序：`uv run main.py`
4. 处理结果输出到test目录

## 配置说明

- `input`: 输入数据目录路径
- `output`: 输出数据目录路径
- `process`: 处理器启用配置
- `camera`: 相机配置列表

## 扩展开发

添加新的数据处理器：
1. 在core目录创建新的处理器类
2. 继承BaseProcessor基类
3. 实现process()抽象方法
4. 在core/__init__.py中导出新类
5. 在main.py中添加调用逻辑