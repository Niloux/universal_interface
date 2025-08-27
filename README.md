# Universal Interface

## 项目简介

`Universal Interface` 是一个为三维高斯溅射（3D Gaussian Splatting, 3DGS）训练设计的数据预处理工具。它旨在提供一个通用的数据接口，能够将用户自定义的、符合特定格式的驾驶场景或机器人数据集，高效地转换为3DGS模型训练所需的标准数据格式。

项目通过自动化的数据处理流水线，处理包括车辆自车姿态（Ego-Pose）、多摄像头内外参、动态与静态物体的3D轨迹、以及为动态物体生成2D分割掩码等任务，极大地简化了3DGS训练数据的准备流程。

## 主要特性

- **自动化处理流水线**: 从原始数据到训练所需格式的全自动处理。
- **模块化设计**: 清晰的模块划分，易于扩展和维护。
- **可配置性**: 通过 `config.yaml` 文件灵活配置输入/输出路径、相机参数等。
- **动态物体支持**: 能够识别动态物体，并为其生成训练时所需的2D掩码。
- **详细的日志系统**: 提供带颜色区分的日志输出，方便追踪处理过程。

## 项目架构

```
universal_interface/
├── .gitignore           # Git忽略文件
├── config.yaml          # 核心配置文件
├── main.py              # 主程序入口
├── pkl2json.py          # (可选) Pickle转JSON工具
├── pyproject.toml       # 项目依赖与配置 (uv)
├── README.md            # 项目说明文档
├── core/                # 核心处理模块
│   ├── base.py          # 处理器基类
│   ├── camera.py        # 相机内外参及图像处理器
│   ├── dynamic_mask.py  # 动态物体掩码生成器
│   ├── ego_pose.py      # 自车位姿处理器
│   └── track.py         # 3D物体轨迹处理器
├── input/               # 原始数据输入目录 (需按约定格式存放)
│   ├── ego_pose/
│   ├── images/
│   ├── images_metadata/
│   └── objects/
├── output/              # 处理结果输出目录
│   ├── dynamic_mask/
│   ├── ego_pose/
│   ├── extrinsics/
│   ├── images/
│   ├── intrinsics/
│   └── track/
├── utils/               # 工具模块
│   ├── config.py        # 配置加载与管理
│   ├── data_io.py       # 数据读写工具
│   ├── geometry.py      # 几何变换工具
│   ├── logger.py        # 日志工具
│   └── structures.py    # 自定义数据结构
└── test/                # 测试代码目录
```

## 数据处理流水线

程序通过 `main.py` 启动，并按以下顺序执行数据处理：

1.  **EgoPoseProcessor**: 处理自车位姿数据。
2.  **CameraProcessor**: 处理相机内外参和图像数据，将其转换为标准格式。
3.  **TrackProcessor**: 分析3D物体数据，计算轨迹，并区分动态与静态物体。
4.  **DynamicMaskProcessor**: 基于轨迹信息，为动态物体生成2D图像掩码。

## 核心组件

### 处理器 (Processors)

- **`EgoPoseProcessor`**: 读取、转换并输出自车位姿（Ego-Pose）数据。
- **`CameraProcessor`**:
    - 读取多相机元数据，生成独立的内外参文件。
    - 复制并重命名图像文件，以符合 `frame_id_camera_id.png` 的格式。
- **`TrackProcessor`**:
    - 从原始物体数据中构建每个物体的3D轨迹。
    - 判断物体是动态还是静态。
    - 保存轨迹信息、可见性等数据为 Pickle 和 JSON 文件。
- **`DynamicMaskProcessor`**:
    - 加载轨迹数据和相机参数。
    - 将动态物体的3D边界框投影到每个相机的2D图像平面上。
    - 生成并保存对应的2D分割掩码图像。

### 工具模块 (Utils)

- **`config.py`**: 加载并验证 `config.yaml`，提供全局配置访问。
- **`data_io.py`**: 封装了所有文件读写操作，如加载/保存 Pickle, JSON, 图像和相机参数。
- **`geometry.py`**: 提供核心的几何计算功能，如坐标系变换、3D点到2D图像的投影等。
- **`logger.py`**: 提供带颜色和时间戳的日志记录功能。
- **`structures.py`**: 定义了项目中使用的核心数据结构，如 `Box3D`, `TrajectoryData` 等。

## 使用方法

### 1. 环境配置

本项目使用 `uv` 作为包管理工具。

```bash
# 安装uv
pip install uv

# 创建虚拟环境并安装依赖
uv venv
uv sync
```

### 2. 准备数据

将你的自定义数据集按照 `input/` 目录下的结构进行组织。确保 `images`, `images_metadata`, `objects`, `ego_pose` 等子目录中包含所需的数据文件。

### 3. 修改配置

打开 `config.yaml` 文件，根据你的需求进行配置：

- **`input`**: 设置为你的原始数据输入目录。
- **`output`**: 设置为处理结果的输出目录。
- **`camera`**:
    - `positions`: 定义所有相机位置的名称列表。
    - `id_map`: 定义相机位置名称到相机ID（整数）的映射关系。

### 4. 运行程序

完成配置后，在项目根目录运行主程序：

```bash
uv run python main.py
```

处理完成后，所有生成的数据将保存在你指定的 `output` 目录中。

## 扩展开发

如果你需要添加新的数据处理器（例如，处理激光雷达点云），可以遵循以下步骤：

1.  在 `core/` 目录下创建一个新的Python文件（例如 `lidar_processor.py`）。
2.  在文件中创建一个继承自 `BaseProcessor` 的新处理器类。
3.  实现 `process()` 方法，在其中定义你的数据处理逻辑。
4.  在 `main.py` 的 `processors` 列表中，将你的新处理器按正确的顺序添加进去。

## 贡献

欢迎对本项目进行贡献！如果你发现了Bug或有任何改进建议，请随时提交 Issues 或 Pull Requests。
