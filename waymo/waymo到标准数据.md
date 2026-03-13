# 场景（Scene）数据集结构文档

该数据集包含了一个完整场景所需的多种传感器数据、标注信息及坐标变换参数。

## 文件夹及内容结构

| 文件夹名称 | 文件名格式 | 内容详情 |
| --- | --- | --- |
| **Images**<br>

<br>图像 | `png`格式，以帧数(6位数字)+相机号命名<br>

<br>e.g. `000000_0.png` | 默认0号相机为正前方相机。 |
| **Pointclouds**<br>

<br>点云 | `ply`格式，以帧数(6位数字)+雷达号命名<br>

<br>e.g. `000000_0.ply` | 包含点云 x, y, z, intensity 等信息。<br>

<br>采用雷达坐标系。默认0号激光雷达为360°雷达（即三维重建-点云重建的雷达）。 |
| **Labels**<br>

<br>物体标签 | `txt`格式，以6位数字命名<br>

<br>e.g. `000000.txt` | 每一行包含：`frame_id`, `obj_id`, `type`, `x`, `y`, `z`, `l`, `w`, `h`, `heading`<br>

<br>(帧id, 物体id, 物体类别, 物体中心x, y, z, 物体长, 宽, 高, 航向角)。<br>

<br>**车辆坐标系下：** x向前, y向左, z向上。 |
| **Intrinsics_Camera**<br>

<br>相机内参 | `txt`格式，以相机号命名<br>

<br>e.g. `0.txt` | 一行数字以空格间隔，分辨率+3x3内参矩阵平铺。<br>

<br>e.g. `width height fx 0 cx 0 fy cy 0 0 1` |
| **Intrinsics_Lidar**<br>

<br>雷达内参 | `txt`格式，以激光雷达号命名<br>

<br>e.g. `0.txt` | 一行数字以空格间隔，包含：<br>

<br>扫描范围_最小距离, 扫描范围_最大距离, 水平视角范围, 垂直视角范围_上, 垂直视角范围_下, 分辨率_线数, 分辨率_水平点数。 |
| **Extrinsics_Camera**<br>

<br>相机外参 | `txt`格式，以相机号命名<br>

<br>e.g. `0.txt` | **相机坐标系转车辆坐标系：**<br>

<br>一行数字以空格间隔，3x4外参矩阵平铺。<br>

<br>其中 `ext = [R, T]`, $R \in \mathbb{R}^{3 \times 3}$, $T \in \mathbb{R}^{3 \times 1}$。 |
| **Extrinsics_Lidar**<br>

<br>雷达外参 | `txt`格式，以激光雷达号命名<br>

<br>e.g. `0.txt` | **雷达坐标系转车辆坐标系：**<br>

<br>一行数字以空格间隔，3x4外参矩阵平铺。<br>

<br>其中 `ext = [R, T]`, $R \in \mathbb{R}^{3 \times 3}$, $T \in \mathbb{R}^{3 \times 1}$。 |
| **Poses**<br>

<br>车辆位姿 | `txt`格式，以帧数(6位数字)命名<br>

<br>e.g. `000000.txt` | **车辆坐标系转世界坐标系：**<br>

<br>一行数字以空格间隔，3x4 pose矩阵平铺。<br>

<br>其中 `pose = [R, T]`, $R \in \mathbb{R}^{3 \times 3}$, $T \in \mathbb{R}^{3 \times 1}$。 |
| **Others**<br>

<br>其余 | `.xodr`<br>

<br>`.xosc` | 包含高精地图（OpenDRIVE）及场景描述（OpenSCENARIO）等相关文件。 |

---

## 坐标系说明

1. **车辆坐标系定义：**
* **X轴：** 向前
* **Y轴：** 向左
* **Z轴：** 向上


2. **坐标转换关系：**
* **外参（Extrinsics）：** 用于将传感器（相机/雷达）坐标系下的点转换到车辆坐标系。
* **位姿（Poses）：** 用于将车辆坐标系下的点转换到世界坐标系。