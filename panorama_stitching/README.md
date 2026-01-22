# Panorama Image Stitching

## 项目简介
本项目实现了全景图像拼接（Panorama Image Stitching）功能，能够将多张具有重叠区域的图片拼接成一张全景图。该程序使用Python和OpenCV库，通过特征点检测、特征匹配、单应性矩阵计算、图像变换和融合等步骤完成拼接。

## 功能特点
- 支持多张图片的拼接
- 提供两种运行模式：完整实现和快速测试（使用OpenCV内置函数）
- 可生成测试图片用于演示
- 代码结构清晰，注释详细，便于理解和修改

## 环境要求
- Python 3.8+
- OpenCV 4.8.1
- NumPy 1.24.3
- Matplotlib 3.7.2

## 安装依赖
在项目目录下运行以下命令安装依赖：
```bash
pip install -r requirements.txt