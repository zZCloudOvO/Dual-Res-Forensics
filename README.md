# Dual-Res Forensics (DRF): 边界感知与双通道残差深度伪造检测

![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg?style=flat-square&logo=pytorch)
![DeepfakeBench](https://img.shields.io/badge/Based%20on-DeepfakeBench-blue.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Midterm%20Development-orange.svg?style=flat-square)

## 📌 项目背景与创新点

本项目致力于解决通用人脸伪造检测（Generalizable Face Forgery Detection）任务中的泛化性难题。针对传统高通滤波方法在细节提取上的局限性，以及现有模型容易过拟合特定操作痕迹的问题，本项目提出了 **Dual-Res Forensics (DRF)** 框架。

核心创新点严格遵循“渐进式迭代”思路，包含：
1. **双通道特征融合架构**：采用强大的视觉基础模型（CLIP）作为冻结的主通道提取全局语义，同时并联一个轻量级的残差通道（Adapter）强化局部高频细节的捕捉。
2. **残差引导的交叉注意力机制**：以残差信号作为 Query，引导整图 CLIP 特征（Key/Value），实现两路信号的高效融合，显著提升对隐写痕迹的检测敏感度。
3. **聚焦“过渡带”的属性差异学习**：在训练阶段引入 A/B 图像集对比思路（全局一致属性 vs. 局部不一致属性），迫使模型忽略压缩、滤波等常规操作痕迹，精准锁定图像融合边缘的“属性差异过渡带”。

## 📂 模块化项目结构

本项目底层架构参考了目前最权威的 DeepfakeBench 框架，实现了高度模块化和可扩展性：

- `data_processing/`：统一的数据预处理流水线。目前已实现基于 OpenCV 的基础人脸检测与对齐模块（统一输出 256x256 规格）。
- `training/`：模型训练与核心算法库。包含 `model.py`（双通道残差网络结构实现）。
- `evaluation/`：模型验证与指标计算（待完善，计划引入 AUC, AP, EER 等核心指标计算体系）。
- `config/`：YAML 格式的全局配置文件存放区。
- `main.py`：DRF 训练框架的主引擎入口。

## 🚀 快速启动测试

本项目已完成基础网络结构的搭建与跑通验证。你可以通过以下命令快速测试模型的前向传播与简易训练循环：

```bash
# 1. 安装基础依赖
pip install torch torchvision transformers opencv-python numpy

# 2. 运行主训练引擎 (包含模拟数据验证)
python main.py
