---
title: "[论文解读] DynamicDet: A Unified Dynamic Architecture for Object Detection"
description: "DynamicDet 用 Adaptive Router 判断图像难度：容易图像走一个检测器，困难图像再经过第二个 backbone 与检测头，实现同一模型内可调的精度—速度曲线。"
tags: ["CVPR 2023", "目标检测"]
---

# DynamicDet: A Unified Dynamic Architecture for Object Detection

**论文**：[官方论文](https://arxiv.org/abs/2304.05552)  
**代码**：[官方实现](https://github.com/VDIGPKU/DynamicDet)

## 一句话总结

DynamicDet 用 Adaptive Router 判断图像难度：容易图像走一个检测器，困难图像再经过第二个 backbone 与检测头，实现同一模型内可调的精度—速度曲线。

## 研究背景与问题

分类动态网络常在中间层提前退出，但检测需要多尺度特征，单一分类置信度不足以代表图像难度。DynamicDet 因此把路由放在第一套多尺度特征之后，并用检测损失定义退出监督。

## 方法总览

B1 提取 F1，路由器对每个尺度做全局池化并融合，输出难度 q。easy 样本由 D1(F1) 直接预测；hard 样本进入 B2，B2 同时读取原图和来自 B1 的跨阶段特征，最终由 D2(F2) 输出。阈值可在部署时改变。

## 方法详解

### 1. 先联合训练两条检测路径，保证 D1、D2 都具备完整检测能力。

### 2. 根据两条路径的检测损失差构造路由目标，而不是用图像分类难度代替。

### 3. 训练轻量路由器预测 q，并加入计算预算约束，避免所有样本都选重路径。

### 4. 推理时调整阈值，得到不同 easy ratio、FPS 与 AP。

## 实验与证据

COCO 上 Dy-YOLOv7-W6/50 达到 56.1 AP、58 FPS，比 YOLOv7-E6 快 12%；W6/100 达到 56.8 AP、46 FPS，在相近精度下比 YOLOv7-E6E 快 39%。双阶段版本 Dy-Faster R-CNN ResNet50/90 得到 40.4 box AP，高于普通 ResNet101 的 39.4。论文还分别消融路由器开销、退出比例和不同阈值。

## 对 YOLO-Agent 的启发

记录 router_threshold、easy_ratio、D1_AP、D2_AP、单图 FLOPs 和 P95 延迟。首先固定 YOLOv7/YOLO-Agent 权重，只替换动态路径；再画 AP-FPS 曲线。若路由器把几乎所有图送往同一分支，或平均 FPS 上升但 P95 延迟恶化，则路由策略失败。

## 优点

可用一个权重覆盖多个部署预算，并同时适配实时与双阶段检测器。

## 局限

包含两套骨干/头，模型存储和训练成本高于静态检测器；批量推理的分支发散也可能降低 GPU 利用率。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

