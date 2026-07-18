---
title: "[论文解读] Training Strategies for Object Detection with Vision Transformers"
description: "这篇工作面向车载多视角 3D 检测的真实延迟，而不是再设计一个 Transformer：作者依次压缩输入、裁掉无目标区域、调整模型规模，并比较 PyTorch/TensorRT 的 FP32、FP16 部署。"
tags: ["CVPRW 2023", "目标检测"]
---

# Training Strategies for Object Detection with Vision Transformers

**论文**：[官方论文](https://arxiv.org/abs/2304.02186)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

这篇工作面向车载多视角 3D 检测的真实延迟，而不是再设计一个 Transformer：作者依次压缩输入、裁掉无目标区域、调整模型规模，并比较 PyTorch/TensorRT 的 FP32、FP16 部署。

## 研究背景与问题

多视角 Transformer 能利用长程关系预测 BEV 三维框，但车载算力要求固定周期内完成推理。论文强调 MACs、参数量和实际延迟并不等价，因此以 mAP、max-F1 与端到端运行时间共同选择配置。

## 方法总览

基线由多相机图像、相机变换矩阵、CNN/FPN 骨干和 Transformer 检测头组成。优化顺序是先测各阶段耗时，再缩小输入分辨率、删除画面顶部几乎不含真值框的区域、选择更合适的模型宽深度，最后转 TensorRT 并启用半精度。

## 方法详解

### 1. 在同一 GPU 上分别测整网、Backbone+FPN 和检测头，避免用理论 MACs 代替部署时延。

### 2. 把输入从 996×656 降到 598×394；该尺寸仍覆盖主要目标，同时显著减少特征图与注意力计算。

### 3. 统计训练集中目标垂直位置，发现缩放后顶部 50 像素包含的真值不足 0.25%，因此直接裁去该区域。

### 4. 对 PyTorch/TensorRT、FP32/FP16 做交叉测试；TensorRT 通过算子融合进一步减少 kernel 调用。

## 实验与证据

分辨率降至 598×394 后，推理时间改善 52.5%，行人 AP 下降 1.4%，车辆 AP 下降 0.5%。仅把 PyTorch 模型改为 FP16，作者观察到约 50% 的速度提升。所有策略叠加后，运行时间最多改善 63%，代价是约 3% 的检测性能下降；论文还比较 mAP 与 max-F1，提醒固定阈值部署不能只看 mAP。

## 对 YOLO-Agent 的启发

YOLO-Agent 变量设为 input_hw、top_crop、engine∈{pytorch,tensorrt}、precision∈{fp32,fp16}、query_count。每个变量单独切换并记录车辆/行人 AP、max-F1、P50/P95 延迟。若 598×394 在本地导致小目标召回骤降，或 TensorRT 的 P95 延迟未随平均延迟下降，则不采用该策略。

## 优点

直接测工业常用 TensorRT，并把速度收益与类别 AP 损失绑定。

## 局限

数据来自内部自动驾驶集，顶部裁剪和目标分布高度依赖相机安装位置。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

