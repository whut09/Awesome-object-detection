---
title: "Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention"
description: "解析 ASAB 桥接脉冲与人工神经网络、兼顾事件时空建模和 Loihi 2 部署的混合检测器。"
tags: ["CVPR 2025", "目标检测", "事件相机", "SNN-ANN", "ASAB"]
---

# Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention

**会议**：CVPR 2025  
**论文**：[arXiv](https://arxiv.org/abs/2403.10173)  
代码：未发现论文声明的官方代码。

## 一句话总结

论文用 Attention-based SNN-ANN Bridge（ASAB）把稀疏脉冲中的短时运动与空间结构压成稠密特征，再交给 ANN 与 YOLOX 检测头，并让前四层可在 Loihi 2 上低功耗运行。

## 研究背景与问题

事件相机输出 `(x,y,t,p)` 异步事件，Gen1 输入稀疏度约 98%，但纯 SNN 难以提取高层语义，纯 ANN 又会把稀疏事件转成密集计算。已有混合网络常用简单求和折叠时间维，容易丢失不规则脉冲的局部结构；纯 RNN/Transformer 虽精度高，却增加参数、MAC 与高频推理成本。本文要解决的是如何真正连接 SNN 的硬件效率与 ANN 的表征能力，而非只做网络拼接。

## 方法总览

原始事件按 5 ms 划分为 `T×2×H×W` 张量，正负极性各占一通道；四个由卷积、BatchNorm 和 PLIF 神经元组成的 SNN block 生成 `Espike`。ASAB 依次执行 Spatial-aware Temporal（SAT）attention 与 Event-Rate Spatial（ERS）attention，将 `T×C×H'×W'` 脉冲变为 `C×H'×W'` 稠密特征。四个 ANN block 提取高层语义，最终进入 YOLOX 的 FPN 与检测头；Hybrid+RNN 变体还在 ASAB 后插入两个 DWConvLSTM。

## 方法详解

SAT 先做 Channel-wise Temporal Grouping，把 `Espike` 转为 `C×T×H'×W'`。Time-wise Separable Deformable Convolution（TSDC）为每个时间步设置独立 deformable-convolution group，以学习稀疏、形变的事件结构；Temporal Attention 再由局部空间上下文计算时间关系，将短时脉冲累积为空间表示。可变形采样负责找事件边缘，时间注意力决定保留哪些时刻。

ERS 将 SAT 输出沿时间维求和得到事件率图，经 sigmoid 形成空间权重，再与 SAT 特征逐元素相乘。高事件活动区域被增强，孤立脉冲被压低。ASAB 输出与标准二维卷积兼容，后半段 ANN 无需感知脉冲状态；DWConvLSTM 则捕获比 5 ms SNN 步长更慢的动态，形成“短时 SNN—ASAB—慢时 RNN/高层 ANN—YOLOX”数据流。

## 实验与证据

实验使用 Gen1 Automotive Detection（304×240、39 小时、car/pedestrian）和 Gen4（1280×720、15 小时，增加 two-wheeler）。基础模型在 Gen1/Gen4 的 COCO-style mAP 分别为 0.35/0.27，仅 6.6M 参数；Gen1 上高于 Events-RetinaNet 的 0.34、EMS-RES34 的 0.31。Hybrid+RNN 为 7.7M 参数、Gen1 mAP 0.43，低于 RVT-B 的 0.47，但更强调因果高频推理和边缘部署。

关键消融直接验证 ASAB：无 Temporal Attention、把 deformable convolution 换成普通卷积、移除 ERS 时 mAP 分别为 0.33、0.34、0.34；用简单累积替代 ASAB 仅 0.30，完整模型为 0.35，mAP50 从 0.53 升至 0.61。DWConvLSTM 从全放置的 0.42 提升到只保留 L6/L8 的 0.43；SNN/ANN block 采用 4/4，在 mAP 0.35 与计算量间折中。Loihi 2 上四层 SNN 以 int8、6 芯片运行约 1.73 W、2.06 ms/step，低于 5 ms 事件步长；int4 可到 1.16 ms，但 int2 会使 mAP 从约 0.348 跌至 0.224。

## 对 YOLO-Agent 的启发

可把 ASAB 当作事件前端适配器，而不必重写 YOLO neck/head。**Harness** 比较“事件帧直接送入 YOLO”“SNN 后时间求和”“SNN+完整 ASAB”三组，固定事件窗口、增强和检测头；记录 Gen1 mAP、mAP50、每步 p50/p95 延迟、MAC/AC、事件稀疏率及量化精度。完整 ASAB 相对简单求和至少 `+0.04 mAP`、p95 小于 5 ms，int8 相对浮点下降不超过 0.005 才通过。若增益只来自更大 ANN 后端、硬件步长超时，或小目标误检率不降，则判定失败。

## 优点

- SAT、TSDC、ERS 分别对应时间关系、稀疏空间采样和事件率选择，职责清晰。
- 同时报告 ANN、SNN、RNN 基线，以及真实 Loihi 2 功耗和时延。
- 前端稀疏、后端稠密的边界明确，便于替换检测头。

## 局限

- 基础模型精度仍低于强 RNN/SSM 检测器，Hybrid+RNN 训练约需 6 天。
- 硬件测试只覆盖前四层，未测完整端到端系统功耗。
- Gen1/Gen4 类别和道路场景有限，跨相机与极端事件率证据不足。

## 评分

- **创新性**：★★★★☆
- **实验充分度**：★★★★☆
- **部署价值**：★★★★★
- **YOLO-Agent 参考价值**：4.2/5
