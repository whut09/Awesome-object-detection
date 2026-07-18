---
title: "[论文解读] AKCMamba-YOLO: Selective State Space Models For Real-Time Object Detection"
description: "将自适应核卷积与选择性状态空间模型嵌入 YOLO，在近线性复杂度下补充全局依赖和跨尺度语义。"
tags: ["CVPR 2026", "目标检测", "YOLO", "Mamba", "状态空间模型"]
---

# AKCMamba-YOLO: Selective State Space Models For Real-Time Object Detection

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_AKCMamba-YOLO_Selective_State_Space_Models_For_Real-Time_Object_Detection_CVPR_2026_paper.html)  
**代码**: [xlllchen/AKCMamba_YOLO](https://github.com/xlllchen/AKCMamba_YOLO)  
**任务**: 实时目标检测

## 一句话总结

论文以 YOLOv8 为骨架，用 3CAKCMamba 和 4CAKCMamba 替换主干、颈部中的 C2f：自适应核卷积负责不规则局部结构，二维选择性扫描负责长程依赖，通道重标定负责抑制冗余，从而在不采用二次复杂度自注意力的情况下获得全局上下文。

## 背景与问题

纯卷积 YOLO 的优势是算子成熟和延迟低，但长距离关系只能依靠堆层间接建立。Transformer 能直接做全局交互，却在高分辨率检测特征上产生较高计算和访存成本。作者选择状态空间模型作为中间路线：保留随序列长度近线性增长的计算，同时获得输入相关的全局建模。

## 方法总览

模型保持 YOLOv8 的 Backbone-Neck-Head 拓扑，在主干用 3CAKCMamba 替换 C2f，在颈部用 4CAKCMamba 加强跨尺度融合。两个模块都按“自适应局部采样—二维状态空间扫描—通道重标定”的顺序处理特征，只是并行局部分支数量和放置阶段不同。

## 方法详解

```mermaid
flowchart LR
    X[输入特征] --> A[AKConv 自适应采样]
    A --> S[AKSS2D 四向选择性扫描]
    S --> C[AKC Attention 通道重标定]
    C --> M[3CAKCMamba / 4CAKCMamba]
    M --> Y[多尺度检测特征]
```

### 自适应局部建模

AKConv 学习卷积采样位置，而不是始终使用规则方形网格，使局部分支能适配细长、倾斜或被遮挡目标。3CAKC/4CAKC 使用不同数量的自适应核路径形成多尺度局部表征。

### AKSS2D

二维特征按多个方向展开为序列，选择性状态空间单元根据输入决定信息保留与传播。四向扫描降低单一扫描方向偏置，使空间上距离较远的区域可以通过状态递推交互，复杂度相较全局注意力更适合实时检测。

### 主干与颈部的不同配置

主干采用 3CAKCMamba 控制计算量，颈部采用 4CAKCMamba加强跨层、跨尺度语义融合。论文并未整体推翻 YOLO 拓扑，而是围绕 C2f 替换，因此适合做局部模块实验。

## 实验与证据

- 在 COCO2017、输电塔异物数据和作者发布的 2,975 张铁路行人数据上评估。
- 论文报告代表性配置以 14.9G FLOPs 达到 46.3 mAP，相比 YOLOv8-S 提高 1.4 AP，同时减少约 47.9% FLOPs。
- 消融分别验证自适应核、状态空间扫描、重标定模块及主干/颈部放置方式。
- 多数据集结果用于验证复杂背景、遮挡与安全场景，而非只在 COCO 上比较。

## 对 YOLO-Agent 的启发

- 先只替换一个深层 C2f，随后再测试 Neck 放置，避免同时改动导致归因困难。
- 对比标准卷积、可变形卷积、窗口注意力和 AKSS2D，并保持参数量接近。
- Harness 需记录输入分辨率变化下的延迟、显存和小目标 AP，验证线性复杂度优势。
- 额外统计扫描方向、状态维度和自适应采样偏移的稳定性。

## 优点

- 在不引入全局二次注意力的情况下补充长程依赖。
- 改造集中于 C2f 替换，适合做局部、可归因的 YOLO 实验。
- 同时在通用、工业异物和铁路行人场景评估复杂背景能力。

## 局限

- 状态空间扫描和自适应采样的导出支持弱于标准卷积。
- 多模块串联后，收益来源和硬件实际延迟仍依赖实现。
- 自建铁路数据规模较小，跨行业泛化需要独立复现。

## 评分

- **创新性**: ★★★★☆
- **部署价值**: ★★★☆☆
- **YOLO-Agent 参考价值**: ★★★★☆
