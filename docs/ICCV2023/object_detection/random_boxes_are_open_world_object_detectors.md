---
title: "[论文解读] Random Boxes Are Open-world Object Detectors"
description: "RandBox 用每次迭代重新采样的随机区域训练 Fast R-CNN，使 proposal 不被已知类别分布绑架，从而提高未知物体召回。"
tags: ["ICCV 2023", "目标检测"]
---

# Random Boxes Are Open-world Object Detectors

**论文**：[官方论文](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Random_Boxes_Are_Open-world_Object_Detectors_ICCV_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

RandBox 用每次迭代重新采样的随机区域训练 Fast R-CNN，使 proposal 不被已知类别分布绑架，从而提高未知物体召回。

## 研究背景与问题

常规 RPN 只从已知类标注学习 objectness，开放世界测试时容易忽略未标类别。随机 proposal 与已知物体分布独立，可作为工具变量打破这种偏置。

## 方法总览

训练时直接生成随机框并提取 RoI 特征，不使用学习型 RPN。作者设计 matching score：随机框若未与已知真值匹配，不会简单按预测类别分数受到惩罚；已知类分类与类别无关 objectness 共同学习。

## 方法详解

### 1. 按位置、尺度和长宽比随机采样区域，覆盖已知与潜在未知位置。

### 2. Fast R-CNN RoI head 对随机框编码，匹配已知真值的框学习分类与回归。

### 3. 对未匹配框使用无偏 matching score，避免把可能的未知对象全部压成背景。

### 4. 测试时按 objectness 保留未知候选，并进入增量开放世界评价。

## 实验与证据

论文在 Pascal-VOC/MS-COCO 与 LVIS 两套开放世界基准上比较 Faster R-CNN、Transformer OWOD 和既有开放世界方法；RandBox 在已知类精度、未知召回、Wilderness Impact 与 A-OSE 上整体领先。消融表明，固定 proposal 或恢复普通负样本惩罚都会削弱未知召回，说明随机化与 matching score 缺一不可。

## 对 YOLO-Agent 的启发

设置 random_boxes_per_image、scale_range、matching_score、unknown_threshold；同时记录 known mAP、unknown recall、WI、A-OSE。若随机框增加后召回不升而显存线性增长，或未知召回来自大量已知类误报，则不采用。

## 优点

不需要额外未知类标注，且与 Fast R-CNN 训练接口兼容。

## 局限

随机框数量大时 RoI 计算昂贵；开放世界 objectness 仍受 backbone 预训练语义影响。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

