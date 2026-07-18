---
title: "[论文解读] ObjectBox: From Centers to Boxes for Anchor-Free Object Detection"
description: "ObjectBox 让每个目标中心在所有 FPN 层都成为正样本，并以中心 cell 两角到框边距离回归。"
tags: ["ECCV 2022", "目标检测", "ObjectBox", "Anchor-Free", "标签分配"]
---

# ObjectBox: From Centers to Boxes for Anchor-Free Object Detection

**论文**: [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6191_ECCV_2022_paper.php)  
**代码**: [MohsenZand/ObjectBox](https://github.com/MohsenZand/ObjectBox)  
**任务**: 无锚单阶段目标检测

## 一句话总结

ObjectBox 把目标中心 cell 视为形状和尺度无关的 anchor，在每个 FPN level 都监督同一对象，不再用 anchor IoU、尺寸区间或 center sampling 排除尺度，并从中心 cell 的两个角回归到真实框四边。

## 研究背景与问题

YOLO 的 anchor 匹配偏向预设形状，FCOS 用 regression range 把对象限定到单个层级；这些规则会让边界尺寸目标缺少正样本，也引入数据集相关阈值。若只从 cell 中心回归四边，极小对象的数值和 IoU 梯度又容易不稳定。论文希望用一个正位置定义覆盖任意形状，并让所有尺度层都学习每个对象。

## 方法总览

对 GT 中心落入的网格 cell，ObjectBox 在所有 feature levels 标为正样本。回归目标不是从单点到四边，而是从该 cell 左上角预测左/上距离、右下角预测右/下距离，组合得到 box。分类和 objectness 只在中心位置监督。Scale-aware IoU loss 根据目标尺度调整重叠优化，使小框误差不会被大框主导。

## 方法详解

全尺度正样本意味着一个对象会产生多层候选，训练时每层都获得监督，推理仍由 NMS选择最佳尺度。中心 cell 两角提供了非零空间基准，缓解同一点同时预测相反方向距离的不适定性。整个 assignment 不需要 anchor 数、aspect ratio、FCOS range 或数据集特定阈值。

## 实验与证据

论文在 COCO 2017 与 PASCAL VOC 2012 上和 YOLO、FCOS、CenterNet 等比较，ObjectBox 在无需数据集调参的前提下保持有竞争力 AP。消融逐项测试单层/全层监督、中心点/中心 cell 两角回归、普通 IoU 与 tailored IoU loss；全层学习提高召回，新的回归参数化和尺度损失主要改善 AP75 与尺度变化对象。

## 对 YOLO-Agent 的启发

应把 YOLO anchor assignment、FCOS range、ObjectBox 单层和全尺度四组放在同一 backbone/neck 下。记录 APs/APm/APl、每 GT 正样本层数、跨层重复框数、NMS耗时和梯度尺度。若全层监督只提高召回却让重复候选与 NMS 延迟显著增长，或小目标 AP75 下降，说明跨尺度正样本过多；迁移新数据集时应验证无需重新聚类和调整 range 才是核心收益。

## 优点

- 去除 anchor 与尺度区间超参数。
- 每个对象在所有特征层都有学习机会。
- 回归定义简单，可移植到 YOLO 风格检测头。

## 局限

- 全尺度预测增加重复框。
- 单中心 cell 对极大目标的监督仍较稀疏。
- 推理阶段依然依赖 NMS。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **工程价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★★
