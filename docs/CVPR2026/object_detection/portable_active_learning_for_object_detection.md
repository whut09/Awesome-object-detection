---
title: "Portable Active Learning for Object Detection"
description: "解析 PAL 如何只依赖检测推理输出，以 LIUS 实例不确定性和 GUIDE 图像多样性实现跨检测器主动学习。"
tags: ["CVPR 2026", "主动学习", "PAL", "LIUS", "GUIDE"]
---

# Portable Active Learning for Object Detection

**论文**: [arXiv](https://arxiv.org/abs/2605.10349)  
**官方代码**: 未发现论文声明的官方代码  
**任务**: 目标检测主动学习

## 一句话总结

Portable Active Learning（PAL）用每类 logistic classifier 根据“检测置信度+该框关联的 pre-NMS 候选数”判断 TP/FP 边界并计算 LIUS 熵，再用 CWIE、RCDI、RCSP 组成 GUIDE 筛掉信息单一或视觉重复的图像，全程不读取检测器中间特征或修改训练代码。

## 研究背景与问题

检测主动学习需要在有限框标注预算下挑图。许多方法依赖梯度、RoI 特征、额外 loss 或特定训练日程，迁移到 RetinaNet、Faster R-CNN、SSD、YOLOX/YOLO11 时要重写内部接口；只做图像熵又会被高频类别主导，只做实例不确定性则可能反复选择相似场景。PAL 因此要求 acquisition function 只消费标准推理产物与外部图像 embedding。

其核心观察是，一个可靠目标常在 NMS 前形成较密集的候选簇，同时拥有较高最终置信度；FP 往往位于这二维空间的另一侧。这个边界会随主动学习轮次移动，适合用轻量、按类更新的 classifier 表示，而不必访问 backbone feature。

## 方法总览

每轮先用当前检测器在已标注集和未标注池推理。对每个最终框，统计与其 IoU 匹配的 pre-NMS boxes 数、最终 confidence 和预测类别；已标注集还可根据真值标记 TP/FP。Logistic-based Instance Uncertainty Scoring（LIUS）为每类训练二元 logistic classifier，未标注框的 TP 概率经 Shannon entropy 得到实例不确定性。随后按类频率分配预算，每类先取 `2bc` 张候选图，再由 Global Uncertainty and Image Diversity Estimate（GUIDE）重排为 `bc` 张。

## 方法详解

类预算使用标注集和未标注池的检测频率计算稀有度 `rc=1-0.5(nc,l/Nl+nc,u/Nu)`，因此低频类获得更大 `bc`。每类 logistic classifier 输入仅两个数：pre-NMS count 与 detection confidence；概率靠近 0.5 的框熵最高。论文显示 BDD100K 的 bus 类从首轮 TP/FP 大量混叠，逐轮变成 TP 向高置信高候选数区域移动、FP 向低置信区域移动。

GUIDE 包含三项。Class-Weighted Image Entropy（CWIE）累加图中各检测类别熵并乘稀有度，避免高频类控制图像分数；Rare-Class driven Diversity Index（RCDI）对图内唯一类别的稀有度求和，奖励同时含多种长尾类的图；Rank-Conditioned Similarity Penalty（RCSP）用 Google ViT 图像 embedding，对按 LIUS 排序的候选逐个与更高排名图算最大余弦相似度，只惩罚后出现的重复图。最终分数默认 LIUS 权重 `0.9`，GUIDE 总权重 `0.1`，其中 CWIE、RCDI 各 `0.04`、RCSP `0.02`。

pre-NMS count 并非简单的目标密度统计。PAL 先把整图所有 pre-NMS boxes 按 IoU 分配给 NMS 后的每个最终框，因此特征描述的是“有多少相邻候选共同支持这个实例”。一个高置信框若只有极少前置候选，可能是孤立错误；大量候选但最终置信度低，则可能是定位或类别尚未收敛的困难对象。logistic classifier 学到的是这两个信号的联合决策边界，LIUS 再选择边界附近而非单纯低置信样本。

RCSP 的按排名惩罚避免了对称相似度筛选的常见问题。若两张图高度相似，传统做法可能同时降低二者分数，导致一个重要模式完全消失；PAL 保留 LIUS 更高的那张，只惩罚后续重复图。结合每类候选池和 RCDI，同一张包含多个稀有类别的图还可能同时满足多个预算，但实际送标注前需要去重，这也是实现时必须核对图像预算与实例预算差异的地方。

早期轮次是 GUIDE 最有价值的阶段，因为 LIUS 的每类边界尚由少量样本估计，某些 COCO 稀有类甚至不足以训练稳定分类器。随着标注增多，实例边界更清晰，外部多样性项的相对作用下降，个别后期消融还可能超过完整 PAL。因此实际系统可考虑按轮次衰减 GUIDE，而不能把论文固定权重机械地用于无限轮主动学习。

论文同时报告 PPAL 所需额外标注比例，避免只按“选了相同比例图片”比较。检测图像的实例数差异很大，两个方法即便选择相同张数，也可能产生不同框标注成本。PAL 在 COCO 和 VOC 分别让 PPAL 多用 `18.6%` 与 `22.8%` 标注，因此成本归一化后的优势比单纯轮次曲线更有实际意义。

若标注平台按框而非按图计费，类预算和去重顺序必须在选择前显式纳入成本模型。

## 实验与证据

- 评估覆盖 COCO、PASCAL VOC、BDD100K，检测器含 RetinaNet、Faster R-CNN、SSD、YOLOX-Tiny、YOLO11s；基线包括 Random、Entropy、PPAL、CoreSet、CDAL、MIAL、DivProto、LearnLoss。
- BDD 每轮新增 2.5% 数据。RetinaNet 第 2–5 轮 PAL 为 `40.1/43.7/45.7/46.7 mAP50`，PPAL 为 `38.9/42.5/44.4/45.5`，Entropy 最终 `44.8`。YOLOX-Tiny 最终 PAL `13.3`，Entropy `12.2`。
- COCO 上 RetinaNet 最后一轮相对既有最佳提升 `1.4 AP@[.5:.95]`，VOC/RetinaNet 提升 `0.9 mAP50`，BDD/RetinaNet 提升 `1.2 mAP50`。论文统计 PPAL 在 COCO/VOC 平均分别需要多 `18.6%/22.8%` 标注。
- YOLO11s 在 2% 随机种子集后每轮再加 2%，Round 2–5 的 PAL 为 `5.1/7.9/10.2/12.2 AP`，Random 为 `4.1/6.7/8.7/10.7`，证明接口可迁移到 YOLO 系列。
- 消融显示去掉 CWIE、RCDI 或 RCSP 都会降低早期轮次成绩，LIUS-only 在低数据阶段明显变差；GUIDE 权重 `0.1` 接近最优。用独立 validation 训练 logistic classifier 几乎没有收益，XGBoost 则在低频类过拟合，不如简单线性边界。

## 对 YOLO-Agent 的启发

PAL 很适合成为 YOLO-Agent 的标注调度器：Agent 只需统一导出 pre-NMS boxes、NMS 后框与 confidence，就能在不同 YOLO 版本间复用 LIUS；图像相似度由外部 encoder 处理，不污染检测器训练。尤其可将类预算直接对接数据湖，让低频类别在每轮获得明确配额。

**Harness**：以同一初始 2% COCO 标注集比较 Random、图像熵、LIUS-only、完整 PAL，并在 YOLO11s 与 RetinaNet 上各跑四轮；固定每轮图像数和实际框标注数，同时报告按实例成本归一化结果。观测 AP、APrare、每类新增实例、候选重复率、LIUS 校准 AUC、pre-NMS 导出耗时和选择总时长。通过阈值为第二轮起平均至少比 Random 高 `1 AP`，长尾类新增实例占比提高 `20%` 且总体 precision 不下降，跨两种检测器方向一致；若 logistic classifier 某类正负样本不足、RCSP 使场景多样但目标信息下降，或按框数计费后优势消失，则不通过。

## 优点

- 仅用推理输出，跨 one-stage、two-stage 和 YOLO 的可移植性强。
- 同时融合实例边界不确定性、类别不平衡与图像多样性。
- 主表给出多轮、多模型和实际标注量差异，而非只看最后一轮。

## 局限

- 必须能导出 pre-NMS boxes；封闭部署接口可能只返回最终框。
- 极稀有类别缺少足够 TP/FP 时，按类 logistic classifier 无法稳定训练。
- GUIDE 依赖外部图像 encoder，且后期部分分量可能反而压低成绩。

## 评分

- **创新性**: ★★★★☆
- **证据强度**: ★★★★★
- **工程可用性**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★★
