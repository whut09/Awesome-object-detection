---
title: "DQ-DETR: DETR with Dynamic Query for Tiny Object Detection"
description: "解析 DQ-DETR 如何预测图像级目标数量、动态确定 query 数，并按前景相关性从 encoder tokens 中选择查询。"
tags: [ECCV2024, tiny-object-detection, DETR, dynamic-query, aerial-images]
---

# DQ-DETR: DETR with Dynamic Query for Tiny Object Detection

## 论文与代码

- 论文：[ECCV 2024 官方页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9775_ECCV_2024_paper.php)
- 代码：[官方 DQ-DETR 仓库](https://github.com/Justwzb/DQ-DETR)

## 一句话总结

DQ-DETR 先用 Categorical Counting Module 把每张图的目标总数预测成离散区间，再由 Selective Query Recollection Module 按该数量从多尺度 encoder tokens 中挑出最像前景的位置作为 decoder queries，从而避免固定 300 个 query 在密集小目标图上不够、在稀疏图上又浪费。

## 研究背景与问题

DETR 系列通常为所有图像使用固定 query 数。COCO 每图目标有限，这种设计可行；VisDrone、AI-TOD、DOTA-v2.0 等航拍数据却可能一图数百乃至上千个微小实例。query 少时，一对一匹配从容量上限制召回；简单把 query 增到 1000–1500 又会让空查询占多数，增加 decoder 计算并制造更多背景竞争。已有 two-stage DETR 从 encoder 选 top-k proposals，但 k 仍固定，且排序分数主要按类别置信度，早期训练对 tiny objects 不可靠。DQ-DETR 将“要多少 query”和“取哪些 query”拆成两个可学习问题。

## 方法总览

Categorical Counting Module（CCM）对 encoder 多尺度特征做池化与融合，把连续目标数离散为若干 count bins，训练时使用类别交叉熵；推理时取预测区间对应的 query 数，并设置上下限保证数值稳定。Selective Query Recollection Module（SQRM）为每个 encoder token 估计 class-agnostic foreground score，将位置编码、内容特征和参考点组合后排序，按 CCM 给出的动态数量收集 queries。它们接入 DINO：动态 queries 进入 decoder，后续 Hungarian matching、denoising training 和迭代框精化保持不变。

## 方法详解

CCM 不直接回归目标个数，而做 categorical counting，原因是图像计数噪声和长尾跨度会使 L1 回归被高密度样本主导。离散类别允许网络先判断“稀疏、中密、超密”等容量等级；补充材料给出计数模块伪代码，说明其输入来自多层 encoder features，而非最终检测结果反推。预测 bin 再映射成 query budget，所以计数误差只在相邻预算区间内影响 decoder 容量。

SQRM 不是随机截取或固定 top-k 类别 logits。它建立轻量前景分支，对 encoder tokens 输出类别无关 objectness，再按动态 budget 选择；所选 token 的内容向量作为 query content，对应空间位置作为 reference point。独有数据流是“多尺度特征→encoder tokens→CCM 预测 count bin→得到 Nq→SQRM 计算全 token 前景分数→取 top-Nq 内容与坐标→DINO decoder→一对一集合预测”。图像越密，Nq 越大；稀疏图自动缩小 decoder 序列。

## 实验与证据

论文在 VisDrone、AI-TOD 与 DOTA-v2.0 上评测，并以 DINO/Deformable DETR 类模型为主要基线。VisDrone 上 DQ-DETR 达到 37.0 AP、60.9 AP50，超过同配置 DINO-5scale 的 35.5 AP；对小目标密集的场景，提升主要来自 recall 与 AP75，而非只提高低阈值命中。DOTA-v2.0 消融表中，Categorical Counting Module 单独带来约 0.5 mAP，Selective Query Recollection Module 单独约 0.7 mAP，二者合用最佳，说明动态数量和内容选择不是同一作用。

论文还比较固定 query 数：query 从较小值增大时召回先升，但过多会加重背景学习与计算，性能不再单调增加；DQ-DETR 能按图像密度获得更好的精度—成本折中。可视化显示，密集车辆、行人和远距离小目标区域被分配更多查询，而大面积空背景不再占用同等数量。其基线训练还将每图最大预测数提高以容纳密集实例，因此复现时必须区分“最终输出上限”与“动态 decoder query”两种因素。

## 对 YOLO-Agent 的启发

YOLO 没有 decoder query，但 DQ-DETR 的 Categorical Counting Module（CCM）与 Selective Query Recollection Module（SQRM）可迁移为 NMS 前候选预算器：CCM 估计图像密度，SQRM 决定各 FPN 层 top-k 或稀疏 token。VisDrone 主测并用 AI-TOD、DOTA-v2.0 复查时，**对照组**设为固定 top-k=300/600/1200、真实目标数 oracle、仅 CCM 动态 top-k、仅 SQRM 前景重排、CCM+SQRM。所需**指标**包括 APs、AR500/AR1000、漏检率、每图候选数、NMS 时间、密度分桶延迟，以及 CCM 的 bin accuracy、绝对计数误差和预算—真实目标数单调性。超密图 recall 提升不到 2 点、稀疏图候选数减少不足 30%，或相邻计数 bin 抖动造成视频延迟方差过大，任一项即为**失败判断**；若 oracle 有效而 CCM 无效，则定位为计数模块失败，而不是否定候选路由。

## 优点

- 直接解决固定 query 容量与航拍密度变化的结构性矛盾。
- 数量预测与 query 内容选择分工明确，可独立消融。
- 动态减少稀疏图计算，同时为超密图保留召回空间。

## 局限

- 计数低估会形成硬容量瓶颈，后续 decoder 无法补回缺失目标。
- CCM 的离散区间和上下限依赖训练集密度分布。
- 方法以 DETR 为原生载体，迁移到稠密 YOLO 需要重解释 query budget。

## 评分

- 创新性：8/10
- 实证充分性：8/10
- 工程可迁移性：8/10
- 对 YOLO-Agent 价值：8/10
