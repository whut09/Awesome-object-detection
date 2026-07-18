---
title: "[论文解读] Rectify the Regression Bias in Long-Tailed Object Detection"
description: "解释长尾检测的 RCNN 回归偏置，以及 CAB、回归头聚类和合并三种修正方案。"
tags: ["ECCV 2024", "目标检测", "长尾学习", "边界框回归", "LVIS"]
---

# Rectify the Regression Bias in Long-Tailed Object Detection

**论文**：https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4069_ECCV_2024_paper.php  
**官方代码**：未发现论文声明的官方代码  
**会议**：ECCV 2024

## 一句话总结

论文发现 LVIS 稀有类不仅分类差，其 RCNN 类专属框回归损失也显著更高，于是以共享定位统计修正类专属头，并提出 Extra Class-Agnostic Branch（CAB）、Clustering Heads、Merging Heads 三种可插拔方案。

## 研究背景与问题

EQLv2、SeeSaw、ECM、RFS 等方法主要重加权分类梯度、调整分数或重采样，默认框回归能正常学习。作者按 frequent/common/rare 统计后发现，RCNN 末端回归损失呈明显长尾，而 RPN 的类无关回归损失近乎均衡。训练集与验证集的类别平均框尺度比较还显示稀有类尺度偏移更大；少样本和尺度迁移共同使每类独立 `Wi` 难以泛化。

核心假设很明确：回归偏置主要来自类专属 RCNN 回归头，稀有类需要共享统计，频繁类仍应保留专属能力。

## 方法总览

标准 Faster R-CNN 经 backbone、RPN、ROIAlign 和 RCNN feature 后，为每类输出 `ri=Wi^T fn`。CAB 额外学习全类别共享头 `W0` 并与 `Wi` 混合；Clustering Heads 按实例数或平均框尺度排序分组，每组共享头；Merging Heads 直接让 rare、common 或 frequent 的指定集合共用回归器。

## 方法详解

CAB 的有效头为 `W'i=αW0+(1-α)Wi`。`α=0` 是原模型，`α=1` 等价于全部共享；主实验取 `α=0.5`，既向稀有类注入大量类别共享定位统计，又不完全抹掉频繁类模式。

聚类方案先按训练实例数 `n` 或平均框尺度 `s` 排序，再划分 K 组；合并方案更粗，例如仅合并 rare、仅 common 或 rare+common。作者最终把 CAB 用于主结果，因为它实现最简单、能直接叠加现有长尾分类方法；最优数值有时来自合并 common 头，说明收益核心确实是共享回归统计。

## 实验与证据

主数据集为 LVIS1.0，检测器是 Mask R-CNN，覆盖 ResNet-50/101 与 Swin-T/B，并与 CE、RFS、EQLv2、SeeSaw、ECM 组合。CE 基线 AP 23.7、APr 14.2、框 APr 13.4；CAB 在 `α=0.5` 时为 25.1、17.5、18.0。尺度聚类 K=100 得 25.2/16.7/16.7；合并 common 头达 AP 25.5、APr 17.7。

高 IoU 的 AP75–90 改善更明显，符合“框更准”的机制。COCO-LT 最稀有组 AP1 在三种 RFS rate 下分别从 2.3→6.2、3.6→7.5、8.7→10.8。均衡 COCO 上，Faster R-CNN R50 AP 从 37.3→38.3，R101 从 39.4→39.9，ResNeXt-101 从 41.0→41.7。把类无关分支用于 mask head，也从 23.7 提升到 24.1。消融显示 rare 回归损失下降，重复框、漏框和边界均改善。

## 对 YOLO-Agent 的启发

YOLO 通常已采用类无关框回归，不能机械复制 CAB；更有价值的是诊断原则：按类别频次统计 DFL/IoU loss、框尺度偏差和 AP75，确认长尾是否已蔓延到定位。若业务头存在类别条件化回归、专家路由或类别特定 anchor，可加入共享回归残差或按尺度聚类专家。

### Harness

对照组为原始长尾 YOLO、仅分类重加权、共享回归残差、按频次聚类专家、按尺度聚类专家；固定 RFS 与分类损失。观测 rare/common/frequent 回归损失、AP50/AP75、框尺度误差、重复框率及最稀有四分位 AP。通过条件：rare AP75 至少提升 1.5、rare 回归损失下降 10%，frequent AP 下降不超过 0.5；若只升 AP50、尺度误差不降，或类无关 YOLO 根本没有频次相关回归偏置，则判定不适用。

## 优点

- 找到被分类研究遮蔽的回归偏置。
- 三种方案便于验证共享统计的因果假设。
- 在 LVIS、COCO-LT、COCO、box 与 mask 上泛化。
- CAB 改动局部，可叠加已有方法。

## 局限

- 依赖类专属 RCNN 回归头，对默认 YOLO 不一定成立。
- 强基线越高时增益越小，性能上限未解释。
- 聚类统计在分布变化后可能需要重估。

## 评分

| 维度 | 评分 |
|---|---:|
| 问题洞察 | 9.0/10 |
| 方法简洁性 | 8.5/10 |
| 实验证据 | 9.0/10 |
| YOLO-Agent 价值 | 7.5/10 |
| 综合 | 8.5/10 |
