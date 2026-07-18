---
title: "[论文解读] Parallel Rectangle Flip Attack: A Query-Based Black-Box Attack Against Object Detection"
description: "PRFA 利用先验引导的矩形扰动、并行宽度搜索和符号翻转，在有限查询下攻击目标检测器。"
tags: ["ICCV 2021", "目标检测", "黑盒攻击", "PRFA"]
---

# Parallel Rectangle Flip Attack: A Query-Based Black-Box Attack Against Object Detection

**论文**：[官方论文原文](https://openaccess.thecvf.com/content/ICCV2021/html/Liang_Parallel_Rectangle_Flip_Attack_A_Query-Based_Black-Box_Attack_Against_Object_ICCV_2021_paper.html)  
**PDF**：[官方 PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Parallel_Rectangle_Flip_Attack_A_Query-Based_Black-Box_Attack_Against_Object_ICCV_2021_paper.pdf)  
**代码**：[catalog 未提供独立官方代码，返回论文页面](https://openaccess.thecvf.com/content/ICCV2021/html/Liang_Parallel_Rectangle_Flip_Attack_A_Query-Based_Black-Box_Attack_Against_Object_ICCV_2021_paper.html)  
**发表**：ICCV 2021  
**类别**：General Object Detection · Adversarial Attack

## 一句话总结

**Parallel Rectangle Flip Attack（PRFA）** 只读取检测器返回的预测框和 top-1 分数，用先验引导降维确定关键区域，以多矩形并行随机搜索扩大每次查询覆盖，再通过矩形内部扰动符号翻转做深度搜索，从而在约 4000 次查询预算内压低检测 mAP。

## 研究背景与问题

分类黑盒攻击通常面对单个类别向量，检测攻击却要同时破坏数量不定的框、类别和位置；现实 API 还可能只返回每框 top-1 类别与分数，而不是完整 logits。逐像素或单方块搜索因此维度高、一次查询收益低，且不同候选框之间的优化方向可能冲突。

PRFA 把搜索拆成三个紧密连接的部件：**prior-guided dimensionality reduction** 从目标区域或轮廓先验采样矩形；**parallel attack** 在一次查询中同时扰动多个位置以加速宽度搜索；**rectangle flip** 对矩形块整体翻转符号，利用相邻像素相关性推进深度搜索。最终算法动态调整矩形边长、并行点数和步长。

## 方法总览

设干净图像为 `x`、对抗图像为 `x̂`、预测框为 `b_n`、真值框为 `g_m`、预测类别为 `c_n`、真值类别为 `y_m`。攻击成功条件是每个目标均满足 `IoU(b_n,g_m)<τ` 或 `c_n≠y_m`。目标函数在 `||x̂-x||_p≤ε` 下联合最小化高置信框的 IoU 与低置信框的 top-1 分数差；`ζ` 决定当前优化定位还是置信度，`λ` 平衡两项。论文将同类预测框分组，避免逐框调用模型。

## 方法详解

PRFA 先依据检测输出构造关键区域，再采样方形或矩形扰动。并行宽度搜索一次提出 `P` 个位置，早期 `P` 大以覆盖搜索空间，后期逐渐减小以精修；候选只有在目标函数下降时才接受。矩形翻转不是重新估计每个像素，而是把局部块的扰动符号整体反转；论文以卷积响应上界说明，同一语义 clique 内采用一致符号可产生更大的特征变化。

攻击采用 `L∞` 约束，实验中 `ε=0.05`，IoU 成功阈值 `τ=0.50`，步长在指定查询节点减半。`AQ` 表示生成一个对抗样本所需的平均查询次数；攻击后 mAP 越低越强。

## 实验与证据

论文在 COCO 上攻击 Faster R-CNN/ResNet-50、YOLOv3/DarkNet-53、FCOS 和 ATSS，并与 NES、ZO-SignSGD、SignHunter、SquareAttack 比较；NES 预算 4000，ZO-SignSGD 为 4040，其余为 4000。PRFA 将 Faster R-CNN 的 mAP 从 0.51 压到 0.21，ATSS 从 0.54 到 0.20，YOLOv3 从 0.45 到 0.24，FCOS 从 0.54 到 0.23。

Faster R-CNN 消融中，`SS w. Prior`、`SS w. Prior & Flip`、完整 PRFA 的 mAP 分别为 0.26、0.24、0.21，平均查询数为 3666、3342、3331；Flip 贡献 29% 的 mAP 降幅和 97% 的查询节省。相对 SquareAttack，PRFA 再降低约 0.07 mAP并少约 440 次查询。

## 对 YOLO-Agent 的启发

YOLO-Agent 应在推理 API 外围接入 PRFA，不访问梯度：适配器只暴露 `{boxes, top1_class, top1_score}`，攻击器维护 `x̂、ε、query_count、rectangle_size、parallel_P`，每轮批量提交正负符号候选。Harness 对照 Clean、SquareAttack、`SS w. Prior`、完整 PRFA，固定 4000 查询与 `ε=0.05`，报告 mAP、mAP50、mAP75、平均查询数和攻击成功率。若 YOLO mAP 在 4000 查询后仍高于干净值的 70%，或 PRFA 不比 SquareAttack 少至少 200 次查询且 mAP 不低至少 0.02，则判为失败，并检查 API 截断、随机种子和并行矩形更新。

## 优点

- 符合只返回框与 top-1 分数的现实黑盒接口。
- 并行矩形搜索同时降低搜索维度与查询次数。
- 在两阶段、anchor-based 与 anchor-free 检测器上均给出结果。

## 局限

- 仍需数千次查询，面对严格限流或收费 API 成本较高。
- 依赖可见预测框和分数；强截断输出会削弱目标函数信息。
- 数字扰动的有效性不等同于物理世界鲁棒性，迁移性也随结构变化。

## 评分

**8.3/10**：问题设定贴近检测 API，矩形、并行与翻转的消融清楚；但查询成本仍高，防御评测需同时考虑接口裁剪和随机化。
