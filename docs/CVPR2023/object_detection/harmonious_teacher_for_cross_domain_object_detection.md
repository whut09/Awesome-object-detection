---
title: "[论文解读] Harmonious Teacher for Cross-Domain Object Detection"
description: "Harmonious Teacher 用分类分数与定位质量的一致性评价目标域伪框，并按 harmony score 重加权，而不是仅保留高分类置信框。"
tags: ["CVPR 2023", "目标检测"]
---

# Harmonious Teacher for Cross-Domain Object Detection

**论文**：[官方论文](https://openaccess.thecvf.com/content/CVPR2023/html/Deng_Harmonious_Teacher_for_Cross-Domain_Object_Detection_CVPR_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

Harmonious Teacher 用分类分数与定位质量的一致性评价目标域伪框，并按 harmony score 重加权，而不是仅保留高分类置信框。

## 研究背景与问题

跨域自训练中，高分类分不代表框位置准确；硬阈值还会丢掉困难但有价值的目标。论文把分类—定位协调性用于伪标签训练和样本排序。

## 方法总览

EMA teacher 生成目标域预测；源域和目标域都施加分类/定位一致性损失。每个伪框根据分类分数与定位质量计算 harmony measure，student 使用连续权重学习所有候选，再由 EMA 更新 teacher。

## 方法详解

### 1. teacher 对无标注目标图生成类别、框与定位质量。

### 2. 计算 classification-localization consistency，并在源/目标样本上正则化。

### 3. 用 harmony score 连续重加权伪框，替代单一置信阈值筛除。

### 4. student 更新后以 EMA 同步 teacher，循环改进伪标签。

## 实验与证据

覆盖 Cityscapes→Foggy Cityscapes、Cityscapes→BDD100K、Sim10K→Cityscapes、KITTI→Cityscapes。KITTI→Cityscapes 的 FCOS 结果为 65.5 AP，超过 OADA 的 59.2；论文总结相对两阶段 MGA 提升 6.8 mAP、相对一阶段 OADA 提升 5.0 mAP。消融表明一致性正则和 harmony 重加权都贡献增益。

## 对 YOLO-Agent 的启发

导出 cls_score、loc_quality、harmony_weight、teacher_student_gap。分别测试硬阈值、仅一致性损失、仅重加权和完整模型。若定位质量与真实 IoU 相关性低，或加权后伪标签召回下降过多，则不启用。

## 优点

直接修正跨域伪标签中“分类高、定位差”的常见问题，并覆盖多种驾驶域迁移。

## 局限

定位质量估计本身可能受域偏移影响；EMA 错误会通过连续权重长期累积。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

