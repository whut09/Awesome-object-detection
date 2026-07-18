---
title: "[论文解读] StageInteractor: Query-based Object Detector with Cross-stage Interaction"
description: "StageInteractor 让 query detector 的不同解码阶段共享动态算子与标签信息，解决每层独立一对一匹配导致的监督稀疏。"
tags: ["ICCV 2023", "目标检测"]
---

# StageInteractor: Query-based Object Detector with Cross-stage Interaction

**论文**：[官方论文](https://openaccess.thecvf.com/content/ICCV2023/html/Teng_StageInteractor_Query-based_Object_Detector_with_Cross-stage_Interaction_ICCV_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

StageInteractor 让 query detector 的不同解码阶段共享动态算子与标签信息，解决每层独立一对一匹配导致的监督稀疏。

## 研究背景与问题

DETR 类模型在每层单独做 Hungarian matching，同一真值在某层只能监督一个 query；早期阶段即使已有多个合理候选，也会因未匹配而被当作负样本。

## 方法总览

前向阶段用轻量 adapter 复用前序阶段的 dynamic operators，增强当前 query；训练阶段的 Cross-Stage Label Assigner 先收集各层一对一匹配结果，再把类别标签重新分配给各层合适预测。

## 方法详解

### 1. 每层先执行原有 query 更新和一对一匹配，保留标准检测损失。

### 2. 跨阶段交互模块读取历史 dynamic operator，并用 adapter 适配当前 query。

### 3. CSLA 汇总多层匹配到同一真值的候选，把跨层可靠类别标签回填到对应阶段。

### 4. 框回归仍遵循当前层匹配，避免跨层错误坐标直接监督。

## 实验与证据

COCO 上 ResNet-50、100 queries、12 epoch 得到 44.8 AP，比基线高 2.2 AP；APs/APm/APl 为 27.5/48.0/61.3。3× 日程和 300 queries 时，ResNet-50、ResNet-101、ResNeXt-101-DCN、Swin-S 分别为 48.9、49.9、51.3、52.7 AP。给 DINO-Swin-B 加 CSLA 后从 55.8 提到 56.2 AP。

## 对 YOLO-Agent 的启发

变量为 history_stage、operator_reuse、CSLA、query_count。逐层记录匹配真值数、重复 query、分类正样本和 AP。若只增加 query 数就获得全部收益，或 CSLA 提高 AP50 却损害 AP75，应停止迁移。

## 优点

交互和标签分配都可插入现有 query detector，并在 DINO 上验证可迁移性。

## 局限

依赖多阶段 decoder 与 Hungarian matching，不能直接套入没有 query stage 的普通 YOLO 头。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

