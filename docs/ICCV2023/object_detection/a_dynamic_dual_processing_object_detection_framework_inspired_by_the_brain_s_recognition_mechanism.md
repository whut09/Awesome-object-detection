---
title: "[论文解读] A Dynamic Dual-Processing Object Detection Framework Inspired by the Brain's Recognition Mechanism"
description: "DDP 将 CNN 的局部匹配与 Transformer 的全局检索放入同一检测器：Dual-Stream Encoder 搜索跨流融合位置，Dynamic Dual-Decoder 再用选择性掩码协调两个解码器。"
tags: ["ICCV 2023", "目标检测"]
---

# A Dynamic Dual-Processing Object Detection Framework Inspired by the Brain's Recognition Mechanism

**论文**：[官方论文](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_A_Dynamic_Dual-Processing_Object_Detection_Framework_Inspired_by_the_Brains_ICCV_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

DDP 将 CNN 的局部匹配与 Transformer 的全局检索放入同一检测器：Dual-Stream Encoder 搜索跨流融合位置，Dynamic Dual-Decoder 再用选择性掩码协调两个解码器。

## 研究背景与问题

CNN 对纹理和局部形状敏感，Transformer 更擅长全局关系；直接并联两套检测器会成倍增加计算，而且两个解码器可能重复处理同一区域。论文借用“熟悉—回忆”双过程解释，要求模型按目标需求调用不同路径。

## 方法总览

共享 backbone 后分成 CNN 编码流和 Transformer 编码流。作者为两条流的深度与连接建立搜索空间，用 NAS 选择特征交换位置。解码阶段生成选择性 mask：局部证据充分的候选交给 CNN，依赖上下文的候选交给 Transformer，再融合结果。

## 方法详解

### 1. 共享浅层特征，避免完整复制两个骨干。

### 2. 在 Dual-Stream Encoder 中搜索 CNN/Transformer 的层数、连接方向和融合位置。

### 3. 根据候选特征产生选择性掩码，为两个 decoder 分配不同对象。

### 4. 联合优化两路检测损失与掩码，使分工形成，而不是简单平均预测。

## 实验与证据

论文在多种源检测器上报告 3.0–3.7 mAP 的提升，同时不增加 FLOPs。消融显示，朴素双流融合收益有限；搜索得到的编码连接和 selective mask 都是增益来源。比较时必须把双分支参数量、实际延迟与源检测器一起报告，不能只引用 FLOPs。

## 对 YOLO-Agent 的启发

分别关闭 DSE 跨流连接、NAS 搜索结果和 selective mask，记录两解码器处理的候选比例与重复框率。若固定手工融合即可达到完整模型，或掩码关闭后 AP 不变，DDP 的核心设计不成立；若 FLOPs 不变但真实时延明显上升，也不能作为默认组件。

## 优点

把局部与全局建模的互补关系落实到编码和解码两个阶段。

## 局限

结构由搜索空间和双解码器共同决定，复现复杂；“不增 FLOPs”不代表内存访问和并行效率不变。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

