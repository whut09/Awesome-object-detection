---
title: "[论文解读] CoTDet: Affordance Knowledge Prompting for Task Driven Object Detection"
description: "CoTDet 不把任务直接映射到封闭类别，而用 MLCoT 从大语言模型提取“任务—对象例子—视觉属性—理由”的可供性知识，再条件化生成检测 query。"
tags: ["ICCV 2023", "目标检测"]
---

# CoTDet: Affordance Knowledge Prompting for Task Driven Object Detection

**论文**：[官方论文](https://openaccess.thecvf.com/content/ICCV2023/html/Tang_CoTDet_Affordance_Knowledge_Prompting_for_Task_Driven_Object_Detection_ICCV_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

CoTDet 不把任务直接映射到封闭类别，而用 MLCoT 从大语言模型提取“任务—对象例子—视觉属性—理由”的可供性知识，再条件化生成检测 query。

## 研究背景与问题

“可以用来坐”的对象可能是椅子、石块或箱子，类别表无法穷举。任务驱动检测需要抓住支撑、容纳、切割等视觉可供性。

## 方法总览

Multi-Level Chain-of-Thought 先为任务生成候选物体和关键视觉属性，并保留推理理由。知识编码后进入 knowledge-conditional detector，调制 object queries 与框回归；GGNN 传播任务、对象和属性节点之间的关系。

## 方法详解

### 1. 输入自然语言任务，生成多层可供性链而非单一类别列表。

### 2. 把对象示例、视觉属性和理由构成图，并用 GGNN 更新节点表示。

### 3. 由知识表示产生 query，结合 ResNet/Transformer 图像特征定位候选。

### 4. 输出框、mask 与解释理由，按任务相关性而非固定类别筛选。

## 实验与证据

论文构建任务驱动检测评估，并与闭集检测、类别文本提示和其他知识提示比较。完整 CoTDet 相对当时方法提升 15.6 box AP 与 14.8 mask AP。消融显示，去掉多层推理、只给对象名或去掉视觉属性都会下降，说明增益并非仅来自更长文本。

## 对 YOLO-Agent 的启发

变量为 prompt_level、object_examples、visual_attributes、GGNN_steps、rationale_on。重点记录 task recall、box/mask AP 和无关物体误检。若随机理由与真实理由表现相同，或只对训练中见过的对象名有效，则可供性知识未真正进入检测器。

## 优点

把开放任务需求转成可定位的视觉属性，并能输出检测理由。

## 局限

依赖外部语言模型知识，提示错误会系统性漏掉可用但少见的对象。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

