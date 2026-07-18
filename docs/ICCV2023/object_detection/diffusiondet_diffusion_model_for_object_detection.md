---
title: "[论文解读] DiffusionDet: Diffusion Model for Object Detection"
description: "DiffusionDet 将目标框集合建模为扩散去噪：训练学习把加噪真值框还原，推理从随机框开始逐步生成对象。"
tags: ["ICCV 2023", "目标检测"]
---

# DiffusionDet: Diffusion Model for Object Detection

**论文**：[官方论文](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

DiffusionDet 将目标框集合建模为扩散去噪：训练学习把加噪真值框还原，推理从随机框开始逐步生成对象。

## 研究背景与问题

固定 proposal 数量和单次解码限制了传统检测器的测试时伸缩性。扩散模型允许同一权重在推理时改变随机框数量和迭代步数。

## 方法总览

训练先把真值框归一化并按时间步加入高斯噪声，检测头根据图像特征、噪声框和时间编码预测类别及干净框。推理从随机分布采样 boxes，反复调用同一检测头去噪，并用 box renewal 替换低质量框。

## 方法详解

### 1. 为每张图补齐固定数量真值框并采样扩散时间。

### 2. 向框坐标加噪，RoI 特征与时间嵌入共同输入 dynamic head。

### 3. 预测原始框或去噪方向，并优化分类、L1/GIoU 等检测损失。

### 4. 测试时执行若干 DDIM 式更新，可增加 proposals 或迭代次数。

## 实验与证据

COCO、LVIS 和 CrowdHuman 上均与 Faster R-CNN、DETR、Sparse R-CNN 比较。COCO 到 CrowdHuman 的零样本迁移中，仅在测试时增加框数与迭代步数即可分别获得合计 5.3 AP 和 4.8 AP 的提升。消融覆盖采样步数、框数量、box renewal 与多步集成。

## 对 YOLO-Agent 的启发

变量为 num_proposals、sampling_steps、noise_schedule、box_renewal。固定权重画 AP-延迟曲线，并统计不同随机种子的预测方差。若更多步数只重复同一框、收益不能覆盖延迟，或随机种子导致 AP 波动过大，则关闭扩散方案。

## 优点

测试时可调计算预算，不需重新训练不同 proposal 数的模型。

## 局限

多次 RoI 与检测头调用增加延迟；随机采样使部署复现和导出更复杂。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

