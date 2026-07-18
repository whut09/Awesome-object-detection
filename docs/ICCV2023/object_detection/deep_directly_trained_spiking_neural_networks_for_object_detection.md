---
title: "[论文解读] Deep Directly-Trained Spiking Neural Networks for Object Detection"
description: "EMS-YOLO 直接用代理梯度训练脉冲检测器，核心 EMS-ResNet 通过膜电位捷径构建全脉冲残差块，在 4 个时间步内完成 COCO 与 Gen1 检测。"
tags: ["ICCV 2023", "目标检测"]
---

# Deep Directly-Trained Spiking Neural Networks for Object Detection

**论文**：[官方论文](https://openaccess.thecvf.com/content/ICCV2023/html/Su_Deep_Directly-Trained_Spiking_Neural_Networks_for_Object_Detection_ICCV_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

EMS-YOLO 直接用代理梯度训练脉冲检测器，核心 EMS-ResNet 通过膜电位捷径构建全脉冲残差块，在 4 个时间步内完成 COCO 与 Gen1 检测。

## 研究背景与问题

ANN-to-SNN 转换通常需要数百时间步才能逼近连续激活，不适合低延迟检测。直接训练又容易在深层残差网络中出现梯度消失/爆炸，并且框坐标需要连续值。

## 方法总览

输入图像或事件被编码为 T 个时间步。EMS-ResNet 的 residual path 与 membrane shortcut 保持脉冲信息流，多尺度特征送入 YOLO 式 neck；检测头读取最后膜电位回归连续框，同时输出类别。

## 方法详解

### 1. 用 LIF 神经元和 surrogate gradient 直接训练，不先训练 ANN 再转换。

### 2. EMS-Block 让捷径传递膜电位相关信息，论文从理论上分析其梯度稳定性。

### 3. 帧数据和事件流统一成 T 步张量，骨干与颈部保持全脉冲计算。

### 4. 检测头使用最后时刻膜电位表示连续框坐标，随后执行常规 NMS。

## 实验与证据

EMS-YOLO 只用 4 个时间步，优于至少需要 500 步的 ANN-SNN 转换方法；在 COCO 帧数据和 Gen1 事件数据上达到与同结构 ANN 可比的检测性能，并给出 5.83 倍更低的估算能耗。关键消融比较 SEW-ResNet、MS-ResNet 与 EMS-ResNet，以及不同时间步和膜电位捷径。

## 对 YOLO-Agent 的启发

变量为 time_steps∈{2,4,8}、ems_shortcut、lif_threshold、spike_rate。除 COCO mAP 和 Gen1 指标外，按 backbone/neck/head 统计非零脉冲、AC/MAC 次数和真实延迟。若 4 步模型只能靠提高放电率追平 ANN，或能耗优势在检测头连续运算后消失，则不接入。

## 优点

把低时间步、深层残差和连续框回归同时纳入设计，并提供公开 EMS-YOLO 实现。

## 局限

5.83 倍来自操作级能耗估算，具体芯片上的访存、NMS 和膜电位更新可能缩小优势。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

