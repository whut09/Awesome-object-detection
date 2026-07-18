---
title: "[论文解读] Scaled-YOLOv4: Scaling Cross Stage Partial Network"
description: "Scaled-YOLOv4 以 CSP 化和结构感知复合缩放构建 YOLOv4-tiny 与 P5/P6/P7 大模型族。"
tags: ["CVPR 2021", "目标检测", "Scaled-YOLOv4", "CSP", "模型缩放"]
---

# Scaled-YOLOv4: Scaling Cross Stage Partial Network

**会议**: CVPR 2021  
**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html)  
**代码**: 论文页面未给独立官方仓库  
**任务**: 从低端设备到大型 GPU 的实时检测

## 一句话总结

Scaled-YOLOv4 不只按 depth/width/resolution 放大网络，还依据 CSP stage、feature pyramid 层数和输入尺寸改变拓扑，向下得到 YOLOv4-tiny，向上形成 CSP-P5、P6、P7 三种感受野与检测层级。

## 研究背景与问题

分类网络的复合缩放不直接适用于检测：输入放大后小目标需要更浅层，大目标需要更深 pyramid；仅加深会增大串行路径，仅加宽会抬高内存访问。CSP 能把梯度分成两路并减少重复计算，为上下缩放提供结构支点。

## 方法总览

小模型侧分析 Dense、OSA 与 CSP 计算，缩短主干、采用适合低端设备的轻量检测头。大模型侧先把 YOLOv4 backbone/neck/head CSP-ized，再随输入分辨率增加 pyramid stage：P5 使用三检测层，P6/P7 增加下采样层和更大感受野；宽度、深度、输入和 stage 数联合调整。

## 方法详解

向下缩放时，论文比较 Dense block、OSA 与 CSP-OSA 的计算路径，目标是缩短串行卷积并减少 concat 产生的内存流量。向上缩放则把 CSP 放入 backbone、PAN neck 和检测 stage；当分辨率从 P5 扩到 P6/P7 时，不只增加层数，还同步调整每个 stage 的深度与宽度，使新增大感受野层不会吞噬浅层小目标特征。


## 实验与证据

YOLOv4-large 报告 COCO 55.5 AP、73.4 AP50、V100 约 16 FPS；TTA 为 56.0 AP。YOLOv4-tiny 为 22.0 AP、RTX 2080Ti 约 443 FPS，TensorRT FP16 batch4 报 1774 FPS。论文比较 CSP 前后计算、不同 tiny block、P5/P6/P7 与单纯宽深缩放，证明结构层数随分辨率变化的重要性。

## 对 YOLO-Agent 的启发

缩放实验要画 APs/APm/APl、activation memory 和串行关键路径，不能只列参数。对照固定 P5 的宽深放大、增加 P6、增加 P7；同一 GPU、batch1 重测，TTA 与 batch4 不进入实时主表。若新增 P7 只改善大目标且小目标因浅层容量减少而下降，或显存越过部署预算，即判该缩放点失败。

## 优点

- 首批系统展示 YOLO/CSP 的双向缩放。
- 将 pyramid 拓扑纳入复合缩放变量。
- tiny 到 large 覆盖极宽计算范围。

## 局限

- 不同型号速度口径和硬件并不统一。
- 大模型训练与 TTA 成本高。
- anchor/NMS 配置随尺度变化仍需调节。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★☆
