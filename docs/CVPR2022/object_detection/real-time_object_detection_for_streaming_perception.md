---
title: "[论文解读] Real-Time Object Detection for Streaming Perception"
description: "该方法以 DFP 融合前后帧、TAL 按运动趋势加权，直接预测处理完成时的下一帧目标。"
tags: ["CVPR 2022", "流式感知", "DFP", "Trend Aware Loss", "sAP"]
---

# Real-Time Object Detection for Streaming Perception

**会议**: CVPR 2022  
**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Real-Time_Object_Detection_for_Streaming_Perception_CVPR_2022_paper.html)  
**代码**: 论文未提供官方仓库  
**任务**: 有处理延迟的在线目标检测

## 一句话总结

论文用 last/current/next 三元组训练实时 detector：Dual-Flow Perception 的 static flow 保留当前外观，dynamic flow 提取两帧运动趋势，Trend Aware Loss 再按对象速度动态平衡当前位置学习与下一帧 forecasting。

## 研究背景与问题

在线评测中，模型处理当前帧时世界仍在运动，输出会在下一时间点才可用；离线 AP 高的框因此产生系统性位移。Streamer 通过调度、跟踪和 Kalman forecasting 包装已有 detector，本文则把未来预测内置网络，并以 streaming AP（sAP）把准确率与真实延迟共同评价。

## 方法总览

上一帧和当前帧分别提取特征。DFP 的 static flow 通过 residual connection 传递检测语义，dynamic flow 对两帧差异建模并编码运动；融合特征直接监督下一帧 boxes/classes。TAL 根据 GT 位移估计每个对象趋势，慢对象强调定位，快对象提高 forecasting 权重，避免统一 loss 被静止物体主导。

## 方法详解

DFP 的 static flow 对 current-frame feature 做残差传递，保证类别和静态背景线索；dynamic flow 显式组合 last/current feature，学习位移趋势后与静态流融合。TAL 根据相邻 GT box 的位移幅度为每个实例生成权重：运动越快，forecasting 项占比越高；近静止目标则避免被不必要外推。


## 实验与证据

Argoverse-HD 流式设置下，论文相对实时基线提升约 4.9 mAP/sAP，并与 Streamer、Adaptive Streamer 及不同 YOLOX 尺度比较。消融包括单帧预测、简单 concat、DFP static/dynamic flow、固定权重与 TAL；可视化显示普通检测框因处理延迟落后，预测框向运动方向前移。

## 对 YOLO-Agent 的启发

必须用时间戳模拟真实 capture→inference→output，按完成时刻匹配 GT，离线 AP 只能作辅助。对照 Kalman、两帧 concat、DFP、DFP+TAL，分桶报告速度、加速度和转向对象 sAP。若高速直线改善却在急转弯严重过冲，或模型延迟变化后固定 one-frame horizon 失配，就判失败；部署后应按实际毫秒动态换算预测 horizon。

## 优点

- 直接优化“输出时世界状态”而非当前帧真值。
- DFP 将外观与运动分成两条信息流。
- TAL 处理对象速度异质性。

## 局限

- 固定帧间隔假设不适合抖动延迟。
- 突发加速和转向难从两帧推断。
- 双帧特征增加内存与训练成本。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★★
