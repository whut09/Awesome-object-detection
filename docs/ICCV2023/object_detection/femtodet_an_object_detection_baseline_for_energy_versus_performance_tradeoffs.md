---
title: "[论文解读] FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs"
description: "FemtoDet 以实测能耗选择激活、卷积和 neck，并用 IBE 与 RecWR 改善超轻量检测。"
tags: ["ICCV 2023", "目标检测", "FemtoDet", "IBE", "RecWR"]
---

# FemtoDet: An Object Detection Baseline for Energy Versus Performance Tradeoffs

**会议**: ICCV 2023  
**论文**: [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Tu_FemtoDet_An_Object_Detection_Baseline_for_Energy_Versus_Performance_Tradeoffs_ICCV_2023_paper.html)  
**代码**: 论文未提供官方仓库  
**任务**: 常开移动端低能耗检测

## 一句话总结

FemtoDet 不以参数/FLOPs 代理电量，而在 Snapdragon 上逐项测激活、卷积算子和 neck 能耗，再组合 68.77K 参数骨干，以 Instance Boundary Enhancement 和 Recursive Warm-Restart 补偿极小容量的表示与优化缺陷。

## 研究背景与问题

同参数模型可能因 GELU、depthwise kernel 或 feature fusion 产生完全不同功耗；常开摄像头关心瓦特和每帧焦耳，而非桌面 GPU FPS。超轻模型还容易把实例边缘抹平，并在 Mosaic 等强增强分布切换后陷入次优点。

## 方法总览

作者先建立组件能耗表，选择低功耗 ReLU、卷积与 neck 连接构成 FemtoDet。IBE 在有限通道中强化边界响应，让空间差异经局部卷积与重标定回注主特征。RecWR 将训练划为递归周期，每次增强策略变化时 warm restart 学习率，帮助模型重新适应数据分布而非沿旧极小值缓慢收敛。

## 方法详解

IBE 针对超窄通道难同时表达内部纹理和实例边界的问题，从局部卷积响应中提取 boundary-sensitive 分量后回注主干，而不是增加完整高分辨率分支。RecWR 则把强增强引起的数据分布变化视为新的优化阶段，递归恢复较大学习率再衰减；其对照重点是相同总 epoch 下是否比普通 cosine/warm restart 更好。


## 实验与证据

PASCAL VOC 上 FemtoDet 以 68.77K 参数达到 46.3 AP50，在 Snapdragon 865 CPU 为 1.11W、64.47 FPS；COCO 与 TJU-DHD 进一步覆盖复杂场景。组件表显示 GELU 相对 ReLU 可增 4.40% 性能却多 12.50% 能耗。消融分别加入 IBE、RecWR，并比较参数、FLOPs、FPS 与实测功率的失相关。

## 对 YOLO-Agent 的启发

目标板上应读取稳定态功率，报告 joule/frame、温升后 FPS、AP50:95 与 APs，而不是只测冷启动。对照无 IBE、边缘卷积、完整 IBE，以及 cosine、普通 warm restart、RecWR。若 IBE 只改善 VOC AP50 而 COCO AP75 不升，或 RecWR 增益来自更长训练，需判为无效；发生热降频后每帧能耗反升时，模型也不能通过。

## 优点

- 直接把能耗作为检测器设计指标。
- IBE 和 RecWR 分别针对容量与优化瓶颈。
- 在真实移动 CPU 上给出功率数据。

## 局限

- 组件能耗排序具有芯片依赖。
- AP50 不能充分衡量定位精度。
- 极小模型对数据域变化仍很脆弱。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★☆
