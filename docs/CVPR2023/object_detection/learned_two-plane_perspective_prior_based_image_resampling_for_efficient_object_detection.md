---
title: "[论文解读] Learned Two-Plane Perspective Prior Based Image Resampling for Efficient Object Detection"
description: "该方法学习地面与上方平面的透视先验，以可微 warp 放大远处目标并压缩无关区域。"
tags: ["CVPR 2023", "目标检测", "透视先验", "图像重采样", "小目标"]
---

# Learned Two-Plane Perspective Prior Based Image Resampling for Efficient Object Detection

**会议**: CVPR 2023  
**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Ghosh_Learned_Two-Plane_Perspective_Prior_Based_Image_Resampling_for_Efficient_Object_CVPR_2023_paper.html)  
**代码**: 论文未提供官方仓库  
**任务**: 自动驾驶与固定交通相机高效检测

## 一句话总结

论文把道路场景近似为 ground plane 与其上方第二平面，学习两平面的透视参数和对象尺寸分布，由此生成 saliency prior 并 warp 输入，使远处小车占据更多像素、天空和近处空区域被压缩。

## 研究背景与问题

边缘设备通常均匀缩小高分辨率交通图像，最先损失的是远处小目标。FOVEA 等端到端 saliency warp 容易坍塌，前帧框先验又不适合新目标。道路与固定相机包含稳定几何：物体多落在地面或某个高架平面，其像素尺寸随透视位置规律变化，可作为可学习但强约束的先验。

## 方法总览

模型预测 horizon、vanishing geometry 与 two-plane 参数，把图像位置映射到预期对象尺度，得到 geometry-guided density。密度经平滑和归一化转为采样网格，原图被非均匀重采样后送入同一个 detector；检测损失反向更新 prior 参数。第二平面覆盖交通灯、标志等不在地面的对象，避免单 ground-plane 偏置。

## 方法详解

Two-plane prior 将图像中的竖直位置、地平线和相机透视关系映射为期望 object scale；ground plane 解释车辆/行人接地点，上方平面补充交通灯、标牌等悬空目标。两张 density map 合成后积分为单调采样坐标，因此 warp 会连续地把远处区域拉开，而不是以检测框为中心做不连续 crop。


## 实验与证据

自动驾驶设置中，在同 detector 与尺度下，小目标 APs 提升 4.1 点（39%），流式 sAPs 提升 5.3 点（63%）。固定交通相机上，相对 naive downsampling 的 APs 增 12.5 点（195%），相对 SOTA warp 增 4.2 点（63%）。消融比较 uniform、dataset prior、single-plane、two-plane 与 learned geometry，并报告 latency 和 memory。

## 对 YOLO-Agent 的启发

应按相机 ID 学习/冻结 prior，测试相机俯仰变化、坡道和镜头震动；对照 uniform、oracle box density、one-plane、two-plane。记录 APs、warp Jacobian、远近区域像素预算、显存和 sAP。若第二平面只在训练相机有效，换相机 APs 低于均匀缩放，或网格把近处大目标切断，就判定几何先验过拟合。

## 优点

- 将可解释场景几何注入输入采样。
- 对远处小目标和流式指标都有显著证据。
- 与 detector 架构正交。

## 局限

- 非道路、坡地和移动相机姿态会破坏两平面近似。
- warp 改变对象形状和像素密度。
- 每类相机可能需要重新标定或微调。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
