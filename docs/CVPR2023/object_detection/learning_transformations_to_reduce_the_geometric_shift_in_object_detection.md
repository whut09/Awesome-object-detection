---
title: "[论文解读] Learning Transformations To Reduce the Geometric Shift in Object Detection"
description: "该方法针对视场角和视点变化造成的几何域偏移，学习目标域变换并进行自训练，而不是只做颜色或风格对齐。"
tags: ["CVPR 2023", "目标检测"]
---

# Learning Transformations To Reduce the Geometric Shift in Object Detection

**论文**：[官方论文](https://openaccess.thecvf.com/content/CVPR2023/html/Vidit_Learning_Transformations_To_Reduce_the_Geometric_Shift_in_Object_Detection_CVPR_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

该方法针对视场角和视点变化造成的几何域偏移，学习目标域变换并进行自训练，而不是只做颜色或风格对齐。

## 研究背景与问题

同一类别在不同相机焦距、安装位置下会出现尺度、透视和可见区域变化；外观域适应不能修复这种坐标结构差异。

## 方法总览

模型从无标注目标域估计一组可微几何变换，把源域训练样本或预测映射到更接近目标域的几何分布。教师在变换后的目标图上生成伪框，学生联合学习源域真值与目标域一致性。

## 方法详解

### 1. 定义可学习的缩放、裁剪、透视等变换族。

### 2. 用目标域预测统计优化变换参数，使源/目标框几何分布接近。

### 3. 将框随图像同步变换，避免伪标签坐标错位。

### 4. 迭代更新教师伪标签和学生检测器，无需相机参数或目标域框。

## 实验与证据

作者分别构造 camera FoV shift 与 viewpoint shift，并在 KITTI、Cityscapes 等驾驶数据间评估。表格显示，学习变换在两种几何偏移下都优于源模型和只做外观自训练的对照；关闭变换学习后，目标域 AP 的主要增益消失。论文没有把某个变换固定为普适最优，而是强调按目标场景学习。

## 对 YOLO-Agent 的启发

变量为 transform_family、FoV_scale、perspective_strength、pseudo_threshold。按目标框面积和图像纵向位置分桶比较 AP，并做正反变换闭环检查。若图像变换后框不能精确映射回原坐标，或几何统计接近但 AP 不升，则判失败。

## 优点

明确分离几何偏移与外观偏移，且不要求相机内外参。

## 局限

可学习变换族限制了能修正的几何类型，严重遮挡或三维结构变化无法由二维变换恢复。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

