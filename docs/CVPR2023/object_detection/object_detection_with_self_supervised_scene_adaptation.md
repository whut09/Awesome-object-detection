---
title: "[论文解读] Object Detection With Self-Supervised Scene Adaptation"
description: "论文针对固定摄像头视频，让检测器与跟踪器交叉生成伪标签，并利用稳定背景设计无伪影 object mixup 和背景输入，实现逐场景自监督适应。"
tags: ["CVPR 2023", "目标检测"]
---

# Object Detection With Self-Supervised Scene Adaptation

**论文**：[官方论文](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Object_Detection_With_Self-Supervised_Scene_Adaptation_CVPR_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

论文针对固定摄像头视频，让检测器与跟踪器交叉生成伪标签，并利用稳定背景设计无伪影 object mixup 和背景输入，实现逐场景自监督适应。

## 研究背景与问题

固定视角场景的背景长期稳定，但通用检测器不会利用这一先验。直接用自身预测自训练容易确认偏差，普通复制粘贴又会在静态背景上留下明显边缘。

## 方法总览

检测器和 DiMP 跟踪器互相补充伪框；动态背景从长视频中估计，作为额外模态输入。对象从原帧取出后按背景一致性混合到新位置，生成 artifact-free mixup，适应模型仍混入 COCO 源数据防止遗忘。

## 方法详解

### 1. 在视频中运行检测器和跟踪器，用跨模型一致结果形成伪真值。

### 2. 从固定机位序列估计干净背景，显式提供给检测器。

### 3. 执行位置感知、无边缘伪影的对象混合，扩充场景内位置分布。

### 4. 每个场景独立适应，并以原始基模型作为比较起点。

## 实验与证据

Scenes100 含 100 个固定摄像头视频、约 2160 万帧、168.4 万个标注框，平均每视频约 840 个框。评价使用 APG，即 100 个视频上“适应后 AP−基线 AP”的平均值。基线 M2 在适应前 Scenes100 为 41.11 mAP/63.10 AP50；论文方法明显优于 STAC、TIA、LODS 等域适应方案，并消融 cross-teaching、背景模态和无伪影 mixup。

## 对 YOLO-Agent 的启发

每个视频记录 APG、伪标签精度、跟踪补框比例、背景更新周期。分别关闭 tracker、background input、artifact-free mixup。若适应后某些场景下降且无法由伪标签质量解释，或混合图边界被检测器当作目标线索，则不部署。

## 优点

充分利用固定机位的背景等变性，并发布大规模 Scenes100。

## 局限

方法依赖静态相机；相机移动、背景长期变化或跟踪漂移会破坏伪标签。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

