---
title: "[论文解读] SALISA: Saliency-Based Input Sampling for Efficient Video Object Detection"
description: "SALISA 用前帧检测生成显著图，经 TPS-STN 非均匀缩放后让轻量检测器保留小目标细节。"
tags: ["ECCV 2022", "视频目标检测", "SALISA", "TPS-STN", "非均匀采样"]
---

# SALISA: Saliency-Based Input Sampling for Efficient Video Object Detection

**会议**: ECCV 2022  
**论文**: [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6037_ECCV_2022_paper.php)  
**代码**: 论文未提供官方仓库  
**任务**: 高分辨率高效视频检测

## 一句话总结

SALISA 用关键帧重检测器 $D_{key}$ 产生框级显著图，再让 thin-plate-spline spatial transformer 对下一帧做非均匀下采样，放大前景、压缩背景，随后由低 FLOPs 检测器 $D$ 处理小尺寸图并循环更新显著性。

## 研究背景与问题

视频小目标需要高分辨率，但均匀缩小会把远处车辆压成几像素；每帧都跑大检测器又不现实。相邻帧的显著区域通常连续，可把上一帧“哪里值得保留”用于下一帧采样。难点是 TPS-STN 若只靠检测损失，容易折叠到少数区域或学不到明确放大行为。

## 方法总览

第一帧或周期关键帧输入 $D_{key}$，预测框与分数经 saliency map generator 扩散为连续显著图。该图与下一张高分辨率帧进入 TPS-STN，控制点产生平滑采样网格；重要区域占据更多输出像素，背景被强压缩。小图送入 $D$，结果既用于检测，也生成下一时刻显著图。显式 regularization loss 令学习网格模仿内容感知采样目标，抑制退化形变。

## 方法详解

TPS-STN 不是根据检测框直接裁 patch，而是由一组控制点生成整幅连续 warp；相邻区域仍保持平滑映射。论文额外构造显式 sampling target，要求网格将 saliency 高值区域分配更多输出面积。没有该正则时，检测梯度既稀疏又滞后，控制点可能停在近似均匀采样或形成局部折叠。$D_{key}$ 与 $D$ 的更新节奏则决定显著图能否追上新目标。


## 实验与证据

ImageNet-VID 与 UA-DETRAC 上，SALISA 使 EfficientDet-D1 的 mAP 接近使用更高预算的 D2，D2 接近 D3；D1 的 small-object detection 提升 77%，甚至超过 D3 baseline。论文对照均匀 downsampling、不同 key-frame 周期、无显式正则的 TPS、不同 $D_{key}/D$ 组合，并报告计算量而非只给图像尺寸。

## 对 YOLO-Agent 的启发

应固定同一视频解码成本，对 uniform resize、oracle GT saliency、预测 saliency 和 SALISA 网格四组比较；输出 APs、边缘区域 recall、网格 Jacobian 最小值、关键帧峰值延迟和平均 FLOPs。若镜头切换后显著图滞后、采样网格出现折叠，或低速目标增益以新进入画面的漏检为代价，就判定时序采样失败。关键帧周期必须同时满足 p95 latency，而不只是平均算力。

## 优点

- 在输入域节省算力，可与现有检测器组合。
- TPS 提供平滑、可微的非均匀采样。
- 对小目标改善有明确分桶证据。

## 局限

- 新出现目标和镜头切换没有历史显著性。
- 几何扭曲可能损害长宽比敏感类别。
- 系统包含两个检测器与状态循环，部署较复杂。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
