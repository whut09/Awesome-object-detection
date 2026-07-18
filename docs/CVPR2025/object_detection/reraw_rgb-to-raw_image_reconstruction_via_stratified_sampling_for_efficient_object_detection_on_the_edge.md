---
title: "[论文解读] ReRAW: RGB-to-RAW Image Reconstruction via Stratified Sampling for Efficient Object Detection on the Edge"
description: "ReRAW 用 context encoder 与 gamma-space multi-head 重建传感器 RAW，并以亮度分层采样扩充检测训练。"
tags: ["CVPR 2025", "RAW", "ReRAW", "Stratified Sampling", "边缘检测"]
---

# ReRAW: RGB-to-RAW Image Reconstruction via Stratified Sampling for Efficient Object Detection on the Edge

**会议**: CVPR 2025  
**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Berdan_ReRAW_RGB-to-RAW_Image_Reconstruction_via_Stratified_Sampling_for_Efficient_Object_CVPR_2025_paper.html)  
**代码**: [SonyResearch/ReRAW](https://github.com/SonyResearch/ReRAW)  
**任务**: RGB 合成 RAW 与边缘目标检测

## 一句话总结

ReRAW 从 RGB context encoder 提取全局颜色调制，多个 gamma-space heads 生成 RAW candidates 并融合；训练 patch 按高亮 RAW 像素分层采样，随后用合成 BDD-ReRAW 预训练 RTMDet/YOLOX/SSD，再在真实 PASCALRAW 或 NOD-Nikon 微调。

## 研究背景与问题

直接在 RAW 上检测可绕过 ISP、保留 12–16bit 动态范围，但大规模带框 RAW 数据难收集。简单 inverse ISP 在高亮区域误差大，合成 RAW 域差会让检测预训练失效。论文把 RGB-to-RAW 重建质量与下游小模型检测放在同一链路评价。

## 方法总览

RGB patch 与全图上下文进入 encoder，context 参数调节转换；multi-head architecture 在 gamma space 预测多个 sensor-specific RAW 候选，融合后用 logarithm-based high-light loss 强化高值像素。stratified sampling 按亮度区间选训练 patch，避免暗区域数量压倒高动态样本。生成的大型合成 RAW 集用于 detector pretraining，最后接少量真实 RAW fine-tune。

## 方法详解

Multi-head converter 不直接在高动态线性 RAW 域回归单一结果，而在 gamma space 产生多个候选，让不同 head 处理颜色映射歧义；context encoder 输出全局 modulation 参数后再融合候选。Logarithm-based high-light loss 对高值像素相对误差更敏感，stratified sampler 又提高含亮区 patch 的出现频率，两者分别作用于目标函数和数据分布。


## 实验与证据

ReRAW 在五个 RAW 数据集取得 SOTA reconstruction。表 2 比较 L1/L2 与 logarithm high-light loss，表 3验证 context encoder、multi-head 和采样；表 4在 PASCALRAW、NOD-Nikon 上比较 RGB pipeline、RGB 权重直接微调 RAW、传统 inverse ISP 合成和 ReRAW 预训练。RTMDet-s 对训练配方最敏感，YOLOX/SSD 波动较小。

## 对 YOLO-Agent 的启发

自检必须同时看 PSNR/高亮分桶误差与 detector mAP，不能用重建指标替代任务收益。对照随机 patch、亮度 stratified、不同 gamma heads，以及 RGB pretrain→RAW fine-tune 与 ReRAW pretrain。若 PSNR 提高但高亮车辆 AP75 下降，或合成 sensor pattern 与目标相机不匹配导致负迁移，就判定重建链路失败。

## 优点

- 解决标注 RAW 稀缺而非只改 detector。
- 分层采样针对高亮长尾像素。
- 用三种紧凑 detector 验证下游作用。

## 局限

- RGB 已经过 ISP，无法完全恢复丢失 RAW 信息。
- sensor-specific 转换跨相机泛化有限。
- 训练链路增加重建模型和合成数据成本。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★★
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★☆
