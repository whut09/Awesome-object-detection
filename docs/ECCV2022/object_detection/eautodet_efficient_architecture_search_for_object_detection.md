---
title: "[论文解读] EAutoDet: Efficient Architecture Search for Object Detection"
description: "EAutoDet 以 kernel reusing 与 dynamic channel refinement 联合搜索 YOLO backbone 和 FPN。"
tags: ["ECCV 2022", "目标检测", "EAutoDet", "NAS", "Kernel Reusing"]
---

# EAutoDet: Efficient Architecture Search for Object Detection

**会议**: ECCV 2022  
**论文**: [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2483_ECCV_2022_paper.php)  
**代码**: [vicFigure/EAutoDet](https://github.com/vicFigure/EAutoDet)  
**任务**: 高效检测架构搜索

## 一句话总结

EAutoDet 在 COCO 上直接建立 backbone+FPN 可微 supernet，通过 kernel reusing 把同一 edge 的候选操作合并为一次卷积，再用 dynamic channel refinement 搜索有效通道，把检测 NAS 成本压到 1.4 GPU-days。

## 研究背景与问题

检测 supernet 同时含 backbone、FPN 和高分辨率特征，若像 DARTS 那样并行执行所有候选，显存和训练时间难以承受；只搜索 ImageNet backbone 又忽略检测 neck 的配合。EAutoDet 选择 YOLO one-stage 框架，直接优化检测损失与实测速度。

## 方法总览

每条搜索边包含不同卷积候选，但共享/复用一个最大 kernel 权重，通过裁剪或变换产生小 kernel 输出，架构权重再混合候选，避免逐操作完整前向。dynamic channel refinement 为通道设置可学习选择，搜索过程中逐步收紧宽度。backbone 与 FPN 在同一 supernet 中联合更新，选定离散结构后从头训练检测器。

## 方法详解

Kernel reusing 将同一 edge 上不同 kernel size 的候选视为一个最大卷积核的中心裁剪，架构概率只组合共享卷积产生的子输出，因此显存不再随候选数线性增长。Dynamic channel refinement 使用可学习通道门逐步决定有效宽度，并把 backbone stage 的输出通道与 FPN 横向连接一起离散化，避免搜索结束后手工对齐维度。


## 实验与证据

搜索仅需 1.4 GPU-days。COCO test-dev 上找到的快模型达到 40.1 mAP、120 FPS，高配结构为 49.2 mAP、41.3 FPS；迁移到 DOTA-v1.0 旋转检测得到 77.05 mAP50、21.1M 参数。消融比较无 kernel reusing、无动态通道、只搜 backbone 与联合 FPN，验证内存降低和精度收益。

## 对 YOLO-Agent 的启发

Harness 应保存 supernet ranking 与独立重训 ranking 的 Kendall 相关，报告搜索峰值显存、GPU-days、最终 AP/FPS 和候选通道利用率。对照 DARTS 并行操作、仅 kernel sharing、完整 EAutoDet。若共享 kernel 让候选排序与重训结果失相关，或通道 refinement 总偏向最大宽度，就应终止搜索；目标后端若与 V100 latency 排名不一致，需重新建立设备约束。

## 优点

- 联合搜索 backbone 和 FPN，不依赖 ImageNet supernet。
- 搜索成本和显存显著低于常规检测 NAS。
- 结构可迁移到旋转检测。

## 局限

- 权重共享会产生 architecture ranking bias。
- 搜索空间仍受预定义卷积操作限制。
- 120 FPS 的硬件与推理配置需统一复测。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
