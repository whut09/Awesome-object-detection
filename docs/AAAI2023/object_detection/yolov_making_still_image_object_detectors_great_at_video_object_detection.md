---
title: "[论文解读] YOLOV: Making Still Image Object Detectors Great at Video Object Detection"
description: "YOLOV 用 FSM 筛选高置信 proposal，再由 FAM 在参考帧间聚合对象级关系，为 YOLOX 注入时序信息。"
tags: ["AAAI 2023", "视频目标检测", "YOLOV", "FSM", "FAM"]
---

# YOLOV: Making Still Image Object Detectors Great at Video Object Detection

**会议**: AAAI 2023  
**论文**: [arXiv](https://arxiv.org/abs/2208.09686)  
**代码**: [YuHengsss/YOLOV](https://github.com/YuHengsss/YOLOV)  
**任务**: 实时视频目标检测

## 一句话总结

YOLOV 保留 YOLOX 的单帧检测流程，只把各帧候选送入 Feature Selection Module 做 top-k 与粗 NMS，再由 Feature Aggregation Module 在多帧对象特征间执行关系聚合，最后修正关键帧分类与回归。

## 研究背景与问题

视频中的运动模糊、遮挡和罕见姿态会让单帧证据不足，而多数高精度视频检测器依赖两阶段 RPN/ROI 特征，速度难以实时。单阶段检测器的对象表示隐含在密集位置中，若对整张特征图跨帧 attention，计算会随空间尺寸和帧数急剧上升。YOLOV 的关键是先利用成熟单帧预测把时序聚合限制到少量 proposal。

## 方法总览

YOLOX 对关键帧与参考帧产生 boxes、scores 和位置特征。FSM 选取每帧 top-k 置信候选，并以 0.75 NMS 做宽松去重，保留供时序修正的对象 token。FAM 将候选的语义特征、类别分数和框关系编码后进行 self-attention/跨帧关联，把参考帧证据汇入关键帧，增强后的 token 再预测最终分类和位置。

## 方法详解

论文比较 global 与 local reference sampling：全局方案从视频范围抽取参考，覆盖长时外观；局部连续帧保留运动连续性但信息更相似。最终选择全局采样的精度—效率折中。FSM 的粗 NMS 阈值刻意高于最终后处理，防止同一真实对象的偏移框过早丢失。FAM 还比较 cosine similarity 与注意力式选择，并通过降维控制 proposal 关系矩阵成本。

## 实验与证据

ImageNet VID 上，YOLOV 以 YOLOX-S/L/X 为单帧基线，报告约 77% mAP 的高配结果；论文表格拆解参考帧数、global/local sampling、FSM 选择方式、FAM 结构和多尺度聚合。速度表把 detector 与 aggregation 分项列出，例如聚合仅增加数毫秒，明显快于依赖光流或两阶段 ROI 的视频检测基线。

## 对 YOLO-Agent 的启发

Harness 要按视频顺序运行，不能把参考帧特征预先泄漏到计时外。分别测试单帧 YOLOX、仅 FSM 重打分、FSM+FAM，并扫描 top-k、参考帧数和采样跨度；报告 VID mAP、fast/medium/slow motion AP、遮挡恢复率、峰值缓存和每帧延迟。若增益主要来自静态片段、快速运动 AP 下降，或 reference 数增加后延迟线性失控，就说明对象级聚合没有形成有效时序匹配。

## 优点

- 通过 proposal 筛选避免整图跨帧聚合。
- 可建立在不同规模 YOLOX 上，模块边界明确。
- 采样、筛选、聚合均有专门消融。

## 局限

- 首帧漏检的对象不会进入 FSM，后续难以恢复。
- top-k 与 NMS 阈值对拥挤视频敏感。
- 多帧缓存增加在线系统状态管理成本。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★★
