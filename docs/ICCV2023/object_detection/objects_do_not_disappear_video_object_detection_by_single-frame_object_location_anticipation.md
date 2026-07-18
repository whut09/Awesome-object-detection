---
title: "[论文解读] Objects Do Not Disappear: Video Object Detection by Single-Frame Object Location Anticipation"
description: "该方法从单个关键帧预测未来 box trajectory，以跳过中间帧 backbone 计算并支持稀疏标注。"
tags: ["ICCV 2023", "视频目标检测", "轨迹预测", "稀疏标注", "关键帧"]
---

# Objects Do Not Disappear: Video Object Detection by Single-Frame Object Location Anticipation

**会议**: ICCV 2023  
**论文**: [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Objects_Do_Not_Disappear_Video_Object_Detection_by_Single-Frame_Object_ICCV_2023_paper.html)  
**代码**: 论文未提供官方仓库  
**任务**: 单帧驱动的视频框轨迹预测

## 一句话总结

论文不聚合邻帧特征，而在关键帧检测框和 ROI 特征上接 trajectory subnetwork，连同未来时间索引输入多层 FC，直接预测后续 $T$ 帧 box locations，从而跳过这些帧的昂贵 backbone。

## 研究背景与问题

视频检测通常“每帧提特征再融合”，但采样率高时相邻帧冗余。若物体短时间内不会凭空消失，一个静态关键帧包含的姿态、位置与场景先验足以预测短期运动。该设定还天然适配每秒只标少数帧的数据，因为中间轨迹可以被连续函数补齐。

## 方法总览

关键帧 detector 输出 boxes $B_t$ 与对象特征。trajectory network 分别编码 box、ROI feature 和时间索引，concat 后经三层全连接回归 $t+1...t+T$ 的框。顺序损失累积相邻 offset，防止 bag-style 独立框预测产生跳跃轨迹。稀疏监督版本用连接首尾真值的连续函数 $r_t(\cdot)$ 生成 pseudo-box trajectory，约束终点与下个标注关键帧一致。

## 方法详解

Trajectory subnetwork 对 box、ROI appearance 和每个未来 frame index 分别做等维 FC 编码，再 concat 进入两层回归器。Ordered trajectory loss 不把未来框当无序集合，而累积相邻时间 offset 的误差；稀疏标注时，连续函数 $r_t$ 连接两个已标关键帧，产生中间 pseudo trajectory，并强制最后一步落到真实终点。


## 实验与证据

论文先用 MovingDigits 验证单图能否恢复类相关运动，再在 ImageNet VID、EPIC-KITCHENS-55、YouTube-BoundingBoxes 和 Waymo Open 上评测。与逐帧 detector、视频聚合和稀疏标注基线相比，方法同时改善准确率并减少训练/推理时间；消融比较无时间索引、unordered bag loss、ordered trajectory loss 及不同跳帧长度。

## 对 YOLO-Agent 的启发

Harness 要按预测 horizon 绘制 IoU decay，并报告每秒真正执行 backbone 的次数、轨迹连续性和新生/消失对象 recall。基线包括线性匀速、仅 box FC、box+ROI+time。若 $T$ 增大后终点 IoU 低于匀速外推，或跳帧节省被大量纠错关键帧抵消，就判定 anticipation 不成立；对突然入画对象必须单列漏检，不能由已有轨迹指标掩盖。

## 优点

- 将时序预测用于直接减少 backbone 次数。
- ordered loss 明确约束轨迹连续性。
- 可利用稀疏视频框标注。

## 局限

- 单帧难预测突发转向和相机运动。
- 新出现物体在下一关键帧前不可见。
- 长 horizon 的误差会累积。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★★
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
