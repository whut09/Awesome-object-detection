---
title: "[论文解读] LEOD: Label-Efficient Object Detection for Event Cameras"
description: "LEOD 以 RVT teacher-student、自训练、time-flip TTA 与时序过滤统一事件流弱监督和半监督检测。"
tags: ["CVPR 2024", "事件相机", "LEOD", "自训练", "RVT"]
---

# LEOD: Label-Efficient Object Detection for Event Cameras

**会议**: CVPR 2024  
**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_LEOD_Label-Efficient_Object_Detection_for_Event_Cameras_CVPR_2024_paper.html)  
**代码**: 论文未提供官方仓库  
**任务**: 低标注事件目标检测

## 一句话总结

LEOD 先用少量标签训练 recurrent vision transformer（RVT），再离线前向/反向事件序列并用 time-flip TTA 生成伪框，通过 IoU-aware 置信、时序轨迹和 noisy-label suppression 筛选后训练 student，统一 sparse-frame 弱监督与 unlabeled-stream 半监督。

## 研究背景与问题

事件相机微秒级输出，逐时间窗标框极昂贵。普通图像 SSOD 忽略事件 detector 的 recurrent memory，也只能用过去信息在线推断；离线制伪标签却可以利用未来事件。低标注下，短暂噪声脉冲和断续轨迹会产生大量不稳定框。

## 方法总览

阶段一在 1%–10% 标注上训练 RVT-S teacher。阶段二对未标窗口执行正常时间和 time-flip 两次推理，反向序列提供未来上下文；候选按 objectness、per-class IoU、NMS 和跨时间一致性过滤，边界附近的不可靠框被 ignore 而非当背景。student 在人工框与可靠 pseudo boxes 上重训，再迭代更新。

## 方法详解

LEOD 的 reliable label selection 先利用 RVT 输出的 objectness 与 per-class IoU 形成质量分数，再做 NMS 和跨时间轨迹检查。time-flip TTA 将事件极性/时间顺序变换后从未来向过去推理，两方向一致框置信度更高；低质量区域被标为 ignore，student 不会把潜在目标强制学习成背景。


## 实验与证据

Gen1 上，RVT-S 在 1%/2% 标签时分别提高 8.6/7.8 mAP；1Mpx 上 10% 标签甚至超过 100% fully-supervised counterpart。论文覆盖 weakly-supervised 与 semi-supervised 两套比例，并消融 time-flip TTA、temporal filtering、ignore region、伪标签阈值；全标注场景继续提升，说明机制不只补数据量。

## 对 YOLO-Agent 的启发

Harness 要严格按 stream 划分 labeled/unlabeled，检查 future information 只用于离线 teacher，不能泄漏进在线延迟。对照单向伪标签、time-flip、时序过滤和完整 LEOD，报告 pseudo precision/recall、轨迹长度分桶 mAP、标注小时数。若伪框在静止背景事件上密集出现，或 10% 超全监督仅来自不同训练 epoch，应判为失败。

## 优点

- 统一弱监督与半监督事件检测。
- 利用离线未来信息提升伪标签质量。
- 在两套事件数据和多标注比例上验证。

## 局限

- 离线双向推理增加训练制标成本。
- teacher 错误可能在迭代中自增强。
- 在线部署仍受 RVT recurrent state 管理影响。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★★
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★★
