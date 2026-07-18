---
title: "[论文解读] Interactron: Embodied Adaptive Object Detection"
description: "Interactron 学习探索策略与自监督适应损失，让 embodied agent 在测试环境中移动并更新 DETR。"
tags: ["CVPR 2022", "具身感知", "Interactron", "测试时适应", "DETR"]
---

# Interactron: Embodied Adaptive Object Detection

**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Kotar_Interactron_Embodied_Adaptive_Object_Detection_CVPR_2022_paper.html)  
**代码**: [allenai/interactron](https://github.com/allenai/interactron)  
**任务**: 具身交互式目标检测

## 一句话总结

Interactron 让 agent 在测试房间主动移动收集多视角，Supervisor Transformer 同时输出 learned adaptation loss 和下一动作 policy，检测器据此做无标签梯度更新；相对 DETR 提高 11.8 AP、19.1 AP50。

## 研究背景与问题

静态检测假设测试图像预先给定且模型冻结，但机器人可以改变视角：走近物体、绕开遮挡或观察背面。随机移动能提供额外帧，却不保证帧对适应有用；传统 test-time adaptation 依赖熵等手工目标，也未利用可控动作。

## 方法总览

Detector 采用 DETR，处理起始帧和 rollout frames，产生 image features 与 detection tokens。冻结的 Supervisor Transformer 接收这些 token：Detection Token 输出经 MLP 聚合成 learned loss，用于更新 Detector；Policy Tokens 产生动作分布，指导 agent 选择前后左右/旋转。训练时枚举短轨迹，以最终适应后的检测增益 IFGA 选 ideal exploration action，再行为克隆训练 policy。

## 方法详解

Interactron-Rand 保留 learned loss 但随机探索，用于分离“适应目标”和“动作选择”贡献。完整模型每走一步都利用历史帧决定下一动作，只有 Detector 在测试时更新，Supervisor 保持冻结，避免无标签环境中监督器漂移。多帧 fusion baseline 不做参数适应，因此可判断收益是否只是看到更多图像。

## 实验与证据

在具身室内环境中，Interactron 相对 DETR 提升 11.8 AP 与 19.1 AP50；在外观差异显著的新环境中接近使用全监督目标域训练的模型。对照包括单帧 DETR、随机 policy、多帧 Transformer、Interactron-Rand 和 learned policy；学习探索优于随机，learned loss 也优于仅聚合帧。

## 对 YOLO-Agent 的启发

需要在 AI2-THOR 类环境固定动作预算，对照无更新、多帧融合、熵最小化、随机探索、完整 Interactron。记录每步 AP gain、移动距离、碰撞率、适应耗时和更新后原域遗忘。若策略只学会靠近所有物体却在遮挡场景无优势，或无标签更新提高当前房间但破坏已知类，应失败。YOLO 部署还要把反向传播时间计入在线决策周期。

## 优点

- 把主动感知与测试时学习统一起来。
- learned loss 不需要测试标签。
- 动作策略可针对检测收益而非覆盖率训练。

## 局限

- 需要可控相机和可微在线更新资源。
- 轨迹枚举随动作步数快速增长。
- 模拟环境到真实机器人存在域差。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★☆
- **工程价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
