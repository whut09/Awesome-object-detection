---
title: "[论文解读] AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection"
description: "解析 AdaptiveISP 如何用 actor-critic 为每张图动态选择 ISP 模块及参数，并以冻结检测器误差下降作为奖励。"
tags: ["NeurIPS 2024", "目标检测", "图像信号处理", "强化学习", "低照度"]
---

# AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection

**论文**：[官方论文页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cc596a803bedc7a03a87e98c77a22efe-Abstract-Conference.html)  
**代码**：[官方代码](https://github.com/OpenImagingLab/AdaptiveISP)  
**项目页**：[AdaptiveISP](https://openimaginglab.github.io/AdaptiveISP/)  
**发表**：NeurIPS 2024

## 一句话总结

AdaptiveISP 把 RGB-domain ISP 配置建模为 MDP，由 actor-critic 针对每张线性 RGB 图像逐阶段选择 Exposure、CCM、Sharpen/Blur、Tone Mapping 等模块并预测连续参数，以冻结 YOLOv3 检测误差的下降作为奖励，还可通过计算时间惩罚在推理时动态改变精度—延迟取舍。

## 研究背景与问题

传统 ISP 为人眼图像质量设计，固定模块顺序和参数不一定服务检测；已有任务驱动 ISP 方法或只搜索参数，或训练后仍使用固定 pipeline，难以应对低照度数据中随 ISO、噪声、色偏和动态范围变化的场景。直接让检测器吃 raw 数据又会绑定传感器格式，并不总优于经过处理的 RGB。

论文的关键观察是，多数输入只需要少量处理模块，少数困难图才值得继续加工。于是 ISP 不应是一条对所有图相同的长流水线，而应是带停止条件的逐图决策过程，同时联合解决离散模块选择与连续参数回归。

## 方法总览

输入为经简单静态 raw-domain 处理得到的 linear RGB。状态 `si` 是当前阶段图像，动作拆为模块选择 `aM` 与参数 `aΘ`，执行可微 ISP 模块 `F(Mi,Θi)` 得到 `si+1`。CNN policy network 输出模块 softmax 与参数 tanh，CNN value network估计状态价值，二者通过 actor-critic 优化。

基础奖励为冻结检测器误差下降 `D(si)-D(si+1)`。YOLOv3 只向 ISP 回传梯度，自身权重不更新。策略可连续选择模块，直到停止状态或达到最大阶段数；模块使用记录和当前 stage 作为额外输入，使相同模块不会被无意义地反复选择。

## 方法详解

**混合动作策略。** `πM` 对 ISP 模块做离散分布，`πΘ` 为被选模块预测连续控制量。整个轨迹包含图像状态、模块和参数，折扣回报衡量后续所有处理对检测误差的累计改善，因此策略能够接受“当前变化小但为后续模块创造条件”的动作。

**探索与复用约束。** 模块使用记录以 `N` 个通道拼入网络，stage 另占一通道；复用惩罚限制同一模块重复出现。策略输出熵惩罚 `Pe` 的系数从 1 逐步降到 0，训练前期鼓励充分探索模块池，后期转向稳定利用。

**计算时间惩罚。** 论文实测每个 ISP 模块耗时 `Mc`，定义 `Pc=λc Σ Im Mc`。增加 `λc` 后，策略会偏好 Exposure、CCM 等较快模块，减少 Sharpen/Blur、Denoise 等昂贵操作，而无需重新训练检测器或固定整条 pipeline。

## 实验与证据

主要数据包括 LOD（2230 张 14-bit 低照 raw 图，8 类，1830/400 训练验证）、OnePlus（141 张、3 类街景）以及由 COCO val 5000 张转换的 synthetic raw-like COCO。默认冻结 YOLOv3、输入 512×512，并与 Crafting、NeuralAE、DynamicISP、Hyperparameter Optimization、Attention-aware Learning、ReconfigISP、Refactoring ISP 比较。

LOD 上 AdaptiveISP 的 mAP@0.5/@0.75/@0.5:.95 为 71.4/51.7/47.1，高于 Attention-aware Learning 的 70.9/51.0/46.6。用 LOD 训练后跨域测试，OnePlus 达 70.1/49.7/45.0，raw COCO 达 54.9/40.1/37.7，均为表中最佳。把同一 ISP 输出交给未参与策略训练的 YOLOX 和 DDQ，也分别将 mAP@.5:.95 从 39.2 提至 47.2、从 35.9 提至 52.0。

自适应性分析把 LOD 分为三种代表 pipeline 子集，交叉套用 pipeline 后，只有与场景匹配的配置获得最佳结果。高 ISO/高噪声图更偏好 Desaturation，低 ISO 图偏好 CCM；高动态范围倾向 Tone Mapping，明显色偏则选择 White Balance。阶段消融显示 1/2/3/4/5 阶段的 mAP@.5:.95 为 45.7/46.6/46.9/47.1/47.1，收益在后期饱和。

效率表中 `λc=0` 时总处理 14.73 ms、mAP 47.1；`λc=0.1` 时降至 9.20 ms、mAP 45.9。策略显著减少慢模块使用，证明时间惩罚确实改变动作分布，而非只在报告层面做后处理。

## 对 YOLO-Agent 的启发

YOLO-Agent 可把图像预处理纳入 agent action space：根据曝光、噪声和首轮检测反馈选择少量可微增强，而不是固定 CLAHE/去噪配方。若部署硬件有实时约束，reward 应同时包含检测损失、模块延迟和能耗，并允许 agent 提前停止。

**Harness。** 对照组为固定厂商 ISP 或统一手工增强；实验组为 AdaptiveISP，冻结同一 YOLO 权重。按低照、强噪声、色偏和高动态范围四类切片记录 mAP、误检/漏检、平均模块数、P95 延迟和能耗。通过阈值：整体 mAP@.5:.95 提升至少 1 点，最困难切片提升至少 3 点，P95 延迟不超过对照 1.1 倍；效率模式需在 mAP 下降不超过 1.2 点时降低至少 25% ISP 时间。若策略长期只选同一 pipeline，或跨相机 mAP 下降超过 2 点，则判定自适应性/泛化失败。

## 优点

- 同时搜索模块结构和连续参数，并且决策随输入变化。
- reward 直接来自冻结检测器，优化目标与下游任务一致。
- 跨数据集、跨 YOLOX/DDQ 检测器仍有增益。
- 计算惩罚提供可操作的推理时精度—效率调节机制。

## 局限

- 当前依赖可微 ISP 模块，真实相机中的黑盒或硬件模块难以直接纳入训练。
- 策略由 LOD 上的 YOLOv3 信号训练，跨传感器和极端 raw 分布仍可能退化。
- actor-critic 比固定增强更难稳定复现，也需要准确测量目标硬件模块耗时。
- 输出图像面向检测性能，不保证摄影感知质量或其他视觉任务同时最优。

## 评分

**8.9/10。** 将 ISP 变成逐图、任务驱动、成本感知的决策链非常实用，实验覆盖也充分；可微模块和传感器迁移是落地障碍。
