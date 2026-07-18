---
title: "[论文解读] From Easy to Hard：单点监督红外小目标检测的渐进主动学习"
description: "解析 PAL 的 Model Pre-start、EPG、COU、FIU 与衰减式伪标签更新，并复盘 SIRST3 等数据集上的单点监督证据。"
tags: ["ICCV 2025", "红外小目标", "单点监督", "主动学习", "伪标签", "弱监督"]
---

# From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision

**论文**：[arXiv 官方页面](https://arxiv.org/abs/2412.11154)  
**官方代码**：[YuChuang1205/PAL](https://github.com/YuChuang1205/PAL)  
**发表**：ICCV 2025

## 一句话总结

PAL 不让低能力模型一开始就对全部点标注做标签演化，而是通过 **Model Pre-start、Coarse Outer Updates（COU）和 Fine Inner Updates（FIU）**，先学可靠易样本，再逐批吸收难样本并抑制伪标签过度膨胀。

## 研究背景与问题

红外小目标的像素级标注昂贵，单点监督更现实，但 LESPS 直接让所有点标签参与训练，并在固定 epoch 或损失条件下演化，容易出现早期错误扩散、标签区域持续变大以及强骨干无法发挥。PAL 的核心判断是：未经训练的网络没有可靠的难度估计能力，因此“先选难样本”本身就不可信，必须先让模型获得基本任务能力。

**Easy Pseudo-label Generation（EPG）**只处理点标注周围 20×20 patch：Gaussian filtering 抑噪，Canny 提取轮廓，morphological closing 修补并填充；删除不含标注点或面积超阈值的连通域，再以目标级 recall≥0.8 判为 easy。对漏掉的目标，最终把原始点标签补回伪标签，避免易样本筛选以牺牲召回为代价。

## 方法总览

PAL 把单点监督训练组织成 preparation pool 与 training pool 的动态迁移。EPG 先从点周围生成高召回粗掩码并识别易样本，Model Pre-start 只用这些样本建立基本能力；中期 COU 周期性把当前模型已能处理的难样本移入训练池，FIU 同步收缩背景、扩张可信目标；末期停止扩池，只继续细化后来样本。EEDM loss 负责像素优化，但课程与伪标签更新才是框架主体。

## 方法详解

COU 用当前预测的 missed rate Rm 与 false rate Rf 对 preparation pool 重新分组；预测区域若不与任何真点相交就删除，再补入漏检点。阈值 Tm 从 0.2 随 epoch 线性增到 1，保证难样本逐步进入。FIU 以当前伪标签连通域中心裁 patch，用自适应阈值从预测中提取 candidate area，删除与伪标签中心不相交的区域，最后更新 Ln+1：候选区内取旧标签与预测均值，候选区外乘 decay factor λ。这个 λ 同时允许目标区扩张和背景区收缩，防止 LESPS 式无界演化。训练采用强调边缘和困难像素的 **EEDM loss**。

## 实验与证据

数据集为 **SIRST3、NUAA-SIRST、NUDT-SIRST、IRSTD-1K**，样本数分别 2755、427、1327、1001；指标是 IoU、nIoU、Pd、Fa。AdamW 初始学习率 1e-3、batch 16、400 epoch，三阶段比例为 0–0.2、0.2–0.8、0.8–1，COU/FIU 每 5 epoch 执行。论文把 ACM、ALCNet、MLCL-Net、ALCL-Net、DNANet、GGL-Net、UIUNet、MSDA-Net 分别接入 PAL，并与传统方法、全监督和 LESPS 比较。

在 SIRST3-Test 上，MSDA-Net 的 coarse point+LESPS 为 46.26 IoU、45.73 nIoU、85.38 Pd、36.16 Fa；换 PAL 后为 **69.38/71.55/97.41/16.34**。centroid point+PAL 为 **69.21/72.40/97.01/15.70**。八种骨干上 PAL 相对 LESPS 的 IoU 提升范围为 9.68–24.04 个百分点。关键消融中，仅 Model Pre-start 得 56.24 IoU，加入 FIU 为 61.66，完整 MP+COU+FIU 为 69.38；一开始让全部样本进入训练池仅 32.86 或 25.72。λ=0.97 最佳，过小到 0.75 时 Fa 升至 29.45。EEDM 的 69.38 IoU 略高于 BCE 69.26，但明显优于 Dice 64.38，说明主要收益来自课程式伪标签机制，而非损失替换。

## 对 YOLO-Agent 的启发

点监督实例检测可把 PAL 改造成框或掩码伪标签课程。**对照组**：固定同一 YOLO-seg 与点标注，比较全样本直接训练、LESPS 统一演化、仅 EPG 易样本启动、EPG+COU、完整 EPG+COU+FIU，并交叉测试 `λ={1,0.97,0.9}`；再把 EEDM 换回 BCE，以隔离损失贡献。**指标**：每 5 epoch 记录 training pool 占比、伪标签目标级 precision/recall、面积膨胀率、IoU、nIoU、Pd、Fa，以及新加入难样本前后的性能跳变。**失败判断**：若 EPG 的初始 precision 不优于全样本粗标签，COU 扩池后 Fa 连续恶化，FIU 无法阻止面积单调膨胀，或拿掉 EEDM 后 PAL 的优势消失，就说明收益并非来自“由易到难”的池更新，不能迁移到 YOLO-Agent。

## 优点

- 把“模型何时有能力学习难样本”纳入训练状态，而非固定时间触发。
- 可接入八种 SIRST 网络，框架适配性证据充分。
- 对全样本启动、COU、FIU、衰减因子均有直接消融。

## 局限

- EPG 依赖亮点、边缘和小面积先验，对低对比或扩展目标可能失效。
- 多个阈值、三阶段比例及 5-epoch 周期增加调参成本。
- 单点到实例框任务的转换尚未在通用 YOLO 数据上验证。

## 评分

- 问题重要性：★★★★★
- 方法独特性：★★★★☆
- 实验证据：★★★★★
- 工程可迁移性：★★★★☆
- YOLO-Agent 参考价值：★★★★★
