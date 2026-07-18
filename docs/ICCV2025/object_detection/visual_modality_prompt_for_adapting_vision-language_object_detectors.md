---
title: "[论文解读] Visual Modality Prompt for Adapting Vision-Language Object Detectors"
description: "解析 ModPrompt 与 MPDR 如何适配红外、深度模态并保留开放词汇检测器的 RGB 零样本知识。"
tags: ["ICCV 2025", "目标检测", "开放词汇检测", "模态适配", "视觉提示"]
---

# Visual Modality Prompt for Adapting Vision-Language Object Detectors

**论文**：https://arxiv.org/abs/2412.00622  
**官方代码**：https://github.com/heitorrapela/ModPrompt  
**会议**：ICCV 2025

## 一句话总结

ModPrompt 冻结 YOLO-World 或 Grounding DINO，用 U-Net 式编码器—解码器从每张红外/深度输入生成三通道图像条件提示，并以可开关的 Modality Prompt Decoupled Residual（MPDR）修正离线文本嵌入，从而提高目标模态 AP，同时保留 COCO 零样本能力。

## 研究背景与问题

开放词汇检测器主要在 RGB 图文数据上预训练，直接面对 LLVIP、FLIR 红外图像或 NYUv2 深度图时，光谱、纹理与对比度分布改变会使零样本检测退化。全量微调和只调检测头虽能适配，却会改写视觉—语言对齐；固定边框、随机像素、Padding、WeightMap 等视觉提示又对每张图施加相同线性扰动，无法根据场景决定增强目标还是抑制背景。

论文研究带标签目标模态适配：只访问冻结 RGB 检测器和目标模态数据，不用源域训练集；适配后不仅要提高 AP，还要能关闭新增参数，恢复原始离线词表的推理。

## 方法总览

目标图像 `x` 进入 ModPrompt 编码器—解码器 `hφ`，输出值域 0–1 的三通道伪 RGB 提示；它与原图逐像素相加后送入冻结检测器 `fθ`。类别侧先由冻结文本编码器离线生成目标类别 embedding，再叠加独立可学习的 MPDR。检测损失只更新 `hφ` 与残差，不反传检测器和文本骨干；部署时启用残差得到模态词表，或零掩码残差恢复原语义。

## 方法详解

传统视觉提示优化固定 `v`，最小化 `Ldet(fθ(x+v),Y)`；ModPrompt 改为 `Ldet(fθ(x+hφ(x)),Y)`。`hφ` 采用 U-Net 风格，编码器可用 MobileNet 或 ResNet 预训练权重，解码器末层限制为三通道。它没有重建目标图监督，而由冻结检测器的分类与框回归损失塑形，因此学习的是有利于检测的伪影和背景抑制，而非自然图像翻译。

MPDR 为每个离线类别向量学习解耦残差。直接改写 embedding 虽能适配目标域，却会丢失原知识；MPDR 始终保留冻结原向量，并可在测试时零成本开关。文本向量预计算也避免训练中反复运行文本编码器。

## 实验与证据

数据包括 LLVIP-IR（12,025 训练、3,463 测试，仅行人）、FLIR-IR、NYUv2-Depth，检测器为 YOLO-World 与 Grounding DINO。对照含 Zero-Shot、Head Fine-Tuning、Full Fine-Tuning、Fixed、Random、Padding、WeightMap。加入 MPDR 后，YOLO-World 的 ModPrompt 在 LLVIP 为 AP50 96.60、AP75 77.27、AP 67.37；NYUv2 为 44.67、32.53、29.93。Grounding DINO 的 NYUv2 AP 也达到 17.27，显著高于静态提示。

知识保留表中，HFT 的 LLVIP/COCO AP50 为 93.57/0.66，FT 为 97.43/0.10，显示灾难性遗忘；ModPrompt 只训练 3.08M 参数，取得 95.63/51.90，平均 73.77。消融显示扩大提示块不稳定，NYUv2 的 200 像素提示会退化；轻量 MobileNet 有时超过 ResNet。MPDR 对多数提示策略有效；定性结果则暴露小目标、重复框和高 IoU 定位不足。

## 对 YOLO-Agent 的启发

可将其实现为检测器前的可撤销模态适配器：Agent 根据传感器选择 RGB 直通、IR-ModPrompt 或 Depth-ModPrompt，并独立管理词表残差。即使源域图像不参与梯度，训练任务也必须保留 COCO 回放评测，阻止 Agent 只追逐目标域 AP。

### Harness

对照组设为冻结 YOLO-World、仅检测头微调、WeightMap、ModPrompt、ModPrompt+MPDR；统一 LLVIP 划分、词表、阈值、分辨率和训练步数。观测 AP/AP50/AP75、COCO AP50 保留率、参数、延迟、小目标 AP 与重复框率。通过条件：相对最佳输入级提示至少提升 5 AP，COCO AP50 下降不超过 0.5，三种子方向一致，延迟增幅低于 10%；若收益依赖 COCO 下降超过 2 点，或 AP50 上升而 AP75 与小目标 AP 同降，则失败。

## 优点

- 输入条件化提示比静态像素更能处理场景差异。
- 冻结检测器和原文本嵌入，可测量并恢复零样本知识。
- 覆盖两类检测器、红外和深度，证据范围完整。
- MobileNet 版本给出实时化选择。

## 局限

- 仍需目标模态框标注，并非无监督适配。
- 小目标、重复框和精细定位不稳定。
- 生成器增加前处理计算，边缘端成本需另测。

## 评分

| 维度 | 评分 |
|---|---:|
| 方法新颖性 | 8.5/10 |
| 实验证据 | 8.5/10 |
| 模态接入价值 | 9.0/10 |
| 工程成熟度 | 8.0/10 |
| 综合 | 8.5/10 |
