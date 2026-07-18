---
title: "Towards RAW Object Detection in Diverse Conditions"
description: "解析 AODRaw 多条件数据集、RAW 域预训练与跨域蒸馏对恶劣环境检测的作用。"
tags: ["CVPR 2025", "目标检测", "RAW", "AODRaw", "跨域蒸馏"]
---

# Towards RAW Object Detection in Diverse Conditions

**会议**：CVPR 2025  
**论文**：[arXiv](https://arxiv.org/abs/2411.15678)  
**代码**：[lzyhha/AODRaw](https://github.com/lzyhha/AODRaw)

## 一句话总结

论文发布覆盖九种光照/天气组合的 24MP AODRaw，并用 Synthetic ImageNet-RAW 加 logit/feature cross-domain distillation，让检测器直接在 RAW 域预训练，避免依赖额外 neural ISP。

## 研究背景与问题

sRGB 已经过 ISP 去噪、白平衡和色调映射，视觉友好却可能在低照、雨雾中丢失传感器原始信息；RAW 保留更完整信号，但主流 backbone 在 ImageNet-sRGB 上预训练，直接微调 RAW 会出现明显域差。已有 RAW 检测数据通常类别少、条件单一，方法又常在检测器前增加可训练 ISP，精度提升伴随额外延迟。本文先建立复杂评测场，再研究不加前处理模块的 RAW 预训练。

## 方法总览

AODRaw 使用 6000×4000 Bayer RAW 图像，包含 7,785 张、62 类、135,601 个实例，覆盖室内日光/低光，以及室外 clear、fog、rain、fog+rain 与低光组合共九种条件。默认实验先 demosaic 成三通道并 gamma correction，再下采样到 2000×1333；另一设置切成 1280×1280 patch、重叠 300 像素。方法侧把 ImageNet-1K 在线 unprocess 为 16-bit Synthetic ImageNet-RAW，随机改变亮度和相机噪声，用 RAW 学生网络接受分类监督，同时由同结构 sRGB 教师提供 logit 与全局 feature 蒸馏。

## 方法详解

数据按每种条件分别以 7:3 划分，得到 5,445 张训练图、2,340 张测试图。切片设置忽略与 patch IoU 低于 0.4 的目标，生成 71,782 张 patch 和 417,781 个实例；下采样更快，但会忽略面积小于 `32²` 的目标。评估除 AP/AP50/AP75 和尺度 AP 外，还单独报告 `APnormal`、`APlow`、`APrain`、`APfog`，能识别平均值掩盖的恶劣条件退化。

RAW 预训练每次迭代在线执行 unprocessing，随机采样平均亮度与 shot noise；教师保持现成 sRGB 预训练参数。分类标签提供任务监督，SoftMax 后的教师/学生 logits 用 KL divergence 对齐，最后层全局特征用 L1 对齐。训练完的 RAW backbone 再接 Faster/Cascade R-CNN、GFocal 等检测器微调，无需 RAOD、RAW-Adapter 所用 neural ISP。

## 实验与证据

AODRaw 显示显著域差：Cascade R-CNN+ConvNeXt-T 在 sRGB 训练/测试为 34.0 AP，sRGB 训练后直接测 RAW 仅 28.0；RAW 训练测 sRGB 为 21.2，而 RAW/RAW 达 34.8。sRGB 预训练再微调 RAW 只有 33.7 AP，RAW 预训练提升到 34.8，并在低光、雨、雾达到 32.1、36.1、28.4。相比带 neural ISP 的 RAOD 34.4 AP，本文无额外 ISP 仍达到 34.8。

蒸馏消融由无蒸馏 34.1 AP，加入 logit 变为 34.3，再加 feature 达 34.8，AP75 从 35.9 升至 36.7。RAW 学生的 ImageNet-RAW Top-1 从 81.3 提到 81.8，接近 sRGB 教师 82.1；亮度从 791 降至 80 时，蒸馏模型下降 4.6%，小于无蒸馏 5.0%，噪声等级 1 到 10 的下降也由 4.5% 降至 4.2%。实时表中 YOLOv8-n 加 neural ISP 虽从 18.9 到 19.7 AP，却从 188.1 降到 57.6 FPS。

## 对 YOLO-Agent 的启发

应把 RAW 域适配前移到预训练，而非默认串联 ISP。**Harness** 对照 sRGB 预训练→RAW 微调、RAW 预训练、RAW+logit 蒸馏、RAW+logit+feature 蒸馏，以及 neural-ISP+YOLO；按九种条件记录 AP、AP75、APlow/APrain/APfog、tiny recall、FPS 和预处理耗时。无 ISP 的完整蒸馏相对 sRGB→RAW 至少 `+0.8 AP`，三类恶劣条件均不退化，FPS 保留基础 YOLO 的 90% 以上才通过；若提升只来自 normal、任一恶劣条件下降超过 1 点，或 demosaic/gamma 成为延迟瓶颈，则失败。

## 优点

- 数据集扩展分辨率、类别、实例密度和天气/光照组合。
- 以域差矩阵、条件 AP 和噪声/亮度扰动形成证据链。
- RAW 预训练可复用到多类检测器，不增加推理网络。

## 局限

- 预训练 RAW 由 sRGB 合成，与真实传感器噪声仍有差距。
- 默认下采样会牺牲极小目标，完整切片训练成本高。
- 单一相机来源，跨传感器与 Bayer pattern 泛化不足。

## 评分

- **创新性**：★★★★☆
- **实验充分度**：★★★★★
- **数据集价值**：★★★★★
- **YOLO-Agent 参考价值**：4.9/5
