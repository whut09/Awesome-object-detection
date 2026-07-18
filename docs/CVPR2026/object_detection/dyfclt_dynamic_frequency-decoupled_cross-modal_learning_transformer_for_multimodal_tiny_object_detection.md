---
title: "DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning Transformer for Multimodal Tiny Object Detection"
description: "解析 DyFCLT 的动态频带跨模态注意力与选择性平滑增强，用于 RGBT 微小目标检测。"
tags: ["CVPR 2026", "目标检测", "RGBT", "微小目标", "DyFCLT"]
---

# DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning Transformer for Multimodal Tiny Object Detection

**会议**：CVPR 2026  
**论文**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DyFCLT_Dynamic_Frequency-Decoupled_Cross-Modal_Learning_Transformer_for_Multimodal_Tiny_Object_CVPR_2026_paper.html)  
代码：未发现论文声明的官方代码。

## 一句话总结

DyFCLT 用可学习 Frequency-Band Decoupling 把 RGB/IR 的 Q、K、V 分成低中高频，在每个频带独立做跨模态相关，再以 Selective Smoothing Enhancement 抑制背景并恢复微小前景细节。

## 研究背景与问题

RGBT 微小目标往往只有十几像素，既需要低频轮廓，也依赖中高频边缘；简单把 RGB 当低频、IR 当高频会忽略两种模态在各频带都含有有效信息。直接全频 cross-attention 容易让强背景纹理和噪声产生错误相关，尤其在无人机远景、树木遮挡与低照场景。论文因此把“频带内交互”和“背景平滑、前景恢复”拆成两个协同模块。

## 方法总览

可见光与红外分别经 ResNet-50 提取三层多尺度特征。Dynamic Frequency-Band Decoupled Cross-Modal Attention（DFCA）先由 `1×1 point-wise + 3×3 depthwise convolution` 生成 Q/K/V，FFT 后用可学习径向边界分为 `B=3` 个频带，在每个 band 内执行 Band-Wise Frequency Attention，再求和、LayerNorm 和投影。Selective Smoothing Enhancement（SSE）包含 Irrelevant Background Smoothing（IBS）与 Foreground-Relevant Enhancement Fusion（FREF）；两路同层结果经 CSPBlock 融合，最后送入 RT-DETR 风格 decoder。

## 方法详解

Frequency-Band Decoupling 以到频谱原点的半径定义 mask，内边界由正增量累积参数化，保证单调有序；三带从 `{0,1/8,1/4,1/2}` 初始化。每带相关权重由 `IFFT(Qb⊙conj(Kb))` 得到，经 3×3 convolution 和 sigmoid 做空间调制，再乘该带 IFFT 后的 V。所有 `Rb` 聚合后形成跨模态增强特征。Q、K、V 同时分带可避免不同频率泄漏到同一相关计算。

IBS 用卷积预测 foreground mask，训练采用 focal Tversky loss；前景直接保留，背景经两个 3×3 卷积先压缩到 `C/r` 再恢复，以削弱高频背景。FREF 根据平滑后的低层特征预测每位置动态 `K×K` 核，pixel-unshuffle 后分成四组，引导高层特征上采样；guided result 与双线性上采样相加，再和低层前景增强特征进入 CSPBlock。它不是统一锐化，而是以干净低层位置线索恢复高层目标细节。

## 实验与证据

RGBT-Tiny 含 93k 帧、120 万目标、七类，81% 小于 16×16；RGBTDronePerson 有 6,125 对图像、70,880 实例，98% 小于 20 像素；另测 aligned FLIR。DyFCLT 在 RGBT-Tiny 达 48.2 AP、69.1 AP50、41.3 tiny AP、63.2 AR，高于多模态 M2D-LIF 的 38.7/54.9/29.0/51.7，也高于单模态 DQ-DETR 43.6 AP。在 RGBTDronePerson 达 61.0 AP50，FLIR 达 84.1 AP50、45.0 AP。

消融基线为 45.4 AP/65.9 AP50/36.6 tiny AP；单加 DFCA 为 46.8/67.5/37.8，单加完整 SSE 为 46.9/67.4/39.7，两者结合但去 FREF 为 47.4/68.2/40.1，完整模型为 48.2/69.1/41.3。只分解 Query 反而降到 45.2 AP，同时分 Q/K 为 47.1，Q/K/V 全分为 48.2。频带数从 B=1 的 46.1 提到三条可学习带的 48.2；三条静态带仅 46.5，四条可学习带回落到 47.0。

## 对 YOLO-Agent 的启发

DFCA 可部署在双流 neck 的高分辨率层，SSE 更适合 P3/P4 的微小目标恢复。**Harness** 比较全频 cross-attention、静态三带、动态三带、动态三带+IBS、完整 DyFCLT；按尺寸报告 APtiny、APextremely-small、AP50、背景误检、频带边界和延迟。完整方案需相对全频基线提升 ≥3 tiny AP、总体 AP ≥1.5，静态到动态至少 +1 AP，延迟增加不超过 30%；若边界贴边、Q-only 未显著更差却复杂度增加，或背景误检上升，则失败。

## 优点

- 动态频带、Q/K/V 分解和 SSE 均有独立消融。
- 在专门微小目标与常规尺度 FLIR 上均展示泛化。
- FREF 以位置相关核恢复目标，不会无差别增强噪声。

## 局限

- 双 ResNet-50 与频域操作使模型达到 85.5M 参数。
- IBS 需要前景 mask 辅助监督，训练实现更复杂。
- FFT 径向频带忽略方向性，配准误差也会破坏相关。

## 评分

- **创新性**：★★★★★
- **实验充分度**：★★★★★
- **微小目标价值**：★★★★★
- **YOLO-Agent 参考价值**：4.4/5
