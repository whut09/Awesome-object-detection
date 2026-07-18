---
title: "Beyond Duality: A Hybrid Framework of Leveraging Shared and Private Features for RGB-Event Object Detection"
description: "解析 SPFD 的频域共享/私有特征分离、TriAdapt Encoder 与 TriInject Decoder。"
tags: ["CVPR 2026", "目标检测", "RGB-Event", "SPFD", "FCFS"]
---

# Beyond Duality: A Hybrid Framework of Leveraging Shared and Private Features for RGB-Event Object Detection

**会议**：CVPR 2026  
**论文**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Beyond_Duality_A_Hybrid_Framework_of_Leveraging_Shared_and_Private_CVPR_2026_paper.html)  
**代码**：[git-KeYw/SPFD](https://github.com/git-KeYw/SPFD)

## 一句话总结

SPFD 不把 RGB 与事件特征直接二选一或粗暴相加，而是由 Frequency-domain Coherence-based Feature Separation（FCFS）拆成 shared、RGB-private、Event-private 三路，再通过 TriAdapt Encoder 和 TriInject Decoder 分阶段使用。

## 研究背景与问题

RGB 在静态纹理和颜色上强，事件流在高速运动、低照边缘上可靠，但两者既有共同目标轮廓，也有各自噪声与私有信息。传统融合往往假设互补即可 concat/attention，未区分冗余共同分量与真正独特分量；当某一模态能量更强时，注意力还可能被低频 RGB 或高频事件响应支配。本文从频谱统计出发，构造可解释的三路特征并在 encoder/decoder 中分别注入。

## 方法总览

RGB 图像与三通道事件帧经 ResNet-50 得到多尺度 `x/y`。FCFS 做 2D FFT，计算 `Sxx/Syy`、`Sxy`、spectral coherence `γ²` 与 strength balance `η`，生成 `Ms/Mr/Me`，逆 FFT 得到 `zs/zr/ze`。TriAdapt Encoder 以 shared feature 为 query，对两种 private feature 做 deformable attention，再由 gate 融合。TriInject Decoder 以可学习 `γr/γe` 把私有特征注入 memory，最终 DETR 查询输出检测框。

## 方法详解

FCFS 用 `γ²=|Sxy|²/(SxxSyy)` 衡量同频率上两模态是否同步，接近 1 代表共享轮廓；`η=sqrt(SxxSyy)/(Sxx+Syy+ε)` 抑制单模态能量垄断。`Ms=sigmoid((γ²η-τ)/T)` 选择高一致且能量平衡的频率，私有 mask 在 `1-Ms` 区域按谱差与能量占比分配。随后 `Zs=Ms⊙(X+Y)/2`、`Zr=Mr⊙X`、`Ze=Me⊙Y`，三路 IFFT 后保留共享语义、RGB 纹理和事件瞬时边缘。

TriAdapt 每层用 `zs` 作 query，对 `zr`、`ze` 分别 MSDA 得到 `Ur/Ue`；两者均值经线性层和 sigmoid 生成空间/通道 gate `G`，输出 `G⊙WrUr+(1-G)⊙WeUe`。TriInject 把每层私有特征投影为 `Vr/Ve`，与 memory 按可学习系数相加：`Vf=M+γrVr+γeVe`，再作为 cross-attention 的 value。encoder 在区域级选择模态，decoder 在语义深度级控制注入，避免私有信息只融合一次后被稀释。

## 实验与证据

主比较使用 DSEC-Det 的 SFNet annotation 与 PKU-DAVIS-SOD。SPFD 在 DSEC-Det 达 56.7 mAP50、34.6 mAP，高于 SFNet 的 51.4/30.4；在 PKU-DAVIS-SOD 达 62.4/32.0，略高于 SFNet 的 59.6/31.9，参数量 78.6M。DSEC 含八类道路目标；PKU 数据含 car、pedestrian、cyclist 的 87 万余框，覆盖正常、运动模糊和低光。

消融使用 DSEC 原始 annotation，绝对值不可与主表直接对照：线性 concat 的 MI-DETR 多模态基线为 47.3 mAP；加入 FCFS 且只用 shared feature 为 47.9；加入 TriAdapt 后 49.0；完整 TriInject 后 49.2，mAP75 从 54.1 提到 56.8。可视化显示 RGB-private 主要保留低频语义纹理，Event-private 强调高频边缘；TriAdapt gate 在低光目标处偏事件，在静态场景偏 RGB。

## 对 YOLO-Agent 的启发

对 YOLO 可先移植 FCFS 到双流 neck，再决定是否使用 query decoder。**Harness** 设 RGB-only、Event-only、直接 concat、FCFS-shared-only、FCFS 三路 gated fusion 五组；按正常/低光/模糊/静态子集记录 mAP50、mAP75、gate 占比、频谱 mask 熵、参数和延迟。三路方案相对 concat 至少提升 1.5 mAP、低光召回提升 ≥3 点，且遮蔽任一模态时性能平滑下降才通过；若 `Ms` 全接近 0/1、gate 长期偏单模态，或混用 annotation 版本，则失败。

## 优点

- shared/private 分解具有明确频谱统计定义。
- encoder 的区域选择与 decoder 的层级注入承担不同功能。
- 两套 RGB-Event 数据及困难条件有定量和可视化证据。

## 局限

- 78.6M 参数明显高于许多实时检测器，未报告完整速度表。
- FCFS 假设两模态精确同步、配准；错位会破坏 coherence。
- 两模态同时退化时，共享/私有估计都会不可靠。

## 评分

- **创新性**：★★★★★
- **实验充分度**：★★★★☆
- **可解释性**：★★★★★
- **YOLO-Agent 参考价值**：4.0/5
