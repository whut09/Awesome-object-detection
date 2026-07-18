---
title: "[论文解读] RAVEN: Radar Adaptive Vision Encoders for Efficient Chirp-wise Object Detection and Segmentation"
description: "利用逐接收通道快时间 SSM、跨天线注意力与慢时间 SSM，直接从 FMCW ADC 流进行可提前退出的检测和分割。"
tags: ["CVPR 2026", "雷达检测", "Mamba", "FMCW", "提前退出"]
---

# RAVEN: Radar Adaptive Vision Encoders for Efficient Chirp-wise Object Detection and Segmentation

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Sen_RAVEN_Radar_Adaptive_Vision_Encoders_for_Efficient_Chirp-wise_Object_Detection_CVPR_2026_paper.html)  
**任务**: 原始雷达 ADC 检测 / 可行驶区域分割

## 一句话总结

RAVEN 不做传统距离—方位—多普勒 FFT，而是为每个 RX 通道配置快时间 Mamba，利用跨天线注意力和可学习 TX 查询恢复虚拟 MIMO 结构，再以逐 chirp 的慢时间 SSM 在线累积证据，并根据隐状态新颖度提前结束一帧计算。

## 研究背景与问题

FMCW MIMO 雷达的角度来自不同接收天线之间确定性的相位差。把所有 RX 先平均、压成标量或用共享 1×1 混合，相当于固定均匀波束形成，会丢掉阵列方向信息；DDM-MIMO 中多个 TX 波形还在线性混合，过早 tokenization 会进一步纠缠虚拟阵列。传统 FFT 网络保留物理结构但计算重，通用序列模型又常忽略阵列物理。RAVEN 的目标是在原始 I/Q 上保持 RX 与 TX 结构，同时让模型在 chirp 尚未收满时即可输出。

## 方法总览

输入为 `Nc×Ns×2Nrx` 的复数 ADC 帧。第一阶段每个 RX 独立使用 Mamba 处理 fast-time 样本，并池化为每 chirp 的二维 token；第二阶段 Cross-Antenna Attention 显式混合接收通道，并用 transmitter queries 提取 DDM 中潜在 TX 成分，扩展为虚拟 MIMO 特征。随后压缩通道，chirp-wise Mamba 沿 slow-time 更新状态。Conv1D 将序列投影到 `T×H×W` 网格，浅层卷积头分别输出检测热图/框偏移与 freespace mask。

## 方法详解

每个 RX 使用独立 SSM，而非共享编码器，目的是保留天线特有相位和幅度。跨天线模块在每个 chirp 上先把 RX token 投影到高维并加入可学习接收标识，再通过注意力形成 steering-like 权重；面向 DDM 的 TX 查询类似可学习匹配滤波器，从接收混合信号中抽取不同发射分量，最后拼成 `Nrx×Ntx` 虚拟阵列表示。

慢时间 SSM 逐 chirp 读取特征，既可并行训练也可流式推理。训练采用 multi-prefix supervision：多个 chirp 前缀共用投影与检测/分割头，并对同一 GT 计算损失，迫使早期状态可用。推理计算当前状态与历史状态的最小余弦距离 `dL`，按 K 个 chirp 分块求均值；首次低于阈值 `τ` 的块即为退出点。论文在统计曲线中选 `τ=0.2`，性能约在 64 chirp 后趋于饱和。

## 实验与证据

- RaDICaL 使用 77GHz、4RX×2TX TDM-MIMO，ADC 尺寸为 `64×192×8`；RADIal 为 12TX×16RX DDM、192 虚拟天线，ADC 尺寸 `256×512×16`，含车辆中心和可行驶区域标注。
- RaDICaL 上 RAVEN 以 `0.053 GMACs、0.347M` 参数达到 `0.997 Dice、0.082 Chamfer`；FFT-RadNet 为 `41.74 GMACs、4.25M、0.996 Dice`。
- RADIal 全帧模型达到 `0.90 mIoU、0.93 F1、0.95 mAP、0.92 mAR`，距离/角度误差 `0.12m/0.10°`，计算量 `1.02 GMACs`、延迟 `20.08ms`。
- 子帧模型为 `0.85 mIoU、0.89 F1、0.88 mAP`，仅 `0.27 GMACs、9.15ms`。验证曲线显示 32→64 chirp 仍有明显收益，此后 mIoU/F1 增益很小。
- 定性结果也暴露失败：杂波场景中早期目标会短暂出现后消失，分割可能始终不稳定，chirp 新颖度曲线会变得噪声化。

## 对 YOLO-Agent 的启发

- Harness 必须对照共享 RX 编码、逐 RX 编码、无 TX query、无跨天线注意力、完整 RAVEN，分别测角度误差、mAP/mIoU、GMACs 和延迟。
- 早退按 32/64/128/256 chirp 固定预算与自适应阈值比较；分高速、远距、密集车辆、低 SNR 和 DDM/TDM 切片。平均精度可接受但远距召回或角度误差恶化时判失败。
- 监控 `τ` 的校准漂移和退出分布；若噪声场景频繁在错误早期状态饱和，要求最低 chirp 下限或回退全帧。
- 目标硬件必须实测流式状态更新，不能用 GMACs 替代延迟；若 SSM/attention 后端不支持导致子帧模型无速度收益，则机制不通过部署门槛。

## 优点

- 架构直接对应 FMCW 阵列、DDM 发射混合和 chirp 时间结构。
- 同时支持检测、分割、并行训练、流式推理与自适应早退。
- 精度、边界、定位误差、参数、算力和实测延迟指标完整。

## 局限

- 早退在杂波和低质量信号上可能过早确认错误状态。
- RaDICaL 标签来自相机检测投影，存在教师误差与模态偏差。
- 跨不同雷达阵列尺寸、调制方式和芯片后端的泛化尚未充分验证。

## 评分

- **创新性**: ★★★★★
- **证据强度**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★☆☆
