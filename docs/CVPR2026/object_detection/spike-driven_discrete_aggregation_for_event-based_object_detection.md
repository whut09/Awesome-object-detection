---
title: "Spike-driven Discrete Aggregation for Event-based Object Detection"
description: "SDA 用门控循环脉冲神经元只聚合触发放电的事件，并以 MTF 的粗时间尺度膜电位软掩码补充长程上下文。"
tags: ["CVPR 2026", "目标检测", "事件相机", "脉冲神经网络", "SDA", "多时间尺度"]
---

# Spike-driven Discrete Aggregation for Event-based Object Detection

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Spike-driven_Discrete_Aggregation_for_Event-based_Object_Detection_CVPR_2026_paper.html)  
**代码**: 未发现论文声明的官方代码  
**任务**: 事件流目标检测与事件表示学习

## 一句话总结

论文提出 Spiking Discrete Aggregation，让每个空间坐标与极性对应的门控循环脉冲神经元只在事件推动膜电位越阈值时才纳入聚合，再由 Multi-Timescale Fusion 用 SAS 粗粒度时间段生成软掩码调节 SDA 的输入电流。

## 研究背景与问题

事件相机以微秒级时间分辨率记录亮度变化，适合高速和极端光照，但下游检测器需要规则稠密张量。Event Count、Time Surface、Voxel Cube 及既有 SNN 表示通常在采样区间内连续累积全部事件；热噪声、重复触发和与目标无关的变化也会被写入表示。论文认为瓶颈不只在 backbone，而在“采样后究竟聚合哪些事件”，因此把连续聚合改写为可微的离散选择。

## 方法总览

原始事件先按较细间隔 Δta 切成 Event Count slice，形成 Ta×2×H×W 输入。SDA 为每个 (x,y,p) 维护脉冲状态与膜电位，只有触发 spike 的事件膜电位才累加到输出。基础 LIF 被扩展为 Gated Recurrent Spiking Neuron（GRSN），以 reset gate、forget gate 和上一时刻 spike 建模时间依赖。SDA-MTF 另设 Spiking Continuous Aggregation with Adaptive Sampling（SAS）分支，在相邻放电时刻之间构造粗时间尺度表示，并把其膜电位和经 sigmoid 得到的 mask 注入 SDA 更新，最终表示进入全脉冲 YOLOX-M/S。

## 方法详解

在 SDA 中，事件是否被选中与神经元放电状态绑定：同一坐标和极性的事件更新膜电位，只有 s_t=1 的时刻，其 u_t 才进入 gSDA。代理梯度和 BPTT 使阈值操作可联合训练。为避免固定 LIF 衰减无法适应稀疏与噪声，GRSN 用 reset gate 控制旧膜电位保留，用 forget gate 控制旧输入电流，并将当前输入和上一 spike 通过 3×3 卷积写入候选电流；当 reset 接近零时，累计噪声可以被主动清空。

MTF 的 SAS 分支在相邻 spike 时刻之间连续聚合，时间段 Δtm 比 SDA 的 Δta 更粗。简单把两种表示相加并不稳定，因此作者把 SAS 区间膜电位求和后过 sigmoid，得到位置相关软掩码 Mj，再乘到 GRSN 当前候选电流。这样粗尺度分支不直接覆盖细尺度离散结果，而是提示哪些时间区域值得保留。论文配置 Δta=20ms、Δtm=60ms，SNN backbone、FPN 和 head 均使用 P-LIF 与 SEW-Residual，共运行 T=3 个 timestep。

SDA 所谓“离散”不是把时间切片改成硬索引，而是让切片内部的累积由放电事件门控。若一个事件只抬高膜电位却未触发 spike，它不会写入最终表示；随后 reset gate 又可以丢弃遗留状态。SAS 则保留连续累积的粗尺度视角，软掩码把它变成 SDA 的条件信号。这样既避免把两个分支简单相加导致重复噪声，又使长时间段内的运动趋势能影响细粒度事件选择，正是完整 SDA-MTF 优于无 mask 版本的原因。

## 实验与证据

数据集包括 Gen1（304×240）、高分辨率 1Mpx（1280×720、14.65 小时）和 N-Caltech101（8,709 条事件流、101 类）。Gen1 上 DASNN-MTF(M) 达到 43.4 mAP50:95/73.1 mAP50，超过全脉冲 SpikeYOLO 的 38.9/67.2；8.9M 小模型也有 40.5 mAP50:95。1Mpx 上首次报告的全脉冲结果为 30.4 mAP50:95，N-Caltech101 为 40.1/61.9。

消融在 spiking YOLOX-M 上进行：Event Count、Time Surface、Voxel Cube 分别为 32.3、37.3、32.6 mAP50:95；同为 20ms 的连续聚合基线为 41.2，SDA 为 42.8，排除了仅靠更高时间分辨率的解释。SDA-MTF 去掉 GRSN 后降到 40.7；有 GRSN 但无 mask 为 42.8，完整方案为 43.4。接到 ANN 的 YOLOX-S、PVT-S、RVT-B 后，SDA 分别带来 +8.4、+3.0、+1.2 mAP50:95。能耗估算中 DASNN 为 15.66mJ，对比 YOLOX-S 的 59.35mJ，约降低 3.79 倍；高速 Lv4 上 MTF 从 42.5 提升到 44.0。

训练采用 Adam、初始学习率 0.002 和 cosine decay。Gen1 每个标注使用之前 240ms 的事件，1Mpx 沿用 RVT 提供的三段输入。噪声实验同时加入自然噪声与随机噪声，并以噪声事件数占原事件数的比例分级；当比例从 0.1 增到 0.3，SDA 与 SCA 的性能降幅差距继续扩大。可视化也显示在运动模糊增强时，SDA 后的 Event Count 轮廓更清楚，SDA-MTF 热图仍保持目标形状。小模型仅 8.9M 参数却超过 23.1M 的 SpikeYOLO，说明收益并非由模型容量堆叠获得。

速度分桶进一步区分 MTF 的价值：SDA 在 Lv1/Lv2/Lv3/Lv4 为 23.9/47.8/47.9/42.5，加入 MTF 后为 24.3/48.1/48.8/44.0，最快一档增幅最大。ANN 兼容实验中，YOLOX-S 使用 Event Count 只有 37.4，SDA 达 45.8，而 SDA-MTF 为 43.3；PVT-S 也表现为 SDA 略高于 MTF。可见多时间尺度并非对所有架构都优于单 SDA，它主要在全脉冲模型和已有循环时序结构中提供补充，这一点应在工程选择时保留。

能耗表把 Embedding、Backbone、FPN、Head 分开估算。DASNN-MTF 的总能耗为 16.08mJ，略高于只用 SDA 的 15.66mJ，但仍约为 YOLOX-S 的四分之一；额外成本主要来自循环连接与 soft mask 按 MAC 计费。与 EAS-SNN 的 15.91mJ 相比，单 SDA 更低、完整 MTF 略高，说明 0.6 mAP 的提升不是免费的。部署评估应同时报告精度、放电稀疏度和真实硬件功耗，不能只引用理论 AC 优势。

在一百万像素数据上仍能训练全脉冲检测器，也说明离散聚合并非只适用于低分辨率事件传感器；但高分辨率下仅达到三成左右 mAP，距离同数据上的 ANN 仍有明显差距。

## 对 YOLO-Agent 的启发

该文适合启发事件输入前端，而非替换普通 RGB 检测头。Harness 对照组必须包含相同 Δta 的 Event Count、SCA、SDA、SDA+SAS 直接相加、SDA-MTF，并固定 YOLOX-S 主体。观测指标记录 Gen1 mAP50:95、Lv1–Lv4 分速率 AP、不同噪声比例的性能下降、每秒事件吞吐、AC/MAC 能耗估算和实际 GPU 延迟。通过阈值要求相对 SCA 至少 +1.0 mAP、高速 Lv4 至少 +1.0，噪声比 0.3 时降幅减少 15%，同时表示前端延迟不增加超过 25%。失败判断是增益仅来自更密切片，或 ANN 中 MTF 反而稳定低于 SDA，此时应拒绝复杂融合。

## 优点

- 把事件选择与 spike firing 合成一个可训练操作，语义上比无差别累积明确。
- 同时验证全脉冲、CNN、Transformer 和带循环 Transformer 的兼容性。
- 提供运动速度、噪声、可视化和能耗证据，不只比较单一 mAP。

## 局限

- 当前硬件仍需先用 Event Count 近似微秒事件，未实现真正逐事件端到端处理。
- 论文未声明官方代码，GRSN、SAS 和代理梯度细节的复现风险较高。
- 能耗来自算子类型估算，不能替代具体神经形态芯片上的实测功耗。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★★
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
