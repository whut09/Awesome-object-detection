---
title: "[论文解读] SFOD: Spiking Fusion Object Detector"
description: "SFOD 设计 Spiking Fusion Module，在 SNN 中首次完成事件多尺度特征融合，并系统分析解码与损失。"
tags: ["CVPR 2024", "事件相机", "SFOD", "Spiking Fusion Module", "SNN"]
---

# SFOD: Spiking Fusion Object Detector

**会议**: CVPR 2024  
**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Fan_SFOD_Spiking_Fusion_Object_Detector_CVPR_2024_paper.html)  
**代码**: [yimeng-fan/SFOD](https://github.com/yimeng-fan/SFOD)  
**任务**: 事件相机脉冲目标检测

## 一句话总结

SFOD 把多时间步事件送入脉冲 backbone，再用 Spiking Fusion Module 对齐、融合不同尺度的 spike feature maps，配合在 NCAR 预训练中筛选出的 spiking decoding strategy 与 loss，提升 GEN1 检测而保持低 firing rate。

## 研究背景与问题

SNN 天然适合异步事件，但早期模型多关注单尺度分类；检测需要深层语义与浅层定位融合，普通 ANN 的 concat/加法会引入连续值 MAC 或破坏时间维脉冲。解码时选择最后膜电位、时间平均或累计 spike，也会显著影响梯度与框精度。

## 方法总览

事件被体素化为 $T$ 个时间片，脉冲卷积逐步提取层级特征。Spiking Fusion Module 对各尺度做脉冲兼容的上/下采样和通道映射，在时间维保持 spike tensor，再将融合结果送入 detection head。backbone 先在 NCAR 分类预训练，论文比较多种 decoder 和分类损失后选择最佳组合，再迁移 GEN1。

## 方法详解

Spiking Fusion Module 在每个时间步分别对深浅特征做尺寸与通道对齐，融合结果仍以 spike tensor 送往后续神经元；它避免把时间维先求平均成连续 feature map。论文在 NCAR 预训练中比较 membrane-potential decoding、spike-count decoding 及不同 classification losses，选定组合后才迁移到 GEN1 detection head。


## 实验与证据

NCAR 上 SNN 分类达到 93.7% accuracy；GEN1 上 SFOD 为 32.1% mAP，超过此前 SNN detector。主图同时展示 mAP 与 firing rate/模型尺寸。消融聚焦 Spiking Fusion Module、有无多尺度、不同 decoding strategy、loss function 和时间步，证明融合与训练选择都影响最终检测。

## 对 YOLO-Agent 的启发

应记录每尺度 firing rate、融合前后 spike density、mAP 与估算 SyOPs；对照单尺度、ANN 式连续融合、Spiking Fusion Module。decoder 要固定相同时间步比较 last membrane、mean membrane、spike count。若融合后浅层放电暴涨导致能耗抵消，或 NCAR 预训练提升分类却不提升 GEN1 APs，就判定迁移链路失败。

## 优点

- 专门解决 SNN 多尺度融合缺口。
- 分类预训练中的 decoder/loss 有系统实验。
- 同时报精度、放电率和模型大小。

## 局限

- 32.1 mAP 与强 ANN 事件 detector 仍有差距。
- 体素化时间窗影响低延迟优势。
- 能耗需要神经形态硬件实测确认。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
