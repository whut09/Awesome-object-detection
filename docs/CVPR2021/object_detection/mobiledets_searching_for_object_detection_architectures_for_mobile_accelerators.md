---
title: "[论文解读] MobileDets: Searching for Object Detection Architectures for Mobile Accelerators"
description: "MobileDets 把普通卷积、Tucker convolution 与 inverted bottleneck 放入硬件感知检测 NAS。"
tags: ["CVPR 2021", "移动检测", "MobileDets", "NAS", "硬件感知"]
---

# MobileDets: Searching for Object Detection Architectures for Mobile Accelerators

**会议**: CVPR 2021  
**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Xiong_MobileDets_Searching_for_Object_Detection_Architectures_for_Mobile_Accelerators_CVPR_2021_paper.html)  
**代码**: [TensorFlow Models](https://github.com/tensorflow/models/tree/master/research/object_detection)  
**任务**: 多种移动加速器上的目标检测

## 一句话总结

MobileDets 反驳“移动网络必须全用 depthwise inverted bottleneck”，把 regular convolution、Tucker convolution、fused 与 IBN blocks 一同搜索，并分别用 CPU、EdgeTPU、Hexagon DSP、Jetson 的 latency model 产生设备专属骨干。

## 研究背景与问题

depthwise 在移动 CPU 上高效，却可能无法充分利用 EdgeTPU/DSP 的矩阵阵列；相同 FLOPs 在不同算子和芯片上延迟差异很大。分类 NAS 找到的 backbone 也未必适合 SSDLite 的多尺度检测。论文直接以 detection quality 与目标设备 latency 搜索。

## 方法总览

搜索空间允许 IBN、fused IBN、普通卷积和低秩 Tucker block，控制 kernel、扩张率、输出通道和重复次数。控制器采样候选，检测任务训练提供 reward，设备 lookup/model 估计延迟并施加约束。最终每种硬件得到不同 regular/depthwise 分布，再接 SSDLite head 训练 COCO。

## 方法详解

Tucker convolution 用两个 pointwise projection 包住中间 regular spatial convolution，以低秩通道分解换取加速器友好的密集矩阵计算；fused IBN 则把 expansion 与 depthwise 合并为 regular convolution。搜索结果显示 CPU 更偏好 depthwise，而 EdgeTPU/DSP 会在早中期放入更多 regular/Tucker blocks，这正是设备专属结构差异。


## 实验与证据

在相近延迟下，MobileDets 比 MobileNetV3+SSDLite 高 1.7 mAP；相对 MobileNetV2 基线，在 CPU/EdgeTPU/Hexagon/Jetson 分别高 1.9/3.7/3.4/2.7 mAP。与 MnasFPN 比较时，即使无 feature pyramid，在 EdgeTPU 和 DSP 上仍可更准并最多快 2×。搜索空间消融显示 IBN-only 搜索先提高 1.6 mAP，加入 MobileDet blocks 再增 1.6。

## 对 YOLO-Agent 的启发

Harness 必须为每个目标芯片独立建算子 lookup，并验证预测 latency 与真机 p50/p95 的误差。对照 IBN-only、加入 fused、完整 regular/Tucker 空间，报告 APs、峰值内存和能耗。若 EdgeTPU 搜出的 regular conv 模型迁到 CPU 后变慢不算失败；真正失败是同一目标设备上 lookup 排序失真，或检测 AP 增益来自更高输入分辨率而非架构。

## 优点

- 证明算子优劣必须与加速器绑定。
- 搜索直接面向检测而非分类迁移。
- 四类移动平台都有实测证据。

## 局限

- 每种硬件需重新搜索和维护延迟表。
- SSDLite head 限制了检测框架范围。
- NAS 训练成本高于手工缩放。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★★
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★★
