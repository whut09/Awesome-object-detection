---
title: "[论文解读] Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism"
description: "Gold-YOLO 用低阶与高阶 Gather-and-Distribute 分支显式聚合全尺度特征，再向检测层注入全局信息。"
tags: ["NeurIPS 2023", "目标检测", "Gold-YOLO", "Gather-and-Distribute", "特征融合"]
---

# Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism

**会议**: NeurIPS 2023  
**论文**: [arXiv](https://arxiv.org/abs/2309.11331)  
**代码**: [Huawei-Noah Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)  
**任务**: 多尺度实时目标检测

## 一句话总结

Gold-YOLO 将 PAN 的相邻层逐级传递改成 Gather-and-Distribute：先把多尺度特征对齐并汇聚为全局表示，再由 Low-GD 和 High-GD 把低阶细节、高阶语义直接注入对应检测层。

## 研究背景与问题

在 FPN/PAN 中，最浅层取得最深层语义必须经过多次 resize、卷积和 concat，远距离尺度信息会在长链路中衰减。把全部特征一次上采样到最高分辨率虽可建立全连接，却带来通道爆炸和内存带宽压力。Gold-YOLO 试图构建一条短路径：先集中 gather，再按目标尺度 distribute，同时把昂贵全局建模限制在低分辨率阶段。

## 方法总览

Low-GD 的 Feature Alignment Module（FAM）用池化或插值把若干主干层对齐到中间尺寸并 concat；Information Fusion Module（IFM）经 RepBlock 混合后切分成多组全局特征；Inject 模块从全局组产生 gate 与 embedding，对本地特征逐点调制。High-GD 在深层重复 gather，加入轻量 Transformer 处理语义关系，再向大目标相关层分发。

## 方法详解

### Low-GD

Low-GD 面向小目标定位和纹理。FAM 不把所有层拉到最大分辨率，而选折中尺度控制计算。IFM 的输出不是一张共享图，而按接收层切片；Inject 使用 sigmoid gate 乘本地响应，再叠加全局 embedding，因此注入不会直接覆盖空间细节。

### High-GD

深层特征尺寸较小，High-GD 才引入 Transformer block 建模长程语义。其前后仍使用 RepBlock 和卷积完成对齐、融合与分发。两条 GD 路径分别服务低阶几何和高阶类别关系，不是机械重复 neck。

### 部署图

RepBlock 在训练期多分支、部署期折叠。FAM、IFM、Inject 使用 resize、pool、conv、sigmoid 和逐元素运算，避免动态采样插件。S/M/L 型号通过宽深系数改变预算。

## 实验与证据

COCO 上论文将 Gold-YOLO 与 YOLOv6、YOLOv7、YOLOv8、DAMO-YOLO 等比较；Gold-YOLO-S 报告 45.4 AP、约 3.5M 参数，Gold-YOLO-L 达 51.8 AP。消融从 Rep-PAN 基线起，分别加入 Low-GD、High-GD，并拆分 FAM、IFM、Inject 与 Transformer 位置；APs/APm/APl 用于验证不同分发阶段，而不只展示总 AP。

## 对 YOLO-Agent 的启发

Harness 可保存每个 Inject gate 的均值、熵、空间热图和尺度间相关性，并比较 PAN、全尺度直接 concat、Low-GD、Low+High-GD 四条数据流。除 APs/APl 与 TensorRT 延迟，还应报告 resize 耗时和峰值显存。若 gate 长期饱和为 0 或 1、Low-GD 没有改善 APs，或对齐操作使显存上涨超过两成但精度增益落入随机波动，就应否决当前注入层选择。

## 优点

- 缩短非相邻尺度之间的信息传播距离。
- 低阶和高阶聚合采用不同计算预算。
- 主要由通用部署算子构成。

## 局限

- 聚合尺寸、切片通道和注入层都是新增超参数。
- 多次 resize/concat 可能受带宽而非算力限制。
- 检测头仍是密集预测并依赖 NMS。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
