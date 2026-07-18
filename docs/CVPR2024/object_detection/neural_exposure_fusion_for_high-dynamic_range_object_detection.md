---
title: "Neural Exposure Fusion for High-Dynamic Range Object Detection"
description: "原创中文详解面向 HDR 检测的 Local Cross-Attention 多曝光特征融合、可微 ISP 与车载实验。"
tags: ["CVPR 2024", "目标检测", "HDR", "多曝光融合"]
---

# Neural Exposure Fusion for High-Dynamic Range Object Detection

**论文**：[CVPR 官方页面](https://openaccess.thecvf.com/content/CVPR2024/html/Onzon_Neural_Exposure_Fusion_for_High-Dynamic_Range_Object_Detection_CVPR_2024_paper.html)  
**官方代码**：未发现论文声明的官方代码。

## 一句话总结

论文绕过为人眼显示而设计的 HDR 图像合成，让多张 LDR 曝光分别经过可微 ISP 与共享特征提取器，再由 Local Cross-Attention 在每个空间位置选择最有检测价值的曝光特征。

## 研究背景与问题

车载相机面对隧道口、逆光、夜间灯牌时，场景动态范围可远超单次 CMOS 曝光。传统多曝光传感器先在硬件 ISP 中做像素级 ExpoFusion，再把一张 tone-mapped HDR 图交给检测器；融合目标偏向平滑视觉效果，可能抹掉噪声较大但对小目标、阴影车辆有用的局部细节。本文把融合监督从“看起来好”改成“检测损失更低”。

## 方法总览

`R1...Rn` 多曝光 RAW 分别进入 differentiable ISP，随后通过共享权重的 feature extractor 得到 `y_j`。Local Cross-Attention 对同一 `(r,c)` 位置跨曝光计算权重 `α_j,r,c`，加权和形成单一 fused feature map，再接 Faster R-CNN 的 RPN、RoI pooling 与检测头。曝光控制、ISP、特征提取、融合和检测损失可端到端联训。

## 方法详解

**Local Cross-Attention Fusion（LCA）**在每个像素位置取 `n×d` 的曝光特征矩阵。key 为 `K=yW^K`，query 是全图共享的可学习矩阵 `Q∈R^{q×d}`；Q 的每一行先做 L2 归一化。`QK^T/√d` 经过同时在 query 轴和 exposure 轴归一化的二维 softmax，得到 `q×n` 权重，再沿 query 轴求和为每个曝光的标量，最后按曝光维加权原始 value 特征。它不投影 value，也不在空间全局做注意力，因此可用 1×1 卷积高效实现。

每条曝光支路先执行可微 ISP：contrast stretching、demosaicing、resize、color correction、低频去噪、sharpening 与 contrast enhancing。默认检测器是自定义 28-layer ResNet 加 Faster R-CNN，LCA 放在 Conv4 后；论文也测试 Conv1/2/3、max pooling、1×1/3×3 convolution fusion 和 second-stage late fusion。

训练集来自搭载 Sony IMX490 的车辆，含 18,790 个 HDR 样本；从每张 24-bit 解压 HDR 图模拟 `n` 张 12-bit LDR RAW。测试集是 1,996 对连续 HDR 帧，第二帧人工标框，覆盖 sunny、cloud/rain、backlight、tunnel、dusk、night，并用 7 个 exposure shifts 模拟难度。

## 实验与证据

指标是 IoU 0.5 的检测 mAP。LDR auto-exposure 基线 Onzon et al. 为 41.1，Raw HDR 为 41.5，Debevec-Malik 41.7，Deep HDR 41.9，PPNE 43.1；默认三曝光、Conv4 后 LCA 达到 **44.2 mAP**。它比 max pooling 43.2、late fusion 43.0、1×1 conv 38.5、3×3 conv 37.9 更好，尤其 traffic sign AP 为 56.3。

内部消融中，普通 1D softmax 为 43.2；不归一化 Q 为 42.3；不乘可学习 `W^K` 为 44.1；完整 LCA 为 44.2。两曝光为 43.7，四曝光为 44.4，作者选择三曝光平衡成本。融合越晚精度越高：Conv1/2/3/4 为 43.3/43.9/44.1/44.2；Jetson Orin AGX 上速度相应为 123/88/70/57 FPS。Conv4 LCA 仅增加 16,896 参数，111 GFLOPs，57 FPS。

## 对 YOLO-Agent 的启发

若 YOLO-Agent 接入多曝光摄像头，应在 neck 之前保留曝光支路，而不是先把 RAW 合成一张图；LCA 可替换为逐尺度、逐像素的曝光门控，并共享 backbone 权重。**Harness**：对照组设置单曝光自动曝光、Raw HDR 像素融合、max pooling、1×1 conv、LCA@P3/P4/P5；观测全局 mAP50-95、暗区/饱和区 AP、小目标 AP、曝光权重熵、端到端 FPS 与功耗。LCA 相对最强像素融合至少提升 1.5 mAP、暗区召回提升 5%、目标设备保持 ≥30 FPS 才通过；若注意力长期塌缩到单曝光、四曝光收益低于 0.3 AP 却成本翻倍，或 ISP 梯度不稳定，则失败。

## 优点

- 融合目标直接由检测损失决定，避免显示质量与机器视觉目标错位。
- 注意力局部化，额外参数和计算量远低于全局 Transformer。
- 同时给出精度、模块消融、融合位置与 Jetson 运行时证据。

## 局限

- 数据集为私有车载 HDR 数据，外部可复现性有限。
- 多曝光采集仍可能遭遇运动、同步和传感器硬件约束。
- 论文以 Faster R-CNN 验证，迁移到多尺度 YOLO neck 需重新选择融合层。

## 评分

- **创新性**：9/10
- **实验充分度**：8/10
- **工程可迁移性**：7/10
- **YOLO-Agent 参考价值**：8/10
