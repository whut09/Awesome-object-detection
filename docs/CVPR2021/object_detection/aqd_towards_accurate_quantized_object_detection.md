---
title: "AQD: Towards Accurate Quantized Object Detection"
description: "解析 AQD 如何量化卷积、归一化与残差连接，以多层级 BatchNorm 和整数缩放实现低比特、全整数目标检测。"
tags: ["CVPR 2021", "object detection", "quantization", "integer inference"]
---

# AQD: Towards Accurate Quantized Object Detection

- **论文页面**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.html)
- **官方 PDF**：[CVPR 2021 Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.pdf)
- **官方代码**：[aim-uofa/model-quantization](https://github.com/aim-uofa/model-quantization)（论文明确给出）

## 一句话总结

AQD 把激活和权重表示为“整数值 η × 共享尺度 α”，再为卷积/全连接、BatchNorm 与 skip connection 分别推导整数运算，并以每个 FPN level 私有的 Multi-level BatchNorm 稳定 2-bit 检测头量化。

## 研究背景与问题

低比特分类网络常只量化卷积，BN 和残差加法仍调用浮点单元；在检测器中，FPN 多尺度分布和密集预测头又让 2-bit 误差远大于分类任务。AQD 要解决的不是“模型文件更小”这一单点，而是输入激活→量化卷积→归一化→残差相加→下一层激活的整条路径都能用整数算术执行，同时维持 RetinaNet/FCOS 的框分类与回归精度。

## 方法总览

训练阶段以 LSQ 风格的可学习量化区间 `νx,νw` 将非负激活映射到 `[0,2^b-1]`、权重映射到对称区间，并用 STE 通过取整。卷积累加得到整数 `ηconv` 与尺度 `αconv`；BN 的均值、方差和仿射参数被重新整理为可用定点乘加实现的比例；残差两支先对齐尺度，再用整数乘法与 bit-shift 相加。检测头中，每个金字塔层拥有独立 BN 统计和参数，即 **Multi-level BN**。最终 AQD 的卷积、BN、skip connection 均不依赖浮点运算；论文另以 AQD* 表示只量化卷积、保留其他浮点算子的对照。

## 方法详解

### 1. 可学习低比特表示

任意量化张量写成 `x̄=η·α`。激活先截断到 `[0,νx]`，线性映射并取整为 `ηx`；权重截断到 `[-νw,νw]` 后映射为整数 `ηw`。`νx` 和 `νw` 参与训练，使任务损失决定有效动态范围，而不是用固定 min/max。卷积时整数项完成乘累加，尺度项独立传播；因此中间值可在 32-bit accumulator 中计算，再重缩放到下一层的低比特网格。

### 2. Multi-level BatchNorm

FCOS/RetinaNet 的检测头在 P3–P7 上共享卷积，但不同层的激活幅值分布明显不同；共用 BN 会把多峰分布压进同一统计量，量化后误差被放大。AQD 保持卷积权重共享，却为每个 FPN level 私有 BN 的均值、方差、缩放和偏置。推理时这些参数可与前级尺度组合成整数乘法、加法及移位；与 GroupNorm 相比，它既保留层级自适应，又无需运行时浮点均值/方差计算。

### 3. 全整数残差连接

skip connection 的两路输入通常拥有不同尺度 `α1,α2`，不能直接相加。AQD 将尺度比近似为整数分子 `c` 与 2 的幂分母 `2^d`，分别重缩放两支的整数表示，再执行整数加法和 bit-shift，输出重新写回统一 `ηy·αy` 格式。池化、ReLU 等其余层也在这一表示下传递，从而避免整数/浮点执行单元之间的数据交换。

## 实验与证据

- **数据、模型与指标**：COCO 检测验证集，评价 AP、AP50、AP75、APS/M/L；在 RetinaNet 和 FCOS 上使用 ResNet-18/34/50，对比全精度、Group-Net、AQD* 与全整数 AQD，位宽为 2/3/4 bit。
- **FCOS 主结果**：ResNet-18 全精度为 33.9 AP；AQD 4-bit 为 34.1、3-bit 为 33.6、2-bit 为 31.8。ResNet-50 全精度 38.9，AQD 4-bit 38.0、3-bit 37.5、2-bit 35.4。即使极低位宽，仍显著强于 Group-Net 4 bases 的 28.9/32.7 AP。
- **AQD 与 AQD***：FCOS-R50 的 2-bit AQD* 为 36.0 AP，全整数 AQD 为 35.4，代价 0.6 AP；RetinaNet 多组设置中两者差距小于约 0.3 AP，说明消除 BN/skip 浮点并非无损，但额外损失受控。
- **Multi-level BN 消融**：2-bit FCOS-R18 中，共享 SyncBN 26.4 AP、GroupNorm 29.4、Multi-level BN 31.8；R50 分别为 30.3、33.4、35.4。全精度下 Multi-level BN 也达到 R50 的 38.9，说明收益不只是量化校准技巧。
- **敏感组件证据**：2-bit FCOS-R18 只量化 backbone 为 33.8 AP，在此基础上量化 FPN 为 33.2，再量化 heads 降到 32.2；检测头与多尺度特征比 backbone 更敏感，正好解释为何论文重点处理 level-specific normalization。

## 对 YOLO-Agent 的启发

YOLO-Agent 不能把“INT8/INT4 可导出”当作量化成功。应沿着 Conv-BN-activation、CSP/ELAN shortcut、PAN/FPN 多尺度支路逐节点检查整数闭包，并对不同检测尺度保留独立校准统计。量化搜索还应优先保护 neck/head，而不是平均分配位宽，因为 AQD 的组件实验显示主要误差集中在多尺度预测端。

### 专属 Harness：检测头全整数闭包

- **对照组**：A 为 FP32 YOLO；B 仅量化 backbone；C 量化 backbone+neck；D 全部卷积低比特但 BN/shortcut 浮点；E 全整数并共享各尺度 BN 统计；F 全整数且为 P3/P4/P5 分别维护归一化/校准参数。
- **观测指标**：COCO AP/AP75/APS、逐层输出余弦相似度、P3/P4/P5 激活饱和率、导出图中的浮点算子数量、端侧真实延迟与能耗。
- **通过标准**：F 的导出图浮点算子为 0；相对 D 的 AP 下降不超过 0.7，且相对 E 至少恢复 1.0 AP；端侧延迟或能耗必须优于 D，证明整数闭包有实际硬件收益。
- **失败判断**：图中仍存在 BN/Add 浮点节点、共享统计与分层统计无差别、量化 head 造成超过 2 AP 损失，或理论低比特未转化为端侧加速，均判定 AQD 路线未成立。

部署复核应同时导出 AQD 与 AQD* 两张计算图：前者要求 Conv、BN 融合结果、残差重缩放和 Add 全部落在整数域，后者保留浮点归一化作为精度上界。校准报告需按检测层列出量化区间、截断比例和 accumulator 溢出次数；若 P3 小目标层的饱和率远高于 P7，就不能用全网平均误差掩盖。论文没有给出具体芯片实测，因此任何“节能”结论都应以同一编译器、同一频率和同一 batch 的端侧功耗采样补证。

若编译器把低比特张量回退到高精度内核，也应直接判定部署验证失败。

所有回退节点必须逐一列明。

## 优点

- 关注整个算子链的整数化，而非只报告卷积权重位宽。
- Multi-level BN 精确针对检测金字塔的分布差异，消融收益清楚。
- 同时在 RetinaNet 与 FCOS、多骨干、多位宽上验证，结论具有一定普适性。

## 局限

- 论文以算术形式和 AP 为主，缺少具体商用边缘芯片上的端到端延迟、功耗实测。
- 2-bit 全整数模型仍有明显精度下降，尤其 FCOS 对像素级特征质量更敏感。
- 每层级私有 BN 增加部署转换与校准复杂度，现代无 BN 或动态 shape 检测器需重新设计。

## 评分

- **创新性：8.5/10**——把 BN 与残差连接纳入低比特检测的整数闭包。
- **实验充分性：8/10**——精度与组件消融充分，但真实硬件证据不足。
- **工程可迁移性：8.5/10**——对部署图检查和多尺度校准具有直接指导价值。
- **综合评分：8.3/10**。
