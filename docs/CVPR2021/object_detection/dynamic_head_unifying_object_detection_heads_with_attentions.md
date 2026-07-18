---
title: "Dynamic Head: Unifying Object Detection Heads With Attentions"
description: "解析 Dynamic Head 如何沿特征层级、空间位置和通道任务三个维度串联注意力，统一替代多种检测头。"
tags: ["CVPR 2021", "object detection", "attention", "detection head"]
---

# Dynamic Head: Unifying Object Detection Heads With Attentions

- **论文页面**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_Dynamic_Head_Unifying_Object_Detection_Heads_With_Attentions_CVPR_2021_paper.html)
- **官方 PDF**：[CVPR 2021 Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Dynamic_Head_Unifying_Object_Detection_Heads_With_Attentions_CVPR_2021_paper.pdf)
- **官方代码**：[microsoft/DynamicHead](https://github.com/microsoft/DynamicHead)（论文明确给出）

## 一句话总结

Dynamic Head 把 FPN 特征重写为 `level × spatial × channel` 张量，依次通过 Scale-aware、Spatial-aware、Task-aware Attention，在层级上选尺度、空间上做可变形稀疏采样、通道上动态切换任务响应，再直接输出分类与框/中心/关键点预测。

## 研究背景与问题

检测头同时面对小大目标、几何变形和分类/回归等异质任务，但已有工作往往只改 FPN、可变形卷积或某一任务分支。DyHead 的输入是 backbone 产生的多层特征 `{F_l}`；各层在概念上对齐为 `F∈R^{L×S×C}`，中间不做昂贵的全维 self-attention，而把注意力分解到 L、S、C 三个轴；输出仍是标准检测预测。因此它不是泛化的 Transformer 叙述，而是一个针对检测头三个困难逐轴建模的插件。

## 方法总览

**Scale-aware Attention `π_L`** 对空间和通道做全局平均，再经 1×1 卷积与 hard-sigmoid，得到每个特征层的权重；**Spatial-aware Attention `π_S`** 用 deformable convolution 学习稀疏采样偏移 `Δp_k` 和位置重要性 `Δm_k`，并在各层同位置聚合；**Task-aware Attention `π_C`** 对 L×S 全局池化，经两层全连接生成两组线性函数参数，逐通道取二者最大值，形成任务相关的动态激活。多个三注意力 DyHead block 可串联，最终接分类、回归、centerness 或 keypoint 输出层。

## 方法详解

### 1. 层级注意力：选择目标尺度

`π_L` 从每层特征的全局语义摘要计算标量权重，hard-sigmoid 将权重限制在 `[0,1]`。它不同于把所有 FPN 层简单相加：同一 head block 可以根据当前图像内容增强适合目标尺度的层。论文可视化显示学习到的高/低分辨率权重比随目标尺度发生系统变化。

### 2. 空间注意力：稀疏定位判别区域

完整空间注意力对 `H×W` 代价过高，DyHead 直接借助 deformable sampling。偏移和调制系数由中间层特征预测，每个层级在 `p_k+Δp_k` 处采样，再对 L 个层聚合。随着 block 从 2 增加到 6，可视化中的响应从噪声背景逐渐集中到前景轮廓；这也是三模块中单独收益最大的组件。

### 3. 任务注意力：一条分支承担多种预测

`π_C` 为每个通道生成两套 `αF_c+β`，取最大值实现动态分段线性激活，使不同通道按输入内容偏向分类、框回归或中心/关键点任务。单阶段检测器因此可用一条统一 DyHead 分支代替传统独立分类/回归塔；两阶段检测器则在 RoI pooling 前施加 scale/spatial attention，并用 task-aware attention 替换原全连接层。

## 实验与证据

- **数据与协议**：COCO train2017 118K 训练，val2017 5K 做消融，test-dev 做主对比；指标为 AP、AP50、AP75、APS/M/L。消融采用 ATSS+ResNet-50、1× schedule，SGD 初始学习率 0.02。
- **模块消融**：基线 39.0 AP；单独 Scale-aware 为 39.9，Spatial-aware 41.4，Task-aware 40.3；三者完整组合为 42.6 AP、46.4 AP75、56.0 APL。完整增益 3.6 AP，并非单一模块独占。
- **深度/效率**：基线 254.39 GFLOPs、39.0 AP；2 个 block 在少 63.45 GFLOPs 的情况下达 39.5 AP；4/6 个 block 为 42.0/42.6，8 个后饱和至 42.5，10 个降至 42.3。论文最终常用 6 blocks，说明堆叠并非越深越好。
- **跨检测器泛化**：Faster R-CNN 36.4→38.9，RetinaNet 35.7→38.4，ATSS 39.4→42.6，FCOS 38.8→40.0，RepPoints 38.2→39.6 AP，覆盖两阶段/单阶段、anchor-based/anchor-free、box/keypoint 表示。
- **主结果**：test-dev 上 DyHead-ResNeXt-64×4d-101-DCN、2×多尺度训练为 52.3 AP；加入多尺度测试为 54.0 AP、72.1 AP50、59.3 AP75，超过同表 RelationNet++ 的 52.7 AP。无 DCN 的 ResNeXt-101 版本为 47.7 AP。

这些数字也限定了归因边界：39.0→42.6 AP 是 ATSS-R50、1×、val2017 下对完整三轴模块的受控证据；54.0 AP 则同时包含更强 ResNeXt-101-DCN、多尺度训练和多尺度测试。复现时应优先匹配前者的组件增量，再把后者作为系统上限，不能用榜单最高值替代模块消融。

## 对 YOLO-Agent 的启发

DyHead 提供了一个可拆解的 YOLO head 搜索空间：在 PAN/FPN 输出上先学尺度权重，用轻量可变形算子聚焦位置，再以动态通道激活共享分类和回归特征。Agent 应把三个轴分别做开关与预算评估，而不是把“attention head”当作一个不可解释黑盒；尤其要比较共享塔是否真的降低 FLOPs，以及 spatial 模块的增益是否抵消其部署算子成本。

### 专属 Harness：三轴注意力检测头

- **对照组**：A 为原 YOLO 解耦头；B 只加 level/scale attention；C 只加 deformable spatial attention；D 只加 task-aware dynamic activation；E 串联 B+C+D 并共享预测塔；F 为 E 的 2/4/6/8 block 深度扫描。
- **观测指标**：AP、AP75、APS/APL、GFLOPs、TensorRT/ONNX P95 延迟；额外记录每层 scale 权重与目标尺寸的相关性、空间响应落在 GT 框内的能量比例、分类/回归通道梯度夹角。
- **通过标准**：E 相对 A 至少提升 2 AP 和 1 AP75；APS 与 APL 均不得下降；F 应在 4–6 blocks 附近出现收益饱和，且选定配置的真实延迟增幅不超过 15%。
- **失败判断**：只有 C 有收益而 B/D 无贡献、尺度权重与目标大小无相关、共享塔导致分类或回归明显退化、8–10 blocks 仍线性上涨却未核对训练预算，或 GFLOPs 降低但端侧 deformable 算子更慢，均不能宣称复现 DyHead。

复现时应逐 block 导出三类中间量：`π_L` 的层级权重、`Δp/Δm` 的采样位置与调制系数、`π_C` 两个动态分支的激活占比。若最终 AP 上升却这些量与目标尺度、前景区域和任务梯度没有可解释关系，可能只是额外参数带来的容量收益。两阶段检测器还要遵循“scale/spatial 在 RoI pooling 前、task-aware 替换 RoI 后全连接层”的位置，不能把单阶段统一塔的接法直接照搬。

端侧比较必须使用相同输入尺寸与精度模式，并单列可变形算子的实际耗时。

## 优点

- 用 L/S/C 三轴给复杂检测头建立统一且可解释的设计坐标系。
- 对多类检测器均为插件式提升，主干、anchor 形式和目标表示依赖较弱。
- 消融同时报告模块组合、block 深度、GFLOPs 与可视化，证据链完整。

## 局限

- “无额外计算开销”依赖用统一分支替换原重型头；对本来很轻的 YOLO head，deformable sampling 可能增加真实延迟。
- 主结果包含 DCN backbone、多尺度训练和多尺度测试，54.0 AP 不能归因于 DyHead 单一组件。
- 将不同 FPN 层概念性 resize 到统一张量便于推导，但工程实现与硬件访存细节决定最终效率。

## 评分

- **创新性：9/10**——以三个检测特有维度组织注意力，而非直接套通用 self-attention。
- **实验充分性：9/10**——模块、深度、检测器类型、骨干与 SOTA 对比均覆盖。
- **工程可迁移性：8/10**——插件接口清晰，但可变形算子的端侧效率需单独验证。
- **综合评分：8.7/10**。
