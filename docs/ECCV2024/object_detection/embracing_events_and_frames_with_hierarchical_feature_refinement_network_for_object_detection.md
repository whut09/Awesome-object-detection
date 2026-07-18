---
title: "Embracing Events and Frames with Hierarchical Feature Refinement Network for Object Detection"
description: "详解 CAFR 如何通过 BCI 与 TAFR 分层融合事件流和图像帧。"
tags: ["ECCV 2024", "目标检测", "事件相机", "多模态融合"]
---

# Embracing Events and Frames with Hierarchical Feature Refinement Network for Object Detection

**论文**：[官方论文页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11201_ECCV_2024_paper.php)  
**代码**：[官方代码](https://github.com/HuCaoFighting/FRN)

## 一句话总结

Hierarchical Feature Refinement Network 在双 ResNet-50 层级间插入 Cross-Modality Adaptive Feature Refinement（CAFR），先由 Bidirectional Cross-Modality Interaction（BCI）交换事件与帧语义，再用 Two-fold Adaptive Feature Refinement（TAFR）按通道均值和方差对齐模态，最后进入 FPN 与 RetinaNet 检测头。

## 研究背景与问题

RGB 帧提供颜色和纹理，却在高速、低照度、过曝与腐蚀下失真；事件相机动态范围高、响应快，但静止物体和远处小目标几乎不触发事件。简单拼接只叠加两路特征，单向注意力又可能让一种模态支配另一种模态。论文目标是双向交换信息并校正融合后的特征失衡。

## 方法总览

事件流编码为 voxel grid，与 RGB 帧分别进入两条 ResNet-50。每条骨干含 L1–L4，CAFR 插在相邻 block 之间，依次执行 1×1 激活、Mul&Add 混合增强、双向 cross-self-attention、TAFR 统计对齐和通道拼接；多尺度输出经 FPN 产生 P1–P5，再由 RetinaNet 分类与回归子网预测。

## 方法详解

BCI 先把 `Ff`、`Fe` 经 1×1 卷积激活，逐元素相乘形成共享注意并分别残差加回。帧主导分支用事件侧 `Qe,Ke` 对帧侧 `Vf` 做 CrossAtt；事件主导分支则用帧侧 `Qf,Kf` 对事件侧 `Ve` 计算，得到两路受对方引导的特征。

TAFR 对两路 cross-attention 输出做线性投影，再用各自增强特征的通道均值和标准差进行无可学习仿射参数的重标定。校正后的 `Ff'`、`Fe'` 拼接为 `Fo`，既保留各自分布，又抑制跨模态交互后的偏置。

## 实验与证据

PKU-DDD17-Car 含 2,241 个训练帧和 913 个测试帧；DSEC 为 640×480 驾驶数据，评估 car、pedestrian、large vehicle。事件单模态在 PKU/DSEC 为 46.5 mAP50、12.0 mAP，帧单模态为 82.7、25.0，完整 CAFR 达 86.7、38.0。

组件消融中，仅 Mul&Add、CrossAtt、FR 的 mAP 分别为 41.5、42.2、43.6；三者组合且保留双分支达到 46.0 mAP、86.7 mAP50。只留帧主导或事件主导分支为 44.7、42.1。DSEC 上 CAFR 三类 AP 为 49.9/25.8/38.2，平均 38.0，比 EFNet 高 8.0。15 种腐蚀、5 个严重度的 mPC 中，frame-only 为 38.7，CAFR 为 69.5，并高于 DRFuser 67.7 与 EFNet 66.4。

CAFR 的 coarse-to-fine 不是空泛表述。Mul&Add 先用同位置事件—帧乘积形成联合显著性，保留各自残差；CrossAtt 再用另一模态的 query/key 选择本模态 value；TAFR 最后将跨模态语义投影后的特征，以原增强分支的均值和方差重定标。三步分别处理显著区域、长程关系和分布偏移，若改变顺序，统计量的来源也会改变，因此复现应遵循论文数据流。

检测头沿用 RetinaNet：FPN 输出多层特征，分类与框回归各由四个 3×3、256 通道卷积组成，锚框数 `A=9`。这意味着论文提升是在成熟 anchor-based head 上取得，并没有把收益混入新的标签分配或检测损失。迁移到 anchor-free YOLO 时，CAFR 之后的特征幅值可能影响 objectness 与动态分配，应重新检查正样本数和分类/定位梯度比例。

腐蚀集只修改 frame 图像，event 流保持干净，因而 69.5 mPC 证明的是事件模态在 RGB 退化时提供备份，而不是两种传感器同时失真的鲁棒性。15 类包含 gaussian、shot、impulse noise，四种 blur，snow/frost/fog/brightness，以及 contrast、elastic、pixelate、JPEG；每类五档严重度。RENet 在 blur 上很强但 weather 仅 29.9，CAFR 四组为 73.6/57.0/70.6/76.7，表现更均衡。

PKU-DDD17-Car 只有车辆类别且规模较小，DSEC 则有三类并需单应变换对齐 RGB 与事件视角。CAFR 在两个基准均提升，支持泛化性；但 pedestrian AP 25.8 仍明显低于 car 49.9，说明小而稀疏的事件目标仍是主要难点。分距离、速度和事件密度报告将比单一平均 mAP 更能判断模块价值。

表 2 的逐项组合还能判断模块交互：Mul&Add+CrossAtt 只有 42.4 mAP，Mul&Add+FR 为 43.2，CrossAtt+FR 达 45.3，说明统计 refinement 与关系注意的组合最强；加入全部三项才到 46.0。也就是说最便宜的乘加显著性不是独立核心，它更像为后两步提供稳定初值。若 YOLO-Agent 必须裁剪模块，应优先保留 CrossAtt+FR，而不是只留下视觉上直观的乘法注意。

论文的鲁棒性协议在 clean data 训练、corrupted frame 测试，避免模型见过腐蚀增强后取得虚假优势。CAFR 在 noise 73.6、weather 70.6、digital 76.7 均为表中最佳，但 blur 57.0 低于 RENet 的 72.3；平均 mPC 仍最高。部署于高速运动模糊场景时，不能只依据平均数，可能需要把 RENet 式时空机制与 CAFR 结合，或针对 blur 单独训练校准。

## 对 YOLO-Agent 的启发

多模态 neck 应把“建立跨模态关系”和“分布再平衡”拆成两个可搜索阶段。YOLO-Agent 可将 BCI 注册为交互算子、TAFR 注册为低参数校准算子，并根据事件稀疏度和帧曝光质量选择插入层级；搜索反馈必须包含腐蚀鲁棒性，而非只看正常光 mAP。

### 论文专属 Harness

- **对照组**：同一 detector 比较 frame-only、event-only、concat、单向 CrossAtt、BCI、BCI+TAFR，并分别保留两个单分支。
- **观测指标**：干净集 mAP、DSEC 三类别 AP、15 类腐蚀 mPC、远距小目标召回、事件稀疏度分桶 AP 与延迟。
- **通过阈值**：完整双分支相对 concat 在两数据集均提升至少 2 mAP，腐蚀 mPC 提升至少 4 点，静态和运动目标召回不得下降超过 1 点。
- **失败判断**：若 TAFR 后方差塌缩，或延迟增加超过 20% 而 mPC 增益不足 2 点，则回退到浅层单次融合。

事件表示同样需要锁定。论文采用 voxel grid 把异步事件聚合成 CNN 可读张量，聚合时间窗决定运动边缘密度；时间窗过长会叠加轨迹，过短则远处目标事件不足。验证 CAFR 时应将 voxel bin 数、曝光同步、单应对齐误差与 RGB 帧率写入配置，并分别测试正常同步与时间偏移，否则融合模块可能只是在补偿数据预处理差异。

TAFR 中 `ε=1e-5` 防止方差除零。事件极稀疏时某些通道可能近常数，归一化会放大微小噪声；因此应额外统计低方差通道比例，并尝试方差下限或门控跳过，确认增益不是数值放大造成。

## 优点

- BCI 与 TAFR 分工明确，双分支和单模态消融完整。
- 同时覆盖两种分辨率数据，并专门验证 15 类腐蚀鲁棒性。

## 局限

- 双 ResNet-50 与多层 CAFR 成本不低，完整实时吞吐报告不足。
- 对齐依赖标定，统计校准也难处理局部时间错位。

## 评分

- **问题重要性**：★★★★☆
- **方法清晰度**：★★★★☆
- **实验证据**：★★★★★
- **工程可迁移性**：★★★☆☆
- **YOLO-Agent 参考价值**：★★★★☆
