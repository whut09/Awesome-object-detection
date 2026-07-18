---
title: "Dynamic Sparse R-CNN"
description: "解析 Dynamic Label Assignment 与 Dynamic Proposal Generation 如何分别改造 Sparse R-CNN 的监督分配和初始 proposal。"
tags: ["CVPR 2022", "目标检测", "Sparse R-CNN", "最优传输", "动态提议"]
---

# Dynamic Sparse R-CNN

**官方论文**：[CVF 论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_Dynamic_Sparse_R-CNN_CVPR_2022_paper.html) · [官方 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Hong_Dynamic_Sparse_R-CNN_CVPR_2022_paper.pdf)  
**官方代码**：论文正文、参考文献与 CVF 页面均未给出作者仓库；以下实现细节以论文公开算法和实验设置为证据边界。

## 一句话总结

Dynamic Sparse R-CNN 一边用最优传输把六级 Dynamic Head 的正样本配额逐级放宽，一边从 FPN 内容预测四组 proposal experts 的组合权重，使固定查询变成按图像生成的初始框与特征，最终在 COCO val2017 上以 ResNet-50、36 epochs 得到 47.2 AP。

## 研究背景与问题

Sparse R-CNN 以 300 个可学习 proposal boxes 和 proposal features 做集合预测，每一级 Dynamic Head 都接收 FPN、上一阶段框和实例特征，再输出更精确的类别与框。它避免密集 anchors 和后处理，但存在两个彼此独立的静态假设：训练时 Hungarian matching 规定每个真值只能匹配一个 proposal；推理时所有图像共享同一组训练所得 proposal。前者可能让迭代头获得的正监督过少，后者则只表达训练集平均统计，无法针对当前图像的物体布局调整起点。

## 方法总览

论文沿着“监督如何分配”和“查询从哪里来”两条路径改造基线。Dynamic Label Assignment（DLA）把标签匹配写成 Optimal Transport Assignment：真值与背景是供应者，预测框是需求者，每个预测只接收一个标签单位，但一个真值可供应多个单位。Dynamic Proposal Generation（DPG）则维护多组框/特征专家，用 FPN P2–P5 经过 staircase weight generator 得到逐 proposal 的专家权重，线性组合出第一阶段的图像相关输入；后续阶段仍按 Sparse R-CNN 的级联方式迭代细化。

## 方法详解

### DLA：随级联精度增长而增加供给

匹配代价由分类损失与回归损失加权组成，背景项只计算分类代价。每个真值的基础供给量不是常数，而由 Dynamic-k Estimation 决定：取该真值与预测框的 top-`q` IoU，求和并取整得到 `k`；若全部真值的供给超过 proposal 总数的 80%，统一缩放，以保留至少 20% 负样本。论文默认 `q=8`、六个迭代阶段，并使用 unit increasing strategy，让前级匹配严格、后级在框更可靠时提供更多正样本。因为这是 many-to-one 训练，推理端额外使用阈值 0.7 的 NMS。

### DPG：按图像混合 proposal experts

DPG 设 `Ne=4` 组专家，每组含 300 个四维 proposal boxes 与 300 个 256 维 proposal features。权重网络从 P2 开始，将当前金字塔层与上一层输出拼接，经 `3×3` depth-wise stride-2 卷积逐层汇聚；所得多层特征被整理为 `4C×30×30`，通道求和后展平，依次通过 `900×1500` 和 `1500×(Ne·Np)` 两个全连接层。带温度退火的 softmax 产生专家权重，再分别加权框专家和特征专家。该动态输入只注入第一级，却会通过逐级框与实例特征传递影响全部六级预测。

## 实验与证据

- 数据为 COCO 2017：约 118k train、5k val、80 类；ResNet-50/101 + FPN，AdamW，权重衰减 `1e-4`，batch 16，36 epochs，学习率在第 27、33 epoch 衰减，训练尺度短边 480–800、长边不超过 1333。
- ResNet-50 下，Sparse R-CNN 为 45.0 AP，完整模型为 **47.2 AP / 66.5 AP50 / 51.2 AP75 / 30.1 APS**；ResNet-101 达 47.8 AP。相同 R50 与 36 epochs 下，论文还高于 Conditional DETR 4.2 AP、Anchor DETR 5.1 AP。
- 组件消融中，单加 DPG（含 staircase）从 45.0 提至 45.7 AP；再加普通动态 OTA 为 46.0；加入逐级 unit increasing 后达到 47.2。去掉 staircase 时 DPG 仅为 45.3，证明多尺度权重生成贡献 0.4 AP。
- 不使用 DPG 的 matcher 消融显示：固定 `k=2/3` 均为 45.9 AP，Dynamic-k `q=8` 为 46.1，配合两套损失与 unit increasing 为 46.7；`q` 扫描中 4/5/6/7/8/9 对应 46.7/46.7/46.7/46.4/47.2/46.1 AP。
- 专家数 3、4、5 分别得到 45.4、45.7、45.3 AP。代价方面参数量从 77.8M 增至 81.0M，计算量从 23.28 增至 23.30 GFLOPs，但四张 A100 的训练时间由 29 小时增至 37 小时。

## 对 YOLO-Agent 的启发

这里最值得迁移的不是把 Sparse R-CNN 头照搬进 YOLO，而是把“分配策略”和“输入条件化”做成两个可独立验证的 agent action。**Harness** 应固定同一 YOLO 主干、增强、训练轮数与随机种子。**对照组**设置原始 assigner、仅加入基于 top-`q` IoU 的 Dynamic-k、仅加入由 P3–P5 生成权重的多组先验框/查询专家、二者同时启用四条实验线。**指标**：DLA 组记录 mAP50-95、AP75、APS、每个真值的正样本数分布、冲突分配率和 NMS 前候选重复率；DPG 组额外记录专家熵、不同图像间权重方差、首轮召回率、参数量与训练时延。**失败判断**：关闭 unit schedule 后增益不下降、专家权重长期近似均匀、首轮 recall 没有改善，或总 AP 增长不足 0.5 且训练时长增加超过 20%，均说明论文机制未被有效复现，不进入默认 recipe。

## 优点

- 两个模块分别命中 Sparse R-CNN 的监督稀疏与推理查询静态问题，因果链条清楚。
- AP75 与 APS 的显著提升和逐级 AP 曲线支持“更好匹配、更好初始 proposal”的解释。
- DPG 的额外 FLOPs 极小，并可在不改后续 Dynamic Head 的前提下接入。

## 局限

- DLA 重新引入 NMS，削弱了原始集合预测端到端、无后处理的简洁性。
- 专家权重网络虽计算量小，却明显拉长训练时间，工程瓶颈未被细分解释。
- 证据集中在 COCO 与 Sparse R-CNN，尚不能证明逐级 many-to-one 对 DETR 或单阶段 YOLO 同样稳定。

## 评分

- **创新性：8.5/10**——同时动态化标签供给与 proposal 起点，且两者并非同一技巧的重复包装。
- **证据完整度：8.5/10**——主结果、matcher、`q`、专家数、结构与成本均有消融。
- **YOLO-Agent 适配度：7.5/10**——assigner 易迁移，proposal experts 需重新定义为 YOLO 可消费的先验或查询。
- **综合：8.2/10**——是一篇机制明确、收益扎实，但训练成本和跨架构泛化仍待验证的改进工作。
