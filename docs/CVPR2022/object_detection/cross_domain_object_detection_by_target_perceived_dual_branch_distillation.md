---
title: "[论文解读] Cross Domain Object Detection by Target-Perceived Dual Branch Distillation"
description: "TDD 以 SA/TL 双分支、Target Proposal Perceiver 和三阶段自蒸馏完成无标注目标域适配。"
tags: ["CVPR 2022", "目标检测", "域适应", "自蒸馏"]
---

# Cross Domain Object Detection by Target-Perceived Dual Branch Distillation

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/CVPR2022/html/He_Cross_Domain_Object_Detection_by_Target-Perceived_Dual_Branch_Distillation_CVPR_2022_paper.html)  
**代码**：论文未提供官方代码  
**发表**：CVPR 2022

## 一句话总结

Target-perceived Dual-branch Distillation（TDD）用 Source-Adaptive 与 Target-Like 两条 ROI 分支保留不同域知识，再让 MHPCA 从 TL proposal 中为 SA proposal 检索目标域线索，并通过 JDP、CDD、DTR 三阶段教师—学生自蒸馏生成可靠伪框。

## 研究背景与问题

无监督跨域检测同时面对外观域差异与目标域无框标注。只做特征对齐容易抹掉域特有的判别线索；只用 Mean Teacher 又会让初始偏置通过伪标签循环累积。TDD 先把源图经 Fourier style transfer 转为带原标签的 target-like 图像，再要求网络既保留源域对象知识，也学习更接近目标域的外观。真正的目标图像同时进入两条分支，由两种域视角共同产生伪监督。

## 方法总览

网络共享 backbone、FPN、RPN 与 proposal extractor，之后分为 Source-Adaptive（SA）和 Target-Like（TL）检测分支。源图监督 SA，风格迁移图监督 TL；目标图 proposal 同时进入两支。Target Proposal Perceiver 只增强 SA：它将 TL proposal 当作上下文，以带几何权重的 Multi-Head Proposal Cross Attention（MHPCA）补足 SA 对目标域的感知。训练依次执行 Joint-Domain Pretraining、Cross-Domain Distillation 和 Dual-Teacher Refinement；推理只保留 SA 教师分支。

源图到 target-like 图的转换使用 Fourier 域风格迁移，保留物体布局和原框标注，只替换更接近目标域的低频外观。共享提议器因此同时接触 source、target-like 和真实 target 图，学习较通用的候选空间；分开的 ROI head 则防止两种已标注域被强行压到完全相同的分类与回归参数中。目标图进入两支不是简单集成，而是让 TL 提供“更像目标域”的上下文、SA 保留更充分的源标注知识。

## 方法详解

### 1. Target Proposal Perceiver

目标图 `X_t` 经共享提议器得到 `P_t`，两支 FC 编码为 `Φ_SA=F_SA(P_t)`、`Φ_TL=F_TL(P_t)`。MHPCA 以 `Φ_SA` 为 Query，以 `Φ_TL` 为 Key/Value，得到 TL 上下文 `H_TL=W(Q(Φ_SA),K(Φ_TL))V(Φ_TL)`。普通点积相似度为 `A_ij=Q_iK_j^T/σ`，`σ` 是 query 维度平方根；论文再引入 proposal 几何权重 `U_ij`：

`W_ij = U_ij exp(A_ij) / Σ_k U_ik exp(A_ik)`。

多个 attention head 的上下文经 FC 汇总并残差加入 SA 特征，形成 target-perceived `Ψ_SA`。Faster R-CNN 的 ROI head 有两层 FC，因此论文迭代应用两次 MHPCA。注意力是非对称的：TL 指导缺少目标知识的 SA，而不是两边互相改写。

几何权重 `U_ij` 使注意力不只看语义相似度，还考虑两个 proposal 的相对位置和尺度。论文可视化中，一个 SA 分支的 rider proposal 会从 TL 分支检索 motorcycle 和 person 等相关候选作为前三条线索，说明模块学习的是同一场景内的对象上下文，而不是对所有 proposal 做无差别平均。对称 cross-attention 会反过来扰动本已接近目标外观的 TL 分支，因此实验反而较差。

### 2. Dual-Branch Self Distillation

JDP 用源图 `(X_s,Y_s)` 和 target-like 图 `(X_tl,Y_tl)` 初始化网络：`L_JDP=L_RPN^(S+TL)+L_SA^(S)+L_TL^(TL)`。共享 RPN 同时吃两域标注，SA/TL ROI 分类与回归分别由对应域监督。

CDD 固定预训练教师，对目标图两支预测分别做 NMS 和置信度阈值，得到 `Ỹ_t^SA、Ỹ_t^TL`。学生以强增强目标图训练，教师以弱增强产生伪标签；损失 `L_CDD=L_RPN^(T)+L_SA^(T)+L_TL^(T)`，RPN 同时使用两支伪框。DTR 再用 EMA 更新教师：`Θ_teacher←αΘ_teacher+(1-α)Θ_student`。MHPCA 在 CDD 才开始训练，DTR 时随完整学生一起进入教师。论文默认 `α=0.9996`。

CDD 期间仍继续用 JDP 的源图和 target-like 图损失训练学生，防止两条分支在不稳定目标伪标签上遗忘已标注知识。教师先固定，是为了让随机初始化的 Target Proposal Perceiver 有稳定监督；待学生中的 Perceiver 训练成熟后才进入 DTR，用整网 EMA 逐步更新教师。最终 SA 教师已吸收 TL 与目标图信息，推理时无需 style transfer、TL head 或 Perceiver，额外结构只服务训练。

## 实验与证据

实验涵盖 Cityscapes→Foggy Cityscapes（C→F）、SIM10K→Cityscapes（S→C）、KITTI→Cityscapes（K→C）和 Cityscapes→BDD100K daytime（C→B）。S→C 只评 car，SIM10K 有 10,000 张图；K→C 也只评 car；C→B 使用 BDD100K daytime 的 36,278 张训练图和 5,258 张验证图。主要基线包括 DA-Faster、SWDA、GPA、MeGA、SFA、UBT/Mean-Teacher 类方法。

C→F 使用 Cityscapes 2,975 张训练图作为源域、Foggy Cityscapes 无标注训练集作为目标域，并在 500 张 Foggy 验证图上评估。论文使用 Faster R-CNN 的 VGG16 和 ResNet-50 版本，交叉域蒸馏阶段使用 focal loss 与强弱增强，伪标签阈值最终设为 0.7，DTR 的 EMA 为 0.9996。这些固定设置随后用于不同迁移场景，而非每个数据集单独设计一套自训练规则。

ResNet-50 的 TDD 在 S→C 达 63.3 car AP，高于 SFA 52.6、MKT 50.2 和 source-only 42.8；K→C 达 49.8，高于 GPA 47.9 和 source-only 32.5。C→B 达 43.9 mAP，source-only 为 34.1。消融中，双分支但无 Perceiver 在 C→F/S→C/C→B 为 48.3/62.6/42.2，加入 MHPCA 后为 49.2/63.3/43.9；self-attention 只有 46.8/61.0/40.6，对称 cross-attention 也低于非对称方案。

双分支结构消融从只用 source 的 34.8/48.3/34.3 开始；单分支加入 target 后为 41.2/59.0/38.9，再加入 target-like 为 47.4/61.1/39.4，真正双分支达到 48.3/62.6/42.2。这个顺序说明三种域数据都有效，但把 source 与 target-like 的域特性分别保存在两条 head 中，尤其对类别更多、场景更复杂的 C→B 收益最大。

三阶段训练的增益更明显：JDP 为 37.4/56.7/37.5，加入 CDD 为 44.1/62.1/42.7，加入 DTR 达 49.2/63.3/43.9。EMA 太快会破坏教师：`α=0.96` 仅 39.3/59.1/28.7，`0.996` 为 48.4/63.6/41.5，`0.9996` 综合最佳。

## 对 YOLO-Agent 的启发

YOLO 没有 ROI proposal，可把 MHPCA 接在正样本查询或 top-k dense tokens 上：源分支负责原图，TL 分支负责 Fourier/风格迁移图，共享 backbone 与 neck，只保留轻量双 head；目标图的 TL tokens 作为 Key/Value 增强 SA tokens。对照组应为 source-only、单分支 Mean Teacher、双分支无 cross-attention、对称 attention、非对称 TDD。报告目标域 mAP、每类伪标签 precision/recall、两分支伪框 IoU 一致率及额外显存。若 CDD 后伪标签 precision 低于 70%，DTR 连续两轮使 mAP 下降，或双 head 显存增加超过 25% 而目标 mAP 增益不足 2 点，则停止 EMA 精炼并退回单教师高阈值方案。

## 优点

- 用共享提议器与域特异分支同时处理域不变性和域判别性。
- MHPCA 的方向、几何权重与 proposal 语义都针对跨域检测设计。
- 四种迁移设置和完整三阶段消融证明收益不是单一数据集现象。

## 局限

- 双分支和教师—学生副本显著增加训练显存与实现复杂度。
- 依赖 style transfer 产生 target-like 域；迁移质量差时 TL 分支可能提供错误上下文。
- MHPCA 基于两阶段 proposal，迁移到纯 dense detector 需要重新定义 token 与几何关系。

## 评分

**9.1/10**：双分支结构、proposal 级非对称感知和渐进自蒸馏相互配合，跨天气、合成到真实和跨相机结果都强；训练代价与架构专用性是主要负担。
