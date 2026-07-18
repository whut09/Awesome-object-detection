---
title: "[论文解读] HEAD: HEtero-Assists Distillation for Heterogeneous Object Detectors"
description: "解析 HEAD 如何用同构助手头桥接异构检测器的语义鸿沟，以及无教师扩展 TF-HEAD。"
tags: ["ECCV 2022", "目标检测", "知识蒸馏", "异构检测器"]
---

# HEAD: HEtero-Assists Distillation for Heterogeneous Object Detectors

**论文**：[官方论文页面](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2285_ECCV_2022_paper.php)  
**代码**：论文条目未提供官方代码链接  
**发表**：ECCV 2022  
**主题**：异构检测器蒸馏、助手检测头、Teacher-Free KD

## 一句话总结

HEAD 在学生骨干上挂接一个与教师检测头同构的 Assistant，用 Assistant-based KD 先把异构蒸馏改写成头部同构蒸馏，再用 Cross-architecture KD 直接约束学生头；TF-HEAD 则让多个异构助手仅靠标注在线训练，在没有预训练教师时仍向学生骨干和检测头注入不同架构的归纳偏置。

## 研究背景与问题

Faster R-CNN、RetinaNet、FCOS、RepPoints 的头部结构、候选表示和标签分配不同，因此即使骨干都输出 FPN，特征也受到不同优化目标塑形。论文把这种由检测头和训练协议造成的 backbone semantic gap 视为异构蒸馏失败的核心，而不是简单的通道数或空间尺寸不匹配。

传统特征蒸馏通常写成 `D(F_T, φ(F_S))`：教师和学生的中间特征分别为 `F_T、F_S`，`φ` 是 1×1 卷积或线性适配层，`D` 常为 MSE。对同构检测器这通常可行，但当教师与学生的头部推理路径不同，直接 MSE 会让学生模仿一个并非为其架构优化的表示，容易过度约束。

HEAD 的具体实例是 Faster R-CNN 教师与 RetinaNet 学生。它复制教师 R-CNN head 的结构和预训练权重，在学生 FPN 后构造 Assistant；学生特征同时进入原 RetinaNet head 与 R-CNN Assistant，训练结束后删除 Assistant，所以推理结构和开销保持学生原样。

## 方法总览

HEAD 有两条蒸馏路径：

- **AKD（Assistant-based KD）**：教师头处理教师特征，Assistant 处理学生特征；因为二者同构，可逐层匹配中间表示，并以教师同款检测损失监督 Assistant。
- **CKD（Cross-architecture KD）**：将教师头的稀疏 RoI 表示与学生头对应的稠密位置表示对齐，直接蒸馏学生检测头。

学生本身仍按原检测器训练，其监督为 `L_gt^S=L_reg^S+L_cls^S`。完整 HEAD 目标为

\[
L_{HEAD}=L_{gt}^{S}+L_A+L_C,
\]

其中 `L_A` 是 AKD 总损失，`L_C` 是 CKD 损失。

## 方法详解

设教师、学生骨干含 neck 的输出为 `P_T` 与 `P_S`。若教师是两阶段检测器，教师头和 Assistant 都对各自 FPN 特征执行 RoI Align。两阶段学生直接使用其 RPN RoI；一阶段学生没有 RPN，论文把每个 anchor 的类别 logits 转成 class-agnostic objectness，再按 RPN 流程生成 `N` 个 RoI。

教师头与 Assistant 的第 `l` 层中间特征记为 `F_l^T、F_l^A`，共 `L` 层。AKD 的逐层项为

\[
L_l^A=D(F_l^T,\phi(F_l^A)),
\]

`D` 为 MSE，`φ` 为维度适配层。Assistant 还接受真实标注监督 `L_gt^A`，避免盲从教师错误：

\[
L_A=L_{gt}^{A}+\lambda_A\frac{1}{L}\sum_{l=1}^{L}L_l^A.
\]

梯度从 Assistant 回传到学生骨干，迫使 `P_S` 包含足以复现教师头推理过程的信息。这是 HEAD 桥接语义鸿沟的主路径。

CKD 负责学生头。以 R-CNN 教师、RetinaNet 学生为例，教师产生 `N×C'` 的稀疏 RoI 特征；论文追溯每个 RoI 对应的原始 anchor，再从 RetinaNet 分类分支的 `C×H×W` 稠密特征中采样相应位置，组成 `N×C` 的 `F_RoIs^S`：

\[
L_C=\lambda_C D(F_1^T,\phi(F_{RoIs}^S)).
\]

若双方都是一阶段或都是两阶段，则直接对齐对应中间层；`φ` 对卷积特征使用 1×1 卷积，对二维 RoI 特征使用线性层。

TF-HEAD 不需要教师。每个异构 Assistant 由自身的真实标注损失 `L_a^A` 学习，并通过 `L_a^C` 蒸馏学生头：

\[
L_{TF-HEAD}=L_{gt}^{S}+\sum_a(L_a^A+L_a^C).
\]

例如 RetinaNet 学生同时接 FCOS head 和 R-CNN head 两个助手；不同头把点式、anchor 式和 RoI 式先验写入共享学生骨干。论文在首次学习率下降后停止 CKD，因为此后助手与学生头的优化方向开始分离，继续强制模仿会过正则化。

## 实验与证据

实验在 MS COCO 上进行。R18 RetinaNet 基线为 31.7 mAP；以 R50 Faster R-CNN（40.3 mAP）为教师，HEAD 达到 36.2 mAP，超过 G-DetKD 的 35.4，也高于仅 AKD 的 35.5。MNV2 RetinaNet 从 28.5 提升到 32.8。以 RepPoints 教师时，R18 RetinaNet 的 HEAD-AKD 为 34.2，而 SGFI 只有 31.6，说明同构助手比直接异构特征匹配更稳健。

对 FCOS 学生，R18 基线 32.5，Faster R-CNN 教师下 HEAD 达到 36.0；MNV2 基线 30.0，HEAD 为 33.5。两阶段学生同样可用：R18 Faster R-CNN 从 33.9 提升到 36.7，教师为 R50 Cascade R-CNN。

无教师实验采用 R50、2×训练：TF-HEAD 将 RetinaNet 从 38.7 提到 40.2、FCOS 从 41.0 提到 42.3、Faster R-CNN 从 39.4 提到 40.5，均超过 LabelEnc。R18 RetinaNet 的组件消融中，FCOS Assistant 单独带来 1.2 mAP，R-CNN Assistant 单独带来 1.7；两者与 CKD 合用达到 33.9，比 31.7 基线高 2.2。首次降学习率的第 8 个 epoch 后停止 CKD，可从 33.8 小幅提高到 33.9。

## 对 YOLO-Agent 的启发

接入点应放在 YOLO 骨干/neck 输出与原检测头之间：训练时临时挂接一个 RoI-based Assistant 或 anchor-based Assistant，推理时删除。对照组应包括原 YOLO、直接 FPN-MSE、仅 Assistant 标注训练、AKD、AKD+CKD；这样能验证收益是否来自“助手改变骨干优化”而非额外训练参数。

核心指标除 COCO AP、AP75、APS 外，还应记录学生 FPN 与教师/助手中间层的 MSE、CKD 在学习率衰减前后的趋势，以及移除 Assistant 后的真实推理延迟。论文中 R18 RetinaNet 的关键参照是 31.7→35.5（仅 AKD）和 31.7→36.2（完整 HEAD）。若完整方案比直接 FPN-MSE 的 AP 增益不足 0.5，或 CKD 在第一次降学习率后连续 500 iter 上升且验证 AP 不增，则判定跨头约束失效并提前停止 CKD；若删除 Assistant 后模型参数或推理 FLOPs 仍增加，则实现不合格。

## 优点

- 用教师同构头作为中介，明确针对异构检测头塑造出的语义鸿沟。
- AKD、CKD 和真实标注监督职责清楚，可覆盖一阶段、两阶段及其交叉组合。
- Assistant 仅参与训练；TF-HEAD 在没有预训练教师时仍可利用多种检测头先验。

## 局限

- 训练期需额外前向教师和 Assistant，显存、实现复杂度及 RoI/anchor 对齐成本明显增加。
- CKD 的有效训练窗口依赖学习率日程，论文采用经验性的 early stop。
- 助手结构和损失权重仍需按学生类型设计，并非完全无配置的通用蒸馏器。

## 评分

**9.0 / 10**：HEAD 把“异构特征难对齐”转化为“通过同构助手优化学生骨干”，问题与结构高度对应；TF-HEAD 进一步扩大适用范围，但训练资源和对齐工程成本较高。
