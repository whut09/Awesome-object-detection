---
title: "[论文解读] Distilling Object Detectors via Decoupled Features"
description: "解析 DeFeat 如何在 FPN 特征与 RoI 分类头两级解耦前景/背景知识并分别蒸馏。"
tags: ["CVPR 2021", "目标检测", "知识蒸馏", "前景背景解耦"]
---

# Distilling Object Detectors via Decoupled Features

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/CVPR2021/html/Guo_Distilling_Object_Detectors_via_Decoupled_Features_CVPR_2021_paper.html)  
**代码**：未提供  
**发表**：CVPR 2021

## 一句话总结

DeFeat 在检测器 neck 和分类 head 两个层级分别解耦知识：FPN 特征按 GT 掩码分成 object/background 区域独立归一化和加权，RoI proposals 按正/负样本分开做温度化 KL 蒸馏，从而保留背景抑制误检的知识，又避免其数量和梯度淹没前景。

## 研究背景与问题

早期检测蒸馏常假设背景是噪声，只模仿 GT 附近的教师特征。DeFeat 的误差分析却显示，单独蒸馏背景区域与单独蒸馏目标区域都能把 COCO mAP 从 37.7 提到 39.9：目标知识主要改善定位错误，背景知识则降低 background false positives。问题不在于“是否蒸馏背景”，而在于前景、背景面积和优化难度悬殊，不能共享同一归一化与权重。

同样的不平衡也存在于两阶段检测器的分类头。正 proposal 的蒸馏损失持续高于负 proposal，小梯度的背景信息可能被前景梯度覆盖；若所有 proposal 使用同一温度和系数，教师的分类暗知识并未被充分利用。因此论文把 feature-level 与 proposal-level 解耦组成 DeFeat，而不是只设计一种掩码。

## 方法总览

第一层“decoupled-neck”根据 GT 框生成每个 FPN 层的二值掩码，分别计算前景和背景的特征 MSE，并用各自元素数归一化。第二层“decoupled-cls”使用教师 RPN proposals 同时送入教师、学生 RoI head，再把 proposal 按匹配标签分成 positive/negative 两组，以不同温度 $T_{obj},T_{bg}$ 和权重 $\beta_{obj},\beta_{bg}$ 计算 KL。原检测回归和 RPN 损失保持不变。

这里的“decoupled”不是把网络结构拆成两个分支，而是把同一蒸馏张量的统计域拆开。neck 仍输出原 FPN 特征，RoI head 也没有新增推理模块；变化只发生在训练损失的掩码、归一化、温度与系数上。对于多级 FPN，GT 只在其对应尺度产生 object mask，避免同一个目标在所有层重复占据前景区域。

## 方法详解

### 1. Decoupled Neck Features

学生、教师中间特征为 $S,T\in\mathbb R^{H\times W\times C}$，学生先经 adaptation layer $\phi$ 对齐。对 GT 框集合 $B$ 构造二值掩码 $M_{h,w}=1[(h,w)\in B]$；有 FPN 时，GT 先分配到对应尺度再生成各层掩码。解耦特征损失为：

$$
\mathcal L_{fea}=\frac{\alpha_{obj}}{2N_{obj}}\sum M_{h,w}(\phi(S_{h,w,c})-T_{h,w,c})^2
+\frac{\alpha_{bg}}{2N_{bg}}\sum(1-M_{h,w})(\phi(S_{h,w,c})-T_{h,w,c})^2.
$$

$N_{obj}$ 与 $N_{bg}$ 是两类区域的元素数，$\alpha_{obj},\alpha_{bg}$ 控制其贡献。分开归一化使大片背景不会仅因面积大而支配损失，也允许背景使用与目标不同的系数。

### 2. Decoupled Region Proposals

论文使用教师生成的 $K$ 个 proposals，因为教师 proposals 更准确、信息更密集。正 proposal 的学生/教师 logits 分别经温度 $T_{obj}$ softmax；负 proposal 类似使用 $T_{bg}$。分类蒸馏为：

$$
\mathcal L_{cls}=\frac{\beta_{obj}}{K_{obj}}\sum_i b_i\mathcal L_{KL}(p_i^{s,T_{obj}},p_i^{t,T_{obj}})
+\frac{\beta_{bg}}{K_{bg}}\sum_i(1-b_i)\mathcal L_{KL}(p_i^{s,T_{bg}},p_i^{t,T_{bg}}).
$$

$b_i\in\{0,1\}$ 表示 proposal 是否为正，$K_{obj},K_{bg}$ 是两组数量；KL 中乘 $T^2$ 保持梯度量级。总训练目标由 $\mathcal L_{fea}$、解耦分类蒸馏、原 bbox regression 和 RPN loss 组成。

## 实验与证据

实验在 COCO 上覆盖 Faster R-CNN/FPN 与 RetinaNet，多组教师—学生 backbone，并与 FitNets、FGFI 等蒸馏方法比较。关键主实验以 ResNet152 教师和 ResNet50 学生为例。

论文用 COCO 误差类型进一步解释结果：正确类别但框偏移记为 Loc，背景误检记为 BG，另有相似类别、其他类别和漏检。只蒸馏 object regions 时定位错误下降更明显；只蒸馏 background regions 时 BG 比例下降，并获得与前景蒸馏相当的 39.9 mAP。这个证据支持分别加权，而不是把背景增益误认为更多像素带来的普通正则化。

- Faster R-CNN ResNet50-FPN 学生为 37.4 mAP；all-neck 蒸馏 40.1，decoupled-neck 40.4，再加 all-cls 40.5、decoupled-cls 40.8；进一步蒸馏 backbone 达到 40.9。教师为 41.3。
- RetinaNet ResNet50 学生为 36.5，all-neck 39.1，decoupled-neck 39.5，加入 backbone 蒸馏为 39.7；教师为 40.5。说明即使没有 RoI proposal，特征区域解耦本身仍有效。
- proposal 消融中，只用正样本最高为 38.1，只用负样本最高为 38.6；正负统一处理为 38.2，解耦温度与权重后为 38.9。把负样本随机下采样到与正样本相同数量会下降 0.2 AP，表明背景数量本身承载信息。
- proposal 损失曲线显示 negative distillation loss 比 positive 更快下降；两组难度不同正是使用独立温度与系数的原因。若直接把学生 proposals 输入双方 head，结果低于教师 proposals，因为学生候选更不准，也使两个网络比较的区域不够稳定。
- neck 消融显示所有区域等权为 40.1，前景/背景解耦进一步到 40.4；随机掩码反而略降。教师 proposals 喂给双方 head 的结果也优于使用学生 proposals。

论文还比较 scaled GT、Gaussian Mask、fine-grained region 与随机掩码：简单扩大或随机选择区域并不能稳定超过全图蒸馏，Gaussian Mask 甚至更差；在所有区域都参与的前提下再解耦权重，才从 40.1 提到 40.4。这说明 DeFeat 的关键不是找到更“漂亮”的前景形状，而是承认目标区与背景区承担不同知识并分别优化。

## 对 YOLO-Agent 的启发

YOLO 没有 RoI head 时，首要接入点是 neck 输出：按现有 label assigner 将每层位置划为 object/background，两区分别归一化蒸馏。若 YOLO-Agent 有 query/proposal 精修阶段，再增加 decoupled-cls；否则可把已匹配正点与 hard negative 点视为两组 logits。对照组应为无 KD、全图特征 KD、仅 object KD、仅 background KD、decoupled feature KD；指标记录 AP、AP75、背景误检率和漏检率。

失败阈值应围绕论文的两类作用：若 background-only 对背景误检率没有下降，或完整 DeFeat 相对 object-only 的 AP 增益低于 0.2，说明背景掩码、归一化或教师校准无效；若 decoupled feature KD 相对 all-feature KD 低于 0.2 AP，则不应增加复杂度；若正负 logits 蒸馏后 AP75 上升但背景误检增加超过 5%，应降低 $\beta_{obj}$ 或提高负样本权重，而不是简单下采样负样本。

## 优点

- 用实验推翻“背景只会带来噪声”的简单假设，并解释背景知识降低误检的作用。
- feature 与 proposal 两级解耦对应检测器 neck/head 的真实结构。
- 对两阶段和单阶段 RetinaNet 都有效，推理阶段无额外模块。

## 局限

- GT 框内外仍是粗粒度划分，框内背景、遮挡区和上下文没有进一步区分。
- 两阶段版本依赖教师 proposals，对一阶段/无 proposal 检测器只能保留部分设计。
- 多个温度和权重需要调节，教师错误背景也可能被学生继承。

## 评分

- **方法创新：8.5/10**——把检测蒸馏的不平衡落实到特征与 proposal 两级。
- **实验充分：8.5/10**——包含区域、proposal、掩码与共享 proposal 分析。
- **工程可用：8/10**——训练期改动明确，推理零开销，但一阶段适配不完整。
- **综合评分：8.3/10**。
