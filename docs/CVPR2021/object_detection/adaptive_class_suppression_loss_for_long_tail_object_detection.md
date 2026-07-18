---
title: "[论文解读] Adaptive Class Suppression Loss for Long-Tail Object Detection"
description: "Adaptive Class Suppression Loss（ACSL）依据当前样本对各类别的预测置信度，选择性保留真正需要的负梯度，缓解长尾类别被头部样本过度压制。"
tags: ["CVPR 2021", "目标检测", "长尾学习", "ACSL"]
---

# Adaptive Class Suppression Loss for Long-Tail Object Detection

**论文**：[CVF 官方论文页面](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Adaptive_Class_Suppression_Loss_for_Long-Tail_Object_Detection_CVPR_2021_paper.html)  
**代码**：未提供官方代码链接  
**发表**：CVPR 2021  
**类别**：长尾分类损失与检测器微调

## 一句话总结

Adaptive Class Suppression Loss（ACSL）用模型当前输出而非类别频次先验决定负类是否参与损失：真实类始终训练，非真实类只有预测概率高于阈值 $\xi$、即与真实类发生混淆时才产生抑制梯度，从而保护稀有类别分类器，同时保留必要的类别判别学习。

## 研究背景与问题

在长尾检测中，一个头部类别样本对其余所有类别都是负样本。标准 sigmoid BCE 会持续向大量尾类分类器施加负梯度，而尾类自身的正样本很少，最终容易被压到几乎无法激活。Equalization Loss、Balanced Group Softmax 等方法依赖类别频次分组或先验统计，虽能缓解不平衡，却需要为不同数据集重新设计分组边界。

论文用 LVIS 与 Open Images 的分布差异说明固定分组不具通用性。以 BAGS 为例，在 LVIS 上两组分界设在 50–500 之间较好，过小或过大都会明显退化；同一边界又不适合类别更少、单类样本更多的 Open Images。ACSL 因此把判断粒度从“某个类别属于头部还是尾部”下沉到“当前样本是否真的与某个负类混淆”。

## 方法总览

ACSL 替换检测分类子网中的 softmax 交叉熵。对每个训练样本，先对各类 logits 做 sigmoid；真实类别的正损失始终保留，其他类别仅在其预测概率达到 $\xi$ 时保留负损失。低置信负类被视为已经容易区分，无需继续消耗梯度；高置信负类被视为易混类别，仍需压制。

## 方法详解

### 1. BCE 的过度抑制

设样本 $x_s$ 属于类别 $k$，类别 $i$ 的 logit 为 $z_i$，概率为

$$p_i=\frac{1}{1+e^{-z_i}}.$$

定义 $\hat p_i=p_i$（$i=k$）或 $1-p_i$（$i\ne k$），标准二元交叉熵为

$$L_{BCE}(x_s)=-\sum_{i=1}^{C}\log(\hat p_i).$$

其梯度对真实类为 $p_i-1$，对所有负类均为 $p_i$。在头部样本数量远多于尾部样本时，尾类分类器不断收到来自头部样本的负梯度，正向激活被淹没。

### 2. 自适应类别抑制

ACSL 为每一类引入二值权重：

$$L_{ACSL}(x_s)=-\sum_{i=1}^{C}w_i\log(\hat p_i),$$

$$w_i=\begin{cases}
1,&i=k,\\
1,&i\ne k\ \text{且}\ p_i\ge\xi,\\
0,&i\ne k\ \text{且}\ p_i<\xi.
\end{cases}$$

因此负类梯度变为 $w_ip_i$。$\xi$ 是“混淆阈值”：小阈值让多数负类继续被压制，保护作用有限；过大阈值只处理极高置信负类，又会损失类别间判别能力。权重完全由当前样本输出产生，不需要类别频率、采样数或分组边界。

### 3. 两阶段检测训练

论文在 Faster R-CNN-FPN 上先用常规损失训练 12 个 epoch，再用 ACSL 微调分类器 12 个 epoch。表示学习与分类器优化被解耦，ACSL 只替换分类目标；背景对所有类别为负样本，为进一步平衡，论文对 rare、common 类分别随机保留 1% 与 10% 的背景样本。

ACSL 与 Equalization Loss 的差异不只是“不查频次表”。EQL 对同一尾类的负梯度采用类别级统一策略，而 ACSL 对每个样本重新计算 $w_i$：同一个尾类在某个头部样本上若概率很低就不再压制，在语义相近样本上若概率升高则恢复负梯度。因此它既避免全局关闭尾类负样本，也避免 BCE 对所有负类一视同仁。

阈值还反映了训练状态。随着分类器变强，容易负类的 $p_i$ 会降到 $\xi$ 以下并自动退出损失，仍混淆的类别继续收到梯度；这相当于由网络自身维护一个动态 hard-negative 集合。论文没有使用类别样本数来更新该集合，因此从 LVIS 迁移到 Open Images 时沿用相同 $\xi$，仍获得稳定提升。

这一机制只屏蔽负项，真实类别的正梯度始终存在，所以 ACSL 不会因为样本来自 rare 类就降低其正监督。它调整的是类别之间互相压制的频率，而不是重新定义正样本或框匹配。

## 实验与证据

- **LVIS 设置**：LVIS v0.5，Faster R-CNN-FPN 为主，报告 mAP、rare/common/frequent 的 APr/APc/APf；还测试 ResNet-101、ResNeXt-101-64x4d 与 Cascade R-CNN。
- **阈值消融**：1× 基线为 21.18 mAP、4.30 APr；2× 基线为 22.28 mAP、7.38 APr。$\xi$ 从 0.01 增至 0.7 时持续改善，$\xi=0.7$ 达 26.36 mAP、18.64 APr、26.41 APc、29.37 APf；增到 0.9 后降为 25.99 mAP。
- **主要对比**：ResNet-50 上 ACSL 的 26.36 mAP 高于 EQL 的 25.06 和 BAGS 的 25.96，且 APr 为 18.64；ResNet-101 上达到 27.49 mAP，ResNeXt-101-64x4d 上达到 28.93，多尺度测试最高 29.47。
- **更强检测器**：Cascade R-CNN 配 ResNet-101 为 29.71 mAP，配 ResNeXt-101 为 31.47；增益主要来自 rare/common 类，frequent 类没有被用来交换尾类提升。
- **Open Images**：数据含 600 类、1.9M 图像和 16M 框，最频繁类别与最稀有类别图像数相差 126 千倍。相同 $\xi=0.7$ 下，ResNet-50-FPN 从 55.1 AP 提升到 60.3，ResNet-152 从 57.4 到 62.8；500 类对比中 ACSL 为 61.70 AP，高于 Class Aware Sampling 的 56.50 和 EQL 的 57.83。

## 对 YOLO-Agent 的启发

接入点是 YOLO 分类分支的 BCE/Focal 目标：对已分配正样本，保留真实类正项；对其余类别，根据当前 sigmoid 概率与 $\xi$ 生成停止梯度的二值掩码。若 YOLO 使用 quality-aware soft target，应先保留目标类的原质量标签，只把 ACSL 用在非目标类负项，避免破坏分类分数与定位质量的耦合。对照组至少包括原 BCE/Focal、EQL 式频次先验抑制、ACSL 单阶段训练和论文式分类器二阶段微调。

验收应在 LVIS 风格长尾划分上同时看 AP、APr、APc、APf。若 $\xi=0.7$ 不能接近论文从 21.18 到 26.36 mAP、从 4.30 到 18.64 APr 的方向性增益，或 APf 明显低于论文 29.37 的稳定水平，则不能宣称长尾问题得到缓解；若 $\xi=0.9$ 出现论文中的回落，应将搜索范围收回。还应检查负类掩码比例：若多数尾类在头部样本上仍被激活，ACSL 已退化回 BCE。

## 优点

- 不依赖类别频率与人工分组，同一阈值可用于 LVIS 和 Open Images。
- 在样本—类别粒度决定负梯度，兼顾尾类保护和易混类别判别。
- 只改分类损失，可接入 Faster R-CNN、Cascade R-CNN 及 YOLO 分类头。

## 局限

- 硬阈值 $\xi$ 不可微且仍需调参；模型早期置信度不可靠时可能产生错误掩码。
- 论文最佳方案采用两阶段分类器微调和背景采样，收益不能完全归因于单一公式。
- ACSL 处理类别长尾，不直接解决框尺度、实例密度或定位样本不平衡。

## 评分

- **方法简洁性：高**：一个输出驱动掩码即可控制负梯度。
- **长尾效果：高**：LVIS rare 类和 Open Images 均有大幅提升。
- **训练通用性：中上**：不需频次先验，但最佳结果依赖二阶段流程。
- **综合评价：推荐**：是 YOLO-Agent 长尾分类损失中低成本、高优先级的候选。
