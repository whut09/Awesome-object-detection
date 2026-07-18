---
title: "[论文解读] Pseudo-IoU: Improving Label Assignment in Anchor-Free Object Detection"
description: "解析 Pseudo-IoU 如何为 anchor-free 特征点构造伪框，并用 IoU 阈值完成标准化标签分配。"
tags: ["CVPR Workshops 2021", "目标检测", "Anchor-Free", "标签分配"]
---

# Pseudo-IoU: Improving Label Assignment in Anchor-Free Object Detection

**论文**：[官方论文页面](https://arxiv.org/abs/2104.14082)  
**代码**：[官方代码](https://github.com/Barrett-python/Pseudo-IoU)  
**发表**：CVPR Workshops 2021

## 一句话总结

Pseudo-IoU 在每个落入 GT 的 anchor-free 特征点上构造一个与 GT 同宽同高、但以该点为中心的伪框，用伪框和 GT 的 IoU 作为点质量，再以阈值筛选正样本，从而把 anchor-based 检测器成熟的 IoU 分配规则移植到无锚检测头。

## 研究背景与问题

Faster R-CNN、RetinaNet 等 anchor-based 方法天然有候选框，因此能用 IoU 阈值定义正负样本；FCOS 一类 anchor-free 方法直接从特征点预测到 GT 四边的距离，没有预定义框，常把 GT 内部或缩小中心区域的点全部作为正样本。该规则简单，但同一 GT 内靠近边缘和靠近中心的点被同等对待，容易引入低质量回归样本。

论文的核心不是再增加一个质量预测分支，而是为训练期的每个点补出一个可比较的几何对象。这个伪框由 GT 尺寸和点位置唯一决定，不依赖网络预测，因此不增加参数、训练计算分支或推理成本；它把“点是否适合回归该 GT”转换为标准的重叠率问题。

## 方法总览

检测器采用 ResNet-FPN，输出 $P_3$ 到 $P_7$，分类和回归头分别预测类别及 $(l,r,t,b)$。训练时先将特征点投影回输入图像；若点位于某 GT 内，则以该点为中心生成与 GT 同尺寸的伪框，计算 Pseudo-IoU。仅 Pseudo-IoU 大于阈值 $T$ 的点作为正样本，其余为负样本；推理流程仍是分类、框解码、置信度过滤和 NMS。

具体尺度关系沿用 RetinaNet：$P_3,P_4,P_5$ 由 $C_3,C_4,C_5$ 的自顶向下与横向连接产生，$P_6,P_7$ 由步长为 2 的卷积继续下采样，所有层均为 256 通道。论文去掉 $P_2$ 以控制计算量。点 $(x,y)$ 在输入图上的四边距离由 GT 中心 $(x_c,y_c)$、宽高 $(w,h)$ 计算，因此 Pseudo-IoU 必须在统一图像坐标中求值，不能直接拿不同 stride 的特征索引比较。

## 方法详解

### 1. 伪框的构造

设点 $P$ 到 GT 框 $B$ 左、右、上、下边的距离为 $l_B,r_B,t_B,b_B$。GT 的宽高分别为 $l_B+r_B$ 与 $t_B+b_B$。以 $P$ 为中心构造伪框 $A$，令：

$$
l_A=r_A=\frac{l_B+r_B}{2},\qquad
t_A=b_A=\frac{t_B+b_B}{2}.
$$

因此 $A$ 与 $B$ 尺寸完全相同，差别只有中心位置。$S_A=(l_A+r_A)(t_A+b_A)$，$S_B=(l_B+r_B)(t_B+b_B)$。

### 2. Pseudo-IoU

两框交集在四个方向上的距离为 $l_I=\min(l_A,l_B)$、$r_I=\min(r_A,r_B)$、$t_I=\min(t_A,t_B)$、$b_I=\min(b_A,b_B)$，交集面积

$$
S_I=(l_I+r_I)(t_I+b_I),
$$

于是

$$
\mathrm{PseudoIoU}(A,B)=\frac{S_I}{S_A+S_B-S_I}.
$$

该值属于 $[0,1]$：点越接近 GT 中心，两个同尺寸框越重合，质量越高。论文把大于阈值 $T$ 的点标为正样本，而不是像 FCOS centerness 那样在推理时重加权分数。

这也解释了它与 scaled-box 的差异。scaled-box 只判断点是否落在按比例缩小的矩形内，边界是离散的；Pseudo-IoU 则给每个内部点一个连续几何值，再用阈值切分，横纵偏移会共同影响交并比。它与 centerness 的形式也不同：centerness 使用左右、上下距离的最小/最大比乘积，Pseudo-IoU 明确对应两个同尺寸框的重叠，可直接继承“IoU 大于某阈值”的分配语义。

### 3. 检测头与损失

FPN 的每层特征送入不共享权重的分类与回归子网：各含四层 $3\times3$、256 通道卷积，分类末层输出类别数 $C$，回归末层输出四个边距。总损失为

$$
\mathcal L=\frac{1}{N_{pos}}\sum_{x,y}\mathcal L_{FL}(C^A_{x,y},C^B_{x,y})+
\frac{\lambda}{N_{pos}}\sum_{x,y}\mathcal L_{IoU}(R^A_{x,y},R^B_{x,y}).
$$

$C^A、R^A$ 是分类和回归预测，$C^B、R^B$ 是标签，$N_{pos}$ 为正样本数，$\lambda$ 平衡回归损失。Pseudo-IoU 只决定哪些位置进入正样本集合。

## 实验与证据

论文在 PASCAL VOC 07+12 和 MS COCO 上实验。VOC 使用 2007+2012 trainval 训练、2007 test 测试；COCO 使用 trainval35k、minival 5k 与 test-dev。主要对照包括无精细分配的 AF-Baseline、RetinaNet，以及 FCOS 风格 centerness 阈值和 FSAF 风格 scaled-box。

VOC 模型采用 ImageNet 预训练 ResNet-101、固定 BatchNorm、GroupNorm，训练 12 epoch，初始学习率 0.01，在第 8、10 epoch 衰减，回归损失权重 $\lambda=1$；测试分数阈值与 NMS 阈值分别为 0.05 和 0.5。COCO 训练 24 epoch，在第 16、22 epoch 衰减，单尺度短边 800、长边不超过 1333。这些设置表明论文的主要变量确实是训练样本分配，而非额外增强或多尺度测试。

- VOC 上基线为 76.9 mAP，Pseudo-IoU 阈值 0.4 达到 79.0；阈值 0.5 为 78.9，但 0.6、0.7 分别骤降到 45.6、22.1，证明过严筛选会造成正样本不足。
- 在 8、12、16、24 epoch 下，Pseudo-IoU 相对基线分别为 79.0 对 76.9、80.4 对 78.4、80.2 对 78.3、80.1 对 77.8，说明增益不能简单由更长训练替代。
- 最佳 Pseudo-IoU 为 79.0 mAP，高于 centerness 78.7 和 scaled-box 78.3。这里 centerness 被用于筛正样本，而非 FCOS 的输出重加权。
- COCO minival 的 ResNet-50-FPN AF-Baseline 为 33.9 AP；加入 Pseudo-IoU 为 36.8 AP，再加入 GIoU 为 37.1，叠加 centerness branch 为 37.4。test-dev 上 ResNet-101-FPN 为 41.5 AP，使用 DCN 和更强 ResNeXt 后最高报告 44.5 AP。

可视化中，原 anchor-free baseline 出现更多假阳性和不准确框；用 0.5 Pseudo-IoU 采样后，两类错误都明显减少。这与 AP75 从基线 35.3 提升到 39.2 的结果一致，表明改进不只是分类置信度变化，也来自更可靠的回归正样本。

由于计算只使用 GT 几何，训练早期也不会受到未收敛预测框的扰动。

## 对 YOLO-Agent 的启发

最直接接入点是 anchor-free YOLO head 的正样本候选过滤：在现有“点落入 GT/中心区域”之后计算 Pseudo-IoU，并在不改变分类、DFL/IoU 回归和推理结构的前提下增加阈值门控。对照组应为原 assigner、scaled-box、centerness 选样和 Pseudo-IoU；指标至少包含 AP、AP75、每 GT 正样本数、无正样本 GT 比例及训练吞吐。

失败阈值可依据论文敏感性设定：若 $T=0.4$ 相对原 assigner 的 AP 增益低于 0.5，应检查伪框是否按输入图尺度而非特征图尺度计算；若任何尺度上无正样本 GT 比例超过 1%，或 $T=0.5\rightarrow0.6$ 出现类似大幅退化，则阈值过严，应改为分层阈值或保证每个 GT 的 top-1 点；若 AP50 上升但 AP75 不升，说明该规则只减少了误检，没有改善高质量回归，应与现有质量分支做交互消融。

## 优点

- 几何定义清楚，无新增参数和推理开销。
- 让 anchor-free 选样获得可复用的 IoU 阈值语义。
- 与 GIoU 回归和 centerness 分支兼容，适合作为独立 assigner 组件测试。

## 局限

- 伪框固定为 GT 尺寸，质量仅由点到中心的偏移决定，未利用模型当前的分类和回归能力。
- 单一阈值对小目标、低分辨率 FPN 层非常敏感，过高会迅速耗尽正样本。
- 多 GT 重叠时论文没有提供全局冲突优化，仍需依赖既有匹配规则。

## 评分

- **方法创新：7.5/10**——用伪框补齐 anchor-free 的 IoU 几何接口，思路简洁。
- **实验充分：7.5/10**——覆盖 VOC、COCO 和多种阈值，但拥挤冲突分析不足。
- **工程可用：9/10**——实现局部、训练推理均几乎无额外成本。
- **综合评分：8.0/10**。
