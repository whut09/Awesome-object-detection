---
title: "[论文解读] Bucketed Ranking-based Losses for Efficient Training of Object Detectors"
description: "解析 BAP/BRS 如何以负样本桶和原型 logit 降低排序损失的时间与空间成本。"
tags: ["ECCV 2024", "目标检测", "排序损失", "训练效率"]
---

# Bucketed Ranking-based Losses for Efficient Training of Object Detectors

**论文**：[官方论文页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7634_ECCV_2024_paper.php)  
**代码**：官方未提供  
**会议**：ECCV 2024

## 一句话总结

论文把 AP Loss 与 Rank&Sort Loss 中数量巨大的负预测按相邻正样本之间的排名区间分桶，用每个桶的均值 logit 和桶大小代表整组负样本，得到 Bucketed AP（BAP）与 Bucketed Rank-Sort（BRS）Loss，在保留排序梯度目标的同时把训练复杂度降到可用于 Co-DETR 的水平。

## 研究背景与问题

AP Loss、RS Loss 直接优化检测排序，比交叉熵或 Focal Loss 更贴近 AP 评价，也对正负不平衡更稳健；代价是每个正样本都要与大量负样本比较。若正、负预测集合分别为 $P$、$N$，传统实现的核心时间约为 $O(|P||N|)$，原始两两矩阵空间甚至达到 $O((|P|+|N|)^2)$。论文指出 ATSS 在 COCO 上负预测数量可到 $10^8$ 量级，因此已有排序损失只能循环正样本、丢弃平凡负样本，训练时间和显存仍妨碍其进入多头、多 decoder 的 Transformer 检测器。

关键观察是：按分数排序后，落在同一对相邻正样本之间的负预测拥有相同排名作用，在 AP/RS 的 Identity Update 下也会获得相同或极相近的梯度。逐负样本比较因此存在大量重复计算。

## 方法总览

先将全部正负 logits 降序排列，设正样本依次为 $\tilde s_1^+,\ldots,\tilde s_{|P|}^+$，负样本被自然切成

$$
B_1>\tilde s_1^+>B_2>\tilde s_2^+>\cdots>\tilde s_{|P|}^+>B_{|P|+1}.
$$

每个负桶 $B_j$ 保存大小 $b_j$，并以均值 $s_j^b$ 作为 prototype logit。这样参与两两关系计算的对象最多为 $|P|$ 个正样本和 $|P|+1$ 个负原型，而不是全部负预测。BAP 重新定义正样本的排名误差及负梯度分配；BRS 沿用同一正负项，并保留原 RS Loss 的正正排序项。

## 方法详解

### 1. 排序损失与 Identity Update

对预测 $i,j$ 的分数 $s_i,s_j$，差值 $x_{ij}=s_j-s_i$，平滑阶跃函数 $H(x_{ij})$ 用于统计排在 $i$ 前面的预测。Identity Update 将不可导排序目标的更新归结为 primary terms：

$$
\frac{\partial L}{\partial s_i}=\frac{1}{Z}\left(\sum_jL_{ji}-\sum_jL_{ij}\right),
$$

$Z$ 是归一化常数，$L_{ij}$ 表示由样本 $i$ 向样本 $j$ 分配的排序误差。因此只要桶化后正确构造 $L_{ij}$，就能恢复实际 logit 梯度。

### 2. Bucketed AP

令 $x_{ij}^b=s_j^b-s_i$，第 $i$ 个正样本前的假阳性数量和总排名都用桶大小加权，其排名误差为

$$
\mathcal{E}_R^b(i)=\frac{NFP(i)}{rank(i)}=
\frac{\sum_{j=1}^{i}H(x_{ij}^b)b_j}
{\sum_{j=1}^{i}[H(x_{ij})+H(x_{ij}^b)b_j]}.
$$

$b_j$ 是第 $j$ 桶的负样本数，$H(x_{ij}^b)$ 判断该桶原型是否排在正样本 $i$ 前。因为各桶包含数量不同，误差分配概率采用 $p(j^b|i)=b_j/NFP(i)$；先得到正样本与桶原型的 primary term，再把原型梯度除以 $b_j$，平均分回桶内真实负样本。

### 3. BRS 与复杂度

RS Loss 的正负排序项与 AP Loss 相同，所以直接替换为桶化 primary term；正样本之间依据定位质量进行排序的项不涉及负桶，保持原定义。论文证明当平滑宽度 $\delta=0$ 时，BAP/BRS 与 AP/RS 给出完全相同的梯度；时间复杂度为

$$
O(\max((|P|+|N|)\log(|P|+|N|),|P|^2)).
$$

### 4. BRS-DETR

论文把 BRS 接入 Co-DETR。原 decoder 与辅助 decoder 的分类项为 Focal Loss，作者替换成 $L_{BRS}$，保留 $L_1$ 框损失与 GIoU，并动态设置 $\lambda_{bbox}=L_{BRS}/L_{bbox}$、$\lambda_{IoU}=L_{BRS}/L_{IoU}$；ATSS、Faster R-CNN 辅助头也改为相应排序损失。BRS-DETR 因而不是新增网络模块，而是统一替换多个头的分类排序目标。

## 实验与证据

主要检测实验在 COCO trainval35k 115K 图像训练、minival 5K 图像测试，默认 12 epochs、输入 1333×800、4 张 Tesla A100、总 batch 16。模型覆盖 Faster R-CNN、Cascade R-CNN、ATSS、PAA、Mask R-CNN 与 Co-DETR；实例分割还使用 Cityscapes、LVIS。

Faster R-CNN-R50 的 RS/BRS 为 39.4/39.5 AP，但单迭代时间从 0.50s 降至 0.17s；Cascade R-CNN 保持 41.1 AP，1.28s 降至 0.24s，即 5.3×；ATSS 的 RS/BRS 都为 39.8 AP，0.36s 降至 0.15s。Mask R-CNN 在 COCO、Cityscapes、LVIS 上分别获得约 2.2–2.5× 加速并基本保持 AP。合成 logits 实验把损失计算与骨干解耦，BAP 相对 AP 最高达到约 40× 加速。

BRS-DETR 使用 ResNet-50、300 queries、12 epochs 达到 **50.1 AP / 67.4 AP50 / 54.6 AP75**，高于同设置 Co-DETR 的 49.3 AP，也超过论文列出的 DINO 49.4、H-DETR 48.7；训练 Co-DETR 时，RS 的 4.14s/iter 被 BRS 降到 0.69s/iter。Swin-T 与 Swin-L 下 BRS-DETR 分别为 52.3、57.2 AP。相同训练预算消融中，把节省时间转为更多 epochs，Cascade R-CNN 与 Mask R-CNN 相对 RS 再提升 1.2、1.0 AP。

## 对 YOLO-Agent 的启发

最直接接入点是 YOLO 分类损失：在完成正负分配后，收集所有类别 logits 与正样本定位质量，把 Focal/BCE 替换为 BAP 或 BRS；若使用 BRS，正样本间的排序目标应以 IoU 或任务对齐质量排序。对照组必须包含 BCE/Focal、原始 AP/RS、BAP/BRS，并分别记录纯 loss kernel 时间、整步时间、峰值显存和 COCO AP。论文相关指标表明目标是在 AP 变化不超过 0.2–0.3 时获得约 2× 以上训练加速；失败阈值可设为整步加速低于 1.3×、AP 下降超过 0.3、或桶内梯度方差显著增大。若失败，应先检查类别维度是否被错误混桶，以及正样本数量很少时排序和 prototype 构造是否退化。

## 优点

- 直接删除排序损失中重复的正负比较，理论复杂度与实测训练时间一致下降。
- 同时覆盖 AP Loss、RS Loss、两阶段、一阶段、实例分割和 Transformer 检测器。
- 当 $\delta=0$ 时有梯度等价定理，而非仅凭经验近似。
- 省下的训练预算既可降低成本，也可兑换为更多 epochs 和更高 AP。

## 局限

- 每步仍需全局排序，正样本较多时保留 $O(|P|^2)$ 的正正比较成本。
- 均值原型在 $\delta>0$ 时是实践近似，桶内 logits 跨越平滑区间可能破坏严格等价。
- 官方未提供代码，复杂多头 detector 的复现接口与高效 kernel 需要自行实现。

## 评分

- **创新性：9/10**——利用排序区间的梯度同质性压缩负样本比较。
- **实验充分性：9/10**——覆盖三类任务、三套数据和七种检测器，并区分 loss 与整网耗时。
- **可复现性：7.5/10**——公式和设置充分，但缺少官方代码。
- **YOLO-Agent 适配度：8.5/10**——可直接替换分类目标，但需要高效排序与分桶实现。
