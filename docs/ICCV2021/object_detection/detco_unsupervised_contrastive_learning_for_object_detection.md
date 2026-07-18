---
title: "[论文解读] DetCo: Unsupervised Contrastive Learning for Object Detection"
description: "解析 DetCo 的多层监督与全局—局部对比学习如何兼顾检测和分类迁移。"
tags: ["ICCV 2021", "目标检测", "自监督学习", "对比学习"]
---

# DetCo: Unsupervised Contrastive Learning for Object Detection

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/ICCV2021/html/Xie_DetCo_Unsupervised_Contrastive_Learning_for_Object_Detection_ICCV_2021_paper.html)  
**代码**：官方条目未提供代码链接  
**发表**：ICCV 2021

## 一句话总结

DetCo 在 MoCo v2 上同时引入 Multi-Level Supervision（MLS）和 Global-and-Local Contrastive Learning（GLC），让 Res2–Res5 各层都具备判别性，并在整图与九宫格局部块之间建立对比约束，从而获得更适合 FPN 检测、又不牺牲 ImageNet 分类的自监督预训练表示。

## 研究背景与问题

传统自监督对比学习主要优化最终层的整图语义，适合图像分类，却不保证浅层和中层特征能区分局部实例。FPN 检测器恰好需要多个 backbone stage 同时承担不同尺度目标的预测；若只有 Res5 判别性强，检测迁移会受限。相反，DenseCL、InsLoc、PatchReID 等同期方法强化局部检测能力，但论文指出它们在 ImageNet 线性分类上明显退步，形成检测与分类的取舍。

DetCo 的目标不是构造检测框伪标签，而是让同一套无标注 ImageNet 预训练同时学习“各层可分”和“全局—局部一致”。MLS 对应特征金字塔的多尺度要求，GLC 对应检测器把局部区域当作分类对象的机制；两者分别修正网络深度维和空间粒度维的监督缺口。

## 方法总览

DetCo 保留 MoCo v2 的 query encoder、动量更新 key encoder、MLP projection head 与 memory queue。不同之处有两点：第一，在 Res2、Res3、Res4、Res5 后各放置不共享参数的全局 MLP 和独立队列，每个阶段都计算 InfoNCE；第二，除两幅整图增强 \(I_q,I_k\) 外，还生成两组局部 patch 集合 \(P_q,P_k\)，每组由 9 个打乱的网格块组成，经过 backbone 后拼接并送入独立局部 MLP。

每个 stage 同时包含三种正对：global-to-global 保持图像级语义，local-to-local 保持局部组合的一致性，global-to-local 让整图表示与其局部集合互相校准。最终在四个 stage 上加权求和。

## 方法详解

完整损失写为

\[
L(I_q,I_k,P_q,P_k)=\sum_{i=1}^{4}w_i\left(L_{g\to g}^{i}+L_{l\to l}^{i}+L_{g\to l}^{i}\right),
\]

其中 \(i\) 对应 Res2–Res5，\(w_i\) 是层级权重，浅层权重小于深层；\(I\) 表示全局图像视图，\(P\) 表示局部 patch 集。每层使用独立 projection head 与 memory bank，避免不同语义层共享同一负样本空间。

全局 InfoNCE 为

\[
L_{g\to g}=-\log\frac{\exp(q^g\cdot k_+^g/\tau)}{\sum_{j=0}^{K}\exp(q^g\cdot k_j^g/\tau)},
\]

\(q^g\) 是 query encoder 的整图表示，\(k_+^g\) 是同一图像的 key 表示，\(k_j^g\) 包含正样本与队列负样本，\(K\) 为队列候选数，\(\tau\) 为温度。局部集合经过 backbone 得到 9 个特征，拼接后由局部 MLP 形成 \(q^l,k^l\)。于是

\[
L_{g\to l}=-\log\frac{\exp(q^l\cdot k_+^g/\tau)}{\sum_{j=0}^{K}\exp(q^l\cdot k_j^g/\tau)},\quad
L_{l\to l}=-\log\frac{\exp(q^l\cdot k_+^l/\tau)}{\sum_{j=0}^{K}\exp(q^l\cdot k_j^l/\tau)}.
\]

局部增强先随机裁取至少覆盖原图 60% 的区域并缩放到 255×255，再分成 3×3 个 85×85 网格并随机打乱，最后从每格裁成 64×64，减少相邻块边界连续性。全局视图使用 224×224 随机裁剪、翻转、模糊、颜色扰动与 RandAugment。

## 实验与证据

预训练统一采用 ResNet-50，在无标签 ImageNet 上训练 200 epoch；实验覆盖 ImageNet 分类、PASCAL VOC、COCO、Cityscapes、DensePose、关键点与 3D 人体形状。COCO 的 Mask R-CNN R50-C4 90k 设置中，DetCo 得到 39.8 box AP，而 MoCo v2 为 38.9；R50-FPN 下为 40.1，对应 MoCo v2 的 38.9。Sparse R-CNN 使用监督预训练为 45.0 AP，DetCo 提升到 46.5 AP，且小目标 AP 从 27.7 到 30.8。

Cityscapes 上，DetCo 的实例分割 AP 为 34.7、语义分割 mIoU 为 76.5，分别比监督预训练高 1.8 和 1.9。ImageNet 线性分类方面，它比 InsLoc、DenseCL、PatchReID 分别高 6.9、5.0、4.8 个 top-1 百分点，说明检测友好性没有以分类崩塌为代价。

核心消融在 100 epoch MoCo v2 基线上进行。基线为 64.3 top-1、56.3 VOC mAP；只加 MLS 后分类降到 63.2，但检测升到 57.0；只加 GLC 后两者升到 67.1 和 56.8；两者结合为 66.6 和 57.4。分层 SVM 结果更直接：基线 Res2/3/4/5 为 47.1/58.2/70.9/82.1，完整 DetCo 为 51.6/69.7/82.5/84.3，MLS 的主要作用确实集中在中浅层。注意力可视化也显示 DetCo 能激活多目标及更准确边界。

COCO 结果进一步区分了预训练对检测与分割分支的影响。Mask R-CNN R50-C4 中，DetCo 的 box AP/AP75 为 39.8/43.0，mask AP 为 34.7；R50-FPN 中 box AP/AP75 为 40.1/43.9，mask AP 为 36.4。两种架构都优于监督预训练和列出的无监督基线，说明收益不是只适配某一种 neck。DensePose 上 DetCo 为 51.3 AP，也高于监督预训练的 50.8，表明学习到的局部一致性可迁移到比矩形框更细的密集人体对应任务。

MLS 与 GLC 的交互也值得注意：MLS 强制浅层可分，会分走最终层用于整图分类的表示能力；GLC 同时提供整图—局部正对，既提高浅层，也把局部证据重新汇入全局语义。因此完整 DetCo 的分类精度虽略低于仅 GLC，却明显高于原 MoCo v2，并取得最高检测 mAP。这种互补关系是复现时必须保留的，而不能把四层 InfoNCE 当成独立可插拔技巧。

## 对 YOLO-Agent 的启发

适合接入 YOLO-Agent 的位置是 backbone 自监督预训练阶段，而非检测损失内部。为 P2/P3/P4/P5 或对应 CSP stage 分别建立 projection head 和队列，同时生成整图视图及 3×3 局部块视图；预训练结束后丢弃这些头，只把 backbone 权重交给 YOLO 微调。对照组应包括 MoCo v2 单末层监督、仅 MLS、仅 GLC、MLS+GLC，以及 ImageNet 监督预训练。

论文的机制指标是各 stage 线性/SVM 可分性，任务指标应至少看 COCO AP、APS 与 ImageNet 线性 top-1。可将失败阈值设为：完整方案相对 MoCo v2 的 COCO AP 增益低于 0.5，或 P3/P4 线性精度未提升 3 个百分点；若 AP 上升但 top-1 比仅 GLC 下降超过 2 个百分点，则 MLS 权重过强，重现了论文所揭示的分类—检测取舍。对于小目标场景，还应要求 APS 有正增益；论文在 Sparse R-CNN 上的 APS 增幅明显高于总体 AP，若 YOLO-Agent 的 APS 无变化，说明局部块尺度或接入层级不匹配。

## 优点

- 模块与检测器需求一一对应：MLS 服务多尺度金字塔，GLC 服务局部实例判别。
- 不依赖框标注或候选框生成，预训练完成后不增加检测推理成本。
- 在检测、分割、姿态和分类间展示了较均衡的迁移能力。

## 局限

- 四层 projection head、独立队列和九块局部视图显著增加预训练显存与计算。
- MLS 单独使用会降低最终层分类表示，需要 GLC 抵消，层级权重不能直接照搬到不同 backbone。
- 局部块来自规则 jigsaw，并不保证每块对应完整物体，复杂场景中可能把背景共现当作实例知识。

## 评分

- **创新性：8/10**——以深度层级和空间粒度两条轴重构检测友好的对比学习。
- **实验充分性：9/10**——任务范围广，且消融直接验证中间层判别性。
- **可复现性：8/10**——基于 MoCo v2、公式清楚，但预训练资源成本较高。
- **对 YOLO-Agent 价值：8.5/10**——适合作为多尺度 backbone 的无标注预训练方案。
