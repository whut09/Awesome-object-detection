---
title: "UniT: Unified Knowledge Transfer for Any-Shot Object Detection and Segmentation"
description: "解析 UniT 如何把弱监督检测器到全监督分类、回归和分割头的残差映射，借助语言与视觉相似度迁移到零/少样本新类。"
tags: ["CVPR 2021", "object detection", "few-shot", "weak supervision", "instance segmentation"]
---

# UniT: Unified Knowledge Transfer for Any-Shot Object Detection and Segmentation

- **论文页面**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Khandelwal_UniT_Unified_Knowledge_Transfer_for_Any-Shot_Object_Detection_and_Segmentation_CVPR_2021_paper.html)
- **官方 PDF**：[CVPR 2021 Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Khandelwal_UniT_Unified_Knowledge_Transfer_for_Any-Shot_Object_Detection_and_Segmentation_CVPR_2021_paper.pdf)
- **官方代码**：[ubc-vision/UniT](https://github.com/ubc-vision/UniT)（论文脚注明确给出）

## 一句话总结

UniT 让 OICR 弱检测器先为所有类别产生 proposal 级分数，再学习 base 类从弱分类器到 Faster/Mask R-CNN 分类、框回归和掩码头的残差映射，并用 GloVe 语言相似度×proposal 视觉相似度把这些映射加权迁移到 novel 类。

## 研究背景与问题

弱监督检测只有图像级标签，零/少样本检测却通常假设 base 类拥有大量框标注、novel 类只有 0–k 个实例；两类方法长期分开设计。UniT 的输入同时包含所有类别的图像级标签、base 类丰富的框/掩码，以及 novel 类可选的 k 个实例。它不直接复制 base 分类权重，而是学习“弱检测器→强分类/回归/分割器”的结构化修正，再按每个 RoI 的语义关系迁移，因此同一模型可以覆盖 `k=0`、few-shot detection 和 few-shot instance segmentation。

## 方法总览

图像先经 backbone 与 RPN/MCG 得到 region proposals，RoIAlign 生成特征 `z`。弱分支使用 **OICR** 的多个 refinement module 输出所有类的 proposal 分数；监督分支在 base 类上学习三个残差：分类 logits 修正 `ΔW_cls`、框回归器 `W_reg`、掩码头 `W_seg`。对 novel 类，**Lingual Similarity** 由 300 维 GloVe 类名向量内积得到，**Visual Similarity** 由当前 RoI 的 OICR base 类归一化分数得到；二者逐元素相乘并 softmax 成 proposal-aware 矩阵 `S(z)`，再对 base 残差加权求和。若有 k-shot 标注，最后叠加 novel 类直接适配项；若 `k=0`，只使用迁移项。

## 方法详解

### 1. Base 类的弱→强残差

Base 分类器不是从零学习，而是在弱检测 logits 上增加零初始化残差：`f_cls_base(z)=f_weak_base(z)+f_Δcls_base(z)`；框坐标由 proposal `r_box` 加类别回归量，掩码则由 Mask R-CNN 分支直接学习。这个设计把可迁移知识定义为“弱监督还缺什么”，而不是整个检测器参数，因此 novel 类已有的弱语义分数不会被 base 类覆盖。

### 2. Proposal-aware 相似度迁移

语言矩阵 `S_lin[n,b]` 是 novel/base 类名 GloVe 向量内积，提供稳定的类别先验；视觉项 `S_vis(z)` 使用当前 proposal 在 base 类上的 OICR 概率，随 RoI 内容变化。`S(z)=softmax(S_lin ⊙ S_vis(z))` 后，novel 分类器由弱 logits、`S(z)^T` 加权的 base 分类残差和可选 novel 直接残差相加；框回归和掩码头也分别对 base regressors/segmentors 做同样加权。由此，motorbike proposal 可以更多借用 bicycle 的定位修正，而非对所有 novel 类使用固定映射。

### 3. 两阶段训练

Base training 联合优化 Faster/Mask R-CNN 损失与 OICR 弱监督损失，权重 `α=1`，并可同时训练 RPN。Fine-tuning 阶段只用 `L_rcnn` 更新公式中的 novel 直接适配项；`k=0` 时不需微调。OICR 的多个 refinement module 取平均 logits 后 softmax，确保弱分支为每个 proposal 提供可用于视觉相似度的类别分布。

## 实验与证据

- **半监督设定**：COCO 80 类中把 VOC 20 类作为 base、其余 60 类作为 novel；所有类有图像级标签，只有 base 有丰富实例标注。零样本实验使用 VGG-CNN-F 与 MCG proposals，指标为 novel AP50。
- **零样本对比**：LSDA 4.6、LSDA+MIL 5.9、DOCK 14.4 AP50，UniT 达 16.7；全监督上界为 25.2。UniT 未使用 DOCK 额外依赖的 VOC、Visual Genome、SUN 相似性数据，却仍高 2.3 AP50。
- **半监督 few-shot**：Inception-ResNet-v2 下，novel 实例数 `k={12,33,55,76,96}`，UniT 分别为 14.7/17.4/19.3/20.9/22.1 mAP，均高于 NOTE-RCNN 的 14.1/14.2/17.1/19.8/19.9。
- **标准 few-shot**：VOC 三个 novel split 上，UniT 即使 `k=0` 也取得 75.6/56.9/67.5 AP50；COCO novel 类 `k=0` 为 18.9 AP、36.1 AP50，`k=10` 为 21.7 AP、40.8 AP50，明显高于 Wang 等方法的 10.0 AP。
- **分割与消融**：COCO `k=10` 时 UniT box/mask AP 为 22.8/20.5。消融表明视觉+语言相似度带来 +1.4 AP50，迁移 base regressors 再增加 +7 AP50，迁移 base segmentations 令 mask AP50 再增 +7.5；真正的大头来自结构化定位/掩码迁移，而非只做语义分类迁移。
- **等预算证据**：VOC split 1、10 个标注预算下，100% 实例标注的 Wang 等方法为 52.8 AP50；UniT 使用 100% 图像级弱标注、0 个框达到 59.0，优于 50% 弱标注+50% 实例标注的 54.0。

## 对 YOLO-Agent 的启发

对新增类别，YOLO-Agent 可把 base 类检测头相对弱分类器的“定位修正”抽成可迁移对象：先由开放词汇/图像级分类分支给候选框语义分数，再根据类名嵌入和候选局部视觉分布，混合已有类别的框回归或 mask prototype 适配器。这样 Agent 能在零框标注时给出初始检测器，在少量框到达后只微调 novel adapter，而不重训全部类别。

### 专属 Harness：新类的分类、回归与掩码迁移

- **对照组**：A 仅用 weak/OICR 式 novel 分类分数；B 加固定语言相似度迁移分类残差；C 加 proposal-aware 视觉×语言分类迁移；D 在 C 上迁移 base box regressors；E 对分割任务再迁移 base mask heads；F 为相同 k-shot 下直接微调整个 YOLO 头。
- **观测指标**：novel AP50/AP75、box IoU、mask AP50、base 类遗忘量、每类新增可训练参数，以及 `k=0/1/5/10` 曲线。
- **通过标准**：C 相对 B 提升至少 1 AP50；D 必须带来至少 3 AP50 或显著提高 AP75；分割任务 E 相对 D 的 mask AP50 至少提升 3；同时 base AP 下降不超过 1。
- **失败判断**：收益只出现在分类 AP50 而框 IoU/AP75 不升、相似类别权重与 proposal 内容无关、novel 微调导致 base 严重遗忘，或 D/E 不优于只迁移分类器，均说明未复现 UniT 的结构化知识转移。

复现数据切分必须保证 base 与 novel 类集合严格不交叠，同时保留“所有类别有图像级标签、只有 base 有丰富实例标注”的非对称监督。应可视化每个 novel proposal 的 `S(z)`，确认同一 novel 类在不同外观下会借用不同 base 类，而不是退化为固定 GloVe 最近邻。对 `k=0`、`k>0` 还要分开保存参数更新范围：零样本不得偷偷更新 novel 直接适配项，少样本阶段则不应重写 weak branch 和 base transfer 项，否则无法对应论文的两阶段训练。

最终表格还应分别报告 base 与 novel 指标，防止新类收益以旧类遗忘为代价。

并保留完整切分清单。

## 优点

- 用统一公式覆盖零样本、少样本检测和实例分割，监督量变化时不需换架构。
- 相似度是 proposal-aware 的，并同时迁移分类、回归和掩码知识。
- 等标注预算实验揭示图像级标签的成本效益，超出常规 few-shot 榜单比较。

## 局限

- 依赖 OICR 弱检测质量、GloVe 类名语义和 base/novel 可迁移性；视觉差异大的新类可能得到错误映射。
- 不同实验采用 VGG-CNN-F、Inception-ResNet-v2、ResNet 等不同协议，跨表绝对数值不能直接横比。
- 新类仍需充足图像级标签；它不是完全不看 novel 视觉数据的传统 zero-shot 方法。

## 评分

- **创新性：9/10**——将弱→强残差及分类/回归/掩码迁移统一起来。
- **实验充分性：9/10**——覆盖多监督量、多任务、标注预算和关键组件消融。
- **工程可迁移性：7.5/10**——思想适合 adapter 化，但现代 YOLO 中需重构弱分支和类别相关回归。
- **综合评分：8.5/10**。
