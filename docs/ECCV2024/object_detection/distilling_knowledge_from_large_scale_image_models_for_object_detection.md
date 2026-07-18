---
title: "[论文解读] Distilling Knowledge from Large-Scale Image Models for Object Detection"
description: "解析 DLIM-Det 的 Frozen Teacher、Query Position Distill 与 Query Relation Distill。"
tags: ["ECCV 2024", "目标检测", "知识蒸馏", "DETR"]
---

# Distilling Knowledge from Large-Scale Image Models for Object Detection

**论文**：[官方论文页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11200_ECCV_2024_paper.php)  
**代码**：官方条目未提供代码链接  
**发表**：ECCV 2024

## 一句话总结

DLIM-Det 先冻结大规模预训练教师的 backbone、只微调 DETR 检测头，以保留泛化性并形成稀疏 query 布局，再用 Query Position Distill（QPD）和 Query Relation Distill（QRD）把教师 query 的框位置与自注意力关系传给小模型。

## 研究背景与问题

当教师从数亿参数扩到十亿参数时，普通特征模仿并不会自动获得更大收益：教师和学生的语义空间、容量及结构差距增大，FitNet、FGD、MGD 等方法可能停滞甚至损害学生。论文观察到，教师的构建方式也是蒸馏难度的一部分。若把大规模预训练 backbone 全量微调到 COCO，模型更贴合有限标注，却会削弱预训练带来的开放类别泛化，并让大量 object queries 密集围绕同一目标，学生必须同时学会选中一个 query、压低其余重复 query。

DLIM-Det 的切入点是保留大模型“未被下游数据过度改写”的知识。冻结 backbone 后，教师在 LVIS 标签上的平均召回更高；在 COCO val2017 上定义的 Query Density——每个真值周围 IoU 大于阈值的 query 数量——至少下降约三分之一。Swin-L 教师在 IoU 0.5 下由 38.1 降至 21.3，在 IoU 0.7 下由 15.2 降至 7.5，说明 query 空间更稀疏、更容易建立一对一对应。

## 方法总览

方法由两部分组成。**Frozen Teacher** 从大规模预训练模型出发，冻结 backbone，只训练检测头；训练完成后整个教师在蒸馏阶段固定。**Query Distillation** 不强迫学生复制高维 backbone 特征，而是先用 Hungarian Matching 对齐教师与学生 queries，再蒸馏 query 解码出的边界框位置和 decoder 自注意力矩阵。论文还让学生继承教师检测头参数，使跨结构 DETR 也能利用任务头知识。

数据流是：教师/学生分别产生 \(N\) 个 object queries；匹配模块根据类别、L1 框距离和 GIoU 求最优排列；QPD 对齐成对框；QRD 对齐成对 query 在自注意力中的关系；这些损失与学生原检测任务损失共同训练。

## 方法详解

设教师 query 为 \(q_i^{(t)}\)，学生 query 为 \(q_j^{(s)}\)，最优排列为

\[
\hat\sigma=\arg\min_{\sigma\in S_N}\sum_{i=1}^{N}L_{match}(q_i^{(t)},q_{\sigma(i)}^{(s)}),
\]

其中 \(S_N\) 是 \(N\) 个元素的全排列。匹配代价为

\[
L_{match}=\alpha_1L_{KL}(p^{(t)},p^{(s)})+\alpha_2L_1(b^{(t)},b^{(s)})+\alpha_3L_{GIoU}(b^{(t)},b^{(s)}),
\]

\(p\) 是类别概率，\(b=[x,y,w,h]\) 是 query 解码框；\(\alpha_1,\alpha_2,\alpha_3\) 分别设为 2、5、2。

QPD 使用

\[
L_{QPD}=\sum_i\left[\beta_1L_1(b_i^{(t)},b_{\hat\sigma(i)}^{(s)})+\beta_2L_{GIoU}(b_i^{(t)},b_{\hat\sigma(i)}^{(s)})\right],
\]

其中 \(\beta_1=5,\beta_2=2\)。它把教师所有采样 query 的空间位置变成额外回归监督，补足 DETR 一对一分配造成的稀疏真值监督：COCO 每图平均约 7 个实例，而 DINO 有 900 个 queries，正常回归损失只覆盖极少数 query。

QRD 直接利用 decoder 已有自注意力。令 \(A=[a_{ij}]\in\mathbb R^{N\times N}\) 为 query 两两注意力，则

\[
L_{QRD}=\frac1N\sum_i\sum_j|A_{ij}^{(t)}-A_{\hat\sigma(i),\hat\sigma(j)}^{(s)}|.
\]

总目标为 \(L_{overall}=\gamma_1L_{task}+\gamma_2L_{QPD}+\gamma_3L_{QRD}\)，权重依次为 1、2、1。位置描述“query 在哪里”，关系描述“哪个 query 应突出、哪些重复 query 应被抑制”，二者互补。

## 实验与证据

实验在 COCO 上使用 DINO，并覆盖 Swin Transformer、InternImage、ResNet；教师规模从约 200M 到 InternImage-Huge 的 1.08B。Swin-L 教师 58.1 mAP、Swin-T 学生 51.3 mAP，DLIM-Det 把学生提升到 54.4 mAP（+3.1），超过 DETRDistill 的 51.7，也优于 FitNet、MGD、FGD。Intern-Large 到 Intern-Tiny 时从 53.2 提升到 55.9（+2.7）；1.08B 教师到约 30M 学生仍达到 56.1（+2.9）。异构实验中 DINO-Swin-L 蒸馏 Deformable DETR-Swin-T，从 46.1 提至 49.6 mAP。

消融显示，InternImage 上单独 QPD、QRD 均从 53.2 提到 54.2，组合为 54.5；Swin 上分别为 51.6、51.8，组合 52.1。冻结教师即使与全量微调教师控制到近似精度，蒸馏学生仍高 1.3 mAP。六层 decoder 全部蒸馏最好；只蒸馏单层时浅层优于深层。教师 query 选择方面，900 个全用得到 54.2，随机采样 300 个为 54.5，纯前景或纯背景均为 54.2，证明两类 query 都含知识。与两阶段 Teacher Assistant 相比，DLIM-Det 在约一半训练时间下，Swin/Intern 学生分别高 0.7/0.2 mAP。

这些结果还揭示了教师“任务精度”与“可迁移知识”并非单调关系。同一训练日程下，冻结 Swin-L 与 Intern-L 教师分别比全量微调版本低约 1.5 和 2.0 mAP，但其 query 更分散、对 LVIS 非目标类别的召回更强；控制教师最终精度后，冻结版本依然产生更好的学生。换言之，DLIM-Det 不是用较弱教师缩小容量差距，而是借冻结策略保留预训练分布，并把这种分布转化成更稀疏的定位监督。

QPD 对背景 query 的利用尤其不同于只蒸馏正匹配结果的方法。教师中定位不佳、覆盖背景的 query 仍描述了“哪里不应成为目标”以及重复候选之间的空间结构；随机混合采样优于纯前景或纯背景，说明暗知识同时存在于被选中和被抑制的 query。QRD 再从自注意力中补充二者的相互制约，避免学生只逐框复制位置而忽略集合预测的去重机制。

## 对 YOLO-Agent 的启发

YOLO-Agent 若包含 transformer decoder 或 query-based 检测头，可在 decoder 输出处直接接入：冻结大教师 backbone，仅保留其检测头和自注意力；用类别、L1、GIoU 构造 teacher/student query 匹配，再对匹配框和 attention matrix 蒸馏。对照组应为全量微调教师+同一 QD、Frozen Teacher+仅 QPD、Frozen Teacher+仅 QRD、以及普通特征蒸馏，避免把收益误归因于教师精度。

指标除 AP/AP75 外应记录 Query Density、LVIS 类别 AR 和训练成本。论文中随机 300 queries 优于 900 queries，因此首轮可采用混合前景/背景随机采样。失败阈值可设为：Frozen Teacher 相对全量微调教师控制同等教师 AP 后，学生增益不足 0.5 AP；QPD+QRD 不高于最佳单模块；或 Query Density 未下降至少 20%。出现任一情况都说明当前教师没有形成论文所依赖的稀疏 query 特性。若 YOLO 版本完全没有 object query 与自注意力，QRD 缺乏对应实体，不应生搬硬套。

## 优点

- 同时处理“大教师怎么训练”和“DETR 知识怎么传”两个问题，模块关系明确。
- 蒸馏 query 位置与原生注意力关系，比跨尺度高维特征拟合更贴近 DETR 检测机制。
- 对十亿参数教师、异构 detector/backbone 及多种 backbone 都给出增益。

## 局限

- QPD、QRD 强依赖可匹配的 object queries 与 decoder 自注意力，不适用于纯卷积密集头。
- 冻结 backbone 会降低教师本身的 COCO AP，部署前需接受“更弱教师可能更好蒸馏”的非直观选择。
- Hungarian 匹配和多层 query 蒸馏增加训练计算，且 query 采样策略仍需调节。

## 评分

- **创新性：9/10**——把大模型教师构建与 DETR query 蒸馏统一起来。
- **实验充分性：9/10**——覆盖 200M 至 1.08B、同构与异构、组件与采样消融。
- **可复现性：8/10**——公式和权重清楚，但依赖大型预训练模型与高训练资源。
- **对 YOLO-Agent 价值：7.5/10**——对 query-based YOLO 很有价值，对纯密集头适配受限。
