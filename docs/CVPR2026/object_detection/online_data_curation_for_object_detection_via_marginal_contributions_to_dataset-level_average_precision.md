---
title: "Online Data Curation for Object Detection via Marginal Contributions to Dataset-level Average Precision"
description: "解析 DetGain 如何估计单图对全局 mAP 的边际扰动，并用教师学生贡献差进行在线训练样本筛选。"
tags: ["CVPR 2026", "在线数据筛选", "DetGain", "Average Precision", "数据效率"]
---

# Online Data Curation for Object Detection via Marginal Contributions to Dataset-level Average Precision

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Online_Data_Curation_for_Object_Detection_via_Marginal_Contributions_to_CVPR_2026_paper.html)  
**官方代码**: 未发现论文声明的官方代码  
**任务**: 目标检测在线数据筛选 / 训练采样

## 一句话总结

DetGain 不再用易受分类、回归和匹配策略影响的训练 loss 衡量样本价值，而是近似一张图的 TP/FP 插入后对全数据集 mAP 的边际变化，再按“教师 DetGain−学生 DetGain”从增强 super-batch 中挑选当前最值得学习的 20%。

## 研究背景与问题

分类任务可用 teacher loss 与 student loss 的差表示 learnability，但检测 loss 混合分类、框回归、centerness、RPN 采样或 Hungarian matching；不同架构的权重尺度完全不同，同一图在相邻迭代也会剧烈波动。更关键的是，训练 loss 与最终按全数据排序计算的 AP 并不一致，难以作为跨 Faster R-CNN、FCOS、ATSS、Deformable DETR 的统一采样指标。

DetGain 把不可微的 AP 从优化目标移到数据选择层：模型仍按原 loss 训练，但每次先加载更大的 super-batch，用近似全局 AP 贡献给图片排序，只对 top-b 反向传播。这使采样器和检测器内部结构解耦。

## 方法总览

对模型 `f` 和参考检测集合 `D`，单图边际贡献定义为 `δmAP(x)=mAP(D∪{x})-mAP(D)`；教师学生差 `sDG=δmAPteacher-δmAPstudent` 越大，表示教师能从该图产生更有利于全局排序的 TP/FP，而学生尚未掌握。直接重跑 COCO evaluator 太慢，因此论文在 score-threshold 域建模 TP/FP 分数分布，解析计算插入一个检测对 PR 曲线的影响，再跨类别和 IoU 阈值聚合到图级。

## 方法详解

对阈值 `u`，当前 TP/FP 数量由各自 CDF 得到，precision 和 recall 随阈值积分形成 AP。插入一个分数为 `s` 的 TP，会增加自身召回点并改变其后已有 TP 的 precision；插入 FP 不改变 recall，只降低后续 precision。DetGain 对一图所有检测先做标准一对一匹配，在 COCO 的 `0.50:0.05:0.95` IoU 阈值上判断 TP/FP，再平均类别、阈值贡献。稀有类别因 `1/TGTc` 归一化自然获得更大边际影响。

为了在线计算，作者不用每轮拟合真实 score density，而采用统一分布及固定 TP:FP 先验，得到模型无关闭式近似；模拟 super-batch 中该近似与 Beta 拟合版本的平均 Spearman 排名相关约 `0.94`。单纯反复抽高 DetGain 图会收缩到狭窄子空间，因此先对 super-batch 做颜色扰动、仿射、噪声、copy-paste 等强增强，再让 sampler 过滤低质量增强。

DetGain 评价的不是一张图独立的“难度”，而是它插入当前全局排序后的边际作用。同样一个高分 TP，若属于已有大量正确实例的头部类，对每类平均 AP 的影响较小；若属于真值数很少的类别，归一化后贡献更大。高分 FP 则会污染其后大量检测的 precision，因此受到明显负贡献。这样的全局视角解释了为什么 per-image AP 和 classification entropy 即使看似合理，仍无法稳定替代 DetGain。

teacher-student 差也避免了只选择“teacher 最喜欢的干净图”。若 teacher 与 student 都能很好处理一张图，两者边际贡献相近，样本不会长期占据队列；若二者都失败，teacher 也无法提供可学习优势。只有 teacher 对全局 AP 的贡献显著高于 student 时，图像才被视为剩余知识。这使采样顺序随训练状态变化，而不是形成静态数据质量榜单。

统一分布近似的价值主要在排序而非精确数值。在线筛选只关心 super-batch 内谁进入前 20%，所以与拟合分布保持高 Spearman 相关比逐图 `δmAP` 绝对误差更重要。论文以约 `0.94` 的排名相关支持该简化，但部署到类别数、每图候选数或 TP:FP 比例明显不同的数据集时仍应重新验证。

标注噪声实验覆盖假框、漏框、坐标扰动与类别交换，这四种错误分别改变 FP 数、召回、IoU 阈值归属和类别排序。DetGain 仍优于 uniform 与 loss-learnability，说明其全局 AP 信号没有被某一种噪声类型独占。尤其当噪声接近极高比例时，hard loss 会优先抽取被破坏样本，而教师学生边际差更可能把它们降权。

因此评分缓存必须随学生状态定期刷新，长期复用旧排名会失去在线方法的主要优势。

刷新周期本身也需要纳入训练成本消融。

## 实验与证据

- COCO 2017 主实验覆盖 Faster R-CNN、ATSS、FCOS、GFL、VFNet、Deformable DETR，学生均为 ResNet-50；CNN teacher 多用 ResNet-152，Deformable DETR teacher 用 ResNet-101。
- 六个检测器完整方案平均提升约 `2 AP`。Faster R-CNN 从 `37.5` 到 `40.0`，FCOS 从 `38.2` 到 `41.0`，ATSS 从 `39.2` 到 `41.5`，Deformable DETR 从 `46.6` 到 `48.9`；强增强单独使用几乎无益或下降。
- 关键消融：Faster R-CNN 无增强无筛选 val AP `37.4`；仅在线筛选训练 AP 从 `44.6` 升到 `50.3`，但 val 降到 `37.3`；增强+筛选获得 `39.9`，直接证明两者用于“扩展空间”和“聚焦样本”的互补性。
- 统一在线框架下，DetGain 在 Faster R-CNN/FCOS/ATSS 达 `40.0/40.9/41.6 AP`，高于 Image-AP 的 `38.3/39.4/40.0` 和 loss-learnability 的 `38.9/38.1/40.4`。
- teacher 越强收益略增，但同架构 teacher 仍有效。FCOS-R50 基线 `38.5`，DetGain+R50 teacher 为 `40.8`；与 CrossKD 组合达到 `42.2`，说明样本级筛选和特征蒸馏互补。论文还验证假框、漏框、框扰动、换类噪声下的稳定性。

## 对 YOLO-Agent 的启发

YOLO-Agent 可把 DetGain 作为动态数据调度器：保持 YOLO loss、assigner 和训练日程不动，每轮只让 teacher/student 对增强候选前向，依据对全局 AP 排序的预计贡献决定哪些图进入反传。与按最高 loss 硬挖不同，它能跨 anchor-free、one-stage 版本共享同一策略，也能自然提高稀有类别图的优先级。

**Harness**：在固定训练预算下比较 uniform、hard-loss、Image-AP、teacher-student loss gap、DetGain，以及 DetGain 无增强组；super-batch 设 80、回传 16，并保持总反向次数一致。观测总体 AP、APs/m/l、每类采样频率、训练/验证差、评分前向耗时和三个种子方差。通过要求为平均 AP 至少 `+1.0`，三个 YOLO 尺度中至少两个稳定增益，验证 AP 提升而训练—验证 gap 不扩大超过 `1` 点；若只提高训练 AP、稀有类被过采样后 precision 崩溃，或评分额外耗时超过训练总时长 `30%`，则不通过。

## 优点

- 采样信号直接对齐 dataset-level AP，跨检测器和 loss 配方可比。
- 不改模型、loss、推理过程，可与知识蒸馏叠加。
- 对增强过拟合、teacher 容量和标注噪声给出针对性证据。

## 局限

- 每轮需要 teacher 和 student 对 super-batch 额外前向，训练时间上升。
- TP/FP 独立相加与统一分布都是近似，密集重叠检测可能违反假设。
- 强增强策略较朴素，方法效果仍依赖候选数据空间是否包含真正有益样本。

## 评分

- **创新性**: ★★★★★
- **证据强度**: ★★★★★
- **工程可用性**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★★
