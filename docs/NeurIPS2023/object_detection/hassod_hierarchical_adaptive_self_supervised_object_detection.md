---
title: "HASSOD: Hierarchical Adaptive Self-Supervised Object Detection"
description: "解析一种以层次自适应聚类、对象组成层级预测和自适应 Mean Teacher 实现全自监督类别无关检测与实例分割的方法。"
tags:
  - self-supervised-learning
  - object-detection
  - instance-segmentation
  - hierarchical-clustering
  - mean-teacher
---

# HASSOD: Hierarchical Adaptive Self-Supervised Object Detection

## 一句话总结

该工作用冻结的 DINO ViT-B/8 将相邻图像块逐级合并为数量自适应的伪掩码，再依据掩码覆盖关系构建 whole–part–subpart 森林，并以带动态双分支权重的 Mean Teacher 训练 Cascade Mask R-CNN，从而在完全无人工对象标注的条件下同时学习类别无关检测、实例分割与组成层级。

## 研究背景与问题

既有 FreeSOLO、CutLER 等“发现—学习”路线通常先从自监督特征中提取一个或少数显著对象，再用这些伪标签训练检测器并反复重新生成标签；这会遗漏复杂场景中的大量小对象和部件，也不能说明车轮属于车辆、眼睛属于脸等组成关系，而且多轮离线自训练需要反复训练新模型，数据利用率和计算效率均受限制。

## 方法总览

方法分为严格区隔的两个阶段：第一阶段不训练检测器，只用冻结的 ImageNet-DINO ViT-B/8 从 MS-COCO 无标签图像生成掩码及层级伪标签；第二阶段以这些固定初始标签启动 ResNet-50 Cascade Mask R-CNN，并通过 student、EMA teacher 和自适应监督权重持续细化。推理时不再运行 ViT 聚类、CRF 或伪标签发现流程，而由训练后的检测器直接输出框、掩码、置信度以及 whole、part、subpart 层级。

## 方法详解

### 层次自适应聚类

输入图像缩放至 \(480\times480\)，DINO ViT-B/8 产生 \(60\times60\) 个 patch 特征，每个 \(8\times8\) patch 初始化为独立区域。算法计算空间相邻区域的余弦相似度，反复选择最相似的一对：若最高相似度低于 \(\theta_{\text{merge}}\) 则停止，否则合并区域，并以其中所有 patch 特征的均值更新区域表示及邻接相似度。

作者在一次聚类轨迹上记录 \(\theta_{\text{merge}}\in\{0.4,0.2,0.1\}\) 三个停止点并集成结果，使高阈值保留较多细粒度区域、低阈值形成较大整体。后处理包括 CRF、填洞，并剔除面积小于 100 像素、CRF 前后 IoU 小于 0.5，或覆盖超过两个图像角点的候选。

### 组成层级构造

当掩码 \(B\) 覆盖 \(A\) 超过 \(\theta_{\text{cover}}=90\%\)、反向覆盖不足 90%，且 \(B\) 是满足条件的最小掩码时，令 \(A\) 成为 \(B\) 的子节点。全部覆盖边形成森林：根节点标为 whole，根的直接子节点标为 part，其余后代标为 subpart。检测器在前景/背景分类头、框回归头和掩码头之外增加三分类层级预测头。

### 自适应 Mean Teacher

Student 同时接收两类检测损失：

\[
\mathcal L=
\alpha_{\text{label}}\mathcal L_{\text{label}\rightarrow\text{student}}
+\alpha_{\text{teacher}}\mathcal L_{\text{teacher}\rightarrow\text{student}},
\]

其中每个分支包含标准分类、框回归、掩码及层级分类监督。初始 burn-in 仅使用聚类伪标签；随后 teacher 对弱增强图像产生目标，student 在强增强版本上拟合，teacher 参数由 student 的 EMA 更新。余弦调度将学习率从 0.01 降至 0，将 \(\alpha_{\text{label}}\) 从 1.0 降至 0，同时把 \(\alpha_{\text{teacher}}\) 从 2.0 提至 3.0，使训练后期逐渐摆脱静态噪声标签。

## 实验与证据

训练使用 MS-COCO train 与 unlabeled split，共约 24 万张图像、40,000 次迭代、batch size 16，在 4 张 A100 上约需 20 小时；图像数仅为 CutLER 的 \(1/5\)，迭代数为其 \(1/12\)。评测采用类别无关、零样本协议：不在目标数据集微调，每图最多输出 1,000 个预测，并以 Box/Mask AR 为主，因为 AP 会把标注类别之外的合理对象误计为假阳性。

评测集包括 Objects365 v1+v2 的 80,000 张验证图像、LVIS v1.0 的 19,809 张验证图像，以及随机抽取的 50,000 张 SA-1B 图像。相较 CutLER，Box AR 在 Objects365 从 35.8 提至 39.0；LVIS 的 Box AR 从 23.6 提至 26.9、Mask AR 从 20.2 提至 22.5；SA-1B 的 Box AR 从 18.8 提至 29.0、Mask AR 从 17.0 提至 26.0。SA-1B 小对象 Mask AR 由 4.9 提至 12.9，说明层级伪标签尤其有助于发现整体对象内部的小部件。

消融中，三阈值集成伪标签达到 Mask AR 8.9，高于单阈值 0.4 的 7.8；训练基础模型为 20.2，加入层级预测升至 20.6，进一步加入 Mean Teacher 达到 22.1，再使用自适应目标达到 22.4。初始伪标签平均约 12.69 个/图，表明收益并非来自固定 top-k 显著区域。

## 对 YOLO-Agent 的启发

### 专属 Harness

可将该方法迁移为 YOLO-Agent 的伪标签策略实验，但应保持严格控制组：G0 使用固定数量的显著对象伪标签；G1 仅替换为三阈值层次自适应聚类；G2 在 G1 上增加 whole/part/subpart 预测头；G3 增加固定权重 Mean Teacher；G4 使用完整的 \(1\!\rightarrow\!0\) 与 \(2\!\rightarrow\!3\) 余弦权重调度。各组必须使用相同 COCO 图像、YOLO 骨干、增强、训练步数、预测上限和随机种子。

核心指标应为 LVIS 与 SA-1B 的 zero-shot Box AR、Mask AR、\(AR_S\)、每图伪标签数及三次运行标准差，而不能只看闭集 mAP。可证伪标准是：若 G1 相对 G0 未提高多对象场景 Mask AR，或 G2 相对 G1 的 Mask AR 增益低于论文对应的 0.4，且 \(AR_S\) 无改善，则层次标签并未被 YOLO 表示有效吸收；若 G4 在三次独立运行中不比 G3 至少提高 0.3 Mask AR，或出现 teacher 置信度上升而 AR 下降，则应判定自适应目标产生了确认偏差，而非稳定自增强。

## 优点

- 自适应决定每图对象数量，明显改善复杂场景覆盖率。
- 将几何覆盖关系转化为可训练的对象组成监督，兼顾性能、解释性与粒度控制。
- 单次连续 Mean Teacher 训练取代多轮离线重训，计算效率突出。
- 在三个不同标注分布的数据集上进行零样本评测，跨数据集证据较充分。
- 消融能够分别验证聚类、层级头、Mean Teacher 和动态权重的贡献。

## 局限

层级来自特征相似性与掩码包含关系，并不等同于人类语义：相似且重叠的多个实例可能被合并，组成部分差异很大的摩托车可能无法形成整体，文字边界也难以精确定位。初始标签中 subpart 仅约占 10%，导致深层节点预测不足；固定阈值、CRF 和面积过滤亦引入人工先验。模型仍与有监督 SAM 存在显著差距，且训练依赖离线 ViT 聚类和较高 GPU 成本。

## 评分

- **方法创新：9/10**——把对象数量自适应、组成树和 Mean Teacher 有机连接。
- **实验可信度：8.5/10**——协议合理、数据集广、包含三次运行与关键消融。
- **工程实用性：8/10**——推理阶段简洁，但初始伪标签生成成本较高。
- **综合评价：8.7/10**——是从“发现显著物体”迈向“发现完整场景及其组成结构”的重要工作。

官方论文页面：https://proceedings.neurips.cc/paper_files/paper/2023/hash/b9ecf4d84999a61783c360c3782e801e-Abstract-Conference.html

论文正文未提供官方作者代码仓库 URL。
