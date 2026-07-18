---
title: "[论文解读] DiPEx: Dispersing Prompt Expansion for Class-Agnostic Object Detection"
description: "详解 DiPEx 如何在 Grounding DINO 的提示嵌入球面上递归分裂高不确定提示，实现无框标注的类别无关检测。"
tags: ["NeurIPS 2024", "目标检测", "类别无关检测", "提示学习", "视觉语言模型"]
---

# DiPEx: Dispersing Prompt Expansion for Class-Agnostic Object Detection

**论文**：[官方论文页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2c2e95b75a10adbd2359f8ed5c0a38cd-Abstract-Conference.html)  
**代码**：[官方代码](https://github.com/jason-lim26/DiPEx)  
**发表**：NeurIPS 2024

## 一句话总结

DiPEx 从一个自训练的通用 Grounding DINO 提示出发，反复选择激活频率最高的宽语义提示，在单位超球面上旋转生成多个子提示，并用父子一致、子间分散和检测损失共同优化，最终得到一次推理即可覆盖广泛未知物体的提示树。

## 研究背景与问题

类别无关检测只关心“哪里有物体”，不要求输出语义类别。直接给 Grounding DINO 输入 `generic`、`object` 或 `items` 等通用词，虽然利用了视觉语言预训练知识，却会受单个词的语义覆盖边界限制；CoOp/CoCoOp 将提示变成可学习向量，但增加提示数量并不保证语义互补，多个 token 可能聚集在相同方向。

DiPEx 关注的是提示集合的覆盖而非单提示精度。论文把归一化提示嵌入视为单位超球面上的点，以夹角表示语义差异；若某提示在大量样本上频繁激活，说明它承载的语义过宽且不确定，应继续分裂为更细粒度的子提示。

## 方法总览

首先用 UNIVERSAL/CLASS-WIDE 文本查询的零样本检测结果构造伪框集合 `DPSL`，在冻结 Grounding DINO 的情况下，用 `Lbox`、`Lgiou` 和 focal `Lcls` 自训练根提示。第 `l` 轮从 `Pl` 中选激活频率最高的父提示 `v*l`，冻结并放入 `Pparent`，再围绕它初始化 `K` 个子提示。

子提示通过在随机二维轴平面上施加角偏移 `θk∈[-θ,θ]` 得到，既继承父提示中心方向，又产生不同检测结果。各子提示采用自身高置信预测在线更新伪标签；训练后计算全部提示两两夹角的 Maximum Angular Coverage（MAC），当 `αmax` 收敛时停止扩张。

## 方法详解

**球面旋转初始化。** 对父向量随机选择嵌入维度中的两个轴，构造只在该二维平面变化的旋转矩阵 `Rk`，令 `v(l+1,k)=v*l Rk`。这种初始化比独立随机 token 更保留父语义，也为子节点提供足够不同的起点。

**层次化优化。** `Lparent-child` 最大化子提示与固定父原型的余弦相似度，防止分裂后语义漂移；`Lchild-child` 通过温度 `τc` 拉大子提示两两夹角，减少重复覆盖。总损失再叠加 Grounding DINO 的框回归、GIoU 与分类项，因此提示分散必须同时服务于真实可定位区域。

**MAC 终止。** `αmax=max arccos(cos(vi,vj))` 描述当前提示词汇的最大角覆盖。实验中第二轮到第四轮的 MAC 由约 67.7° 增长至 75.95°，末轮增幅变小；论文以此作为无需继续支付整轮自训练成本的停止信号。

## 实验与证据

类别无关检测在 COCO（约 115K train、5K val、80 类）和 LVIS（约 100K train、19.8K val、1000 类）上将所有类别合并为一类。COCO 上 Grounding DINO `generic` 的 AR100/AP 为 44.1/28.3，CoOp 为 61.3/34.6，DiPEx 达 63.2/35.9，并在小物体 AR 上取得 39.2，高于 CoOp 的 36.4。

LVIS 的长尾场景更能体现提示覆盖：DiPEx 的 AR200/AP 为 48.4/15.2，CoOp 为 40.3/14.0，CoCoOp 为 40.7/13.6，SAM 的 AR200/AP 为 42.7/6.1。论文还在 COCO OOD-OD 中把 VOC 20 类视为已知，其余为未知；DiPEx 的未知 AR100 从 `generic` 提示的 43.3 提升到 59.9，未知 AP 从 12.5 提升到 15.7。

分析实验显示，单纯给 CoOp/CoCoOp 增加提示长度不会稳定提高结果，而 DiPEx 随提示数量增长持续受益。角覆盖热图中，父节点附近子提示仍保持较小局部夹角，但全局 MAC 逐轮扩大；激活频率的长尾分布在高频父提示被展开后得到缓解。这些证据分别支持“选择谁分裂”“如何分裂”和“何时停止”。

## 对 YOLO-Agent 的启发

DiPEx 更适合作为 YOLO-Agent 的开放集候选提议器，而不是替换闭集分类头。可冻结一个 Grounding DINO 教师，用扩张后的提示集合生成类别无关伪框，再训练 YOLO 的 objectness/box 分支；也可把提示激活频率用于主动挑选最缺语义覆盖的数据子集。

**Harness。** 对照组使用单一 `generic` 提示教师生成伪框，实验组使用相同教师、阈值和 NMS，仅替换为 DiPEx 提示集；下游 YOLO 保持完全相同的初始化与训练 recipe。观测伪框 AR100、重复框率、伪框精度、YOLO 在 COCO/LVIS 未见类别上的 class-agnostic AP 与单图教师耗时。通过标准：教师 AR100 提升至少 10 点、重复框率不增加超过 15%、下游 AP 提升至少 2 点；若 MAC 增长但 AR100 不升，或教师耗时超过单提示的 4 倍且下游 AP 增益不足 1 点，则视为失败。

## 优点

- 不使用人工框标注，通过 VLM 自训练获得广覆盖的类别无关提示。
- 提示树具有明确的父子语义约束，区别于无结构地堆叠 prompt token。
- COCO、LVIS 和混合 ID/OOD 场景均给出召回与精度证据。
- 扩张完成后可将全部提示放入一次推理，不必逐节点执行树搜索。

## 局限

- 每轮扩张都要在完整数据集上重新自训练，训练成本随层数显著增长。
- `τp`、`τc`、子提示数与提示长度需要人工调节，MAC 收敛也不是任务性能的充分条件。
- 初始 `DPSL` 来自 VLM 自身，教师漏检与偏差会沿提示树传播。
- 论文尚未在开放词汇和增量开放世界检测上完成系统验证。

## 评分

**8.4/10。** 把提示多样性转成可度量的球面层次扩张很有辨识度，实验召回提升扎实；主要代价是多轮自训练和超参数敏感性。
