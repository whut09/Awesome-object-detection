---
title: "InstaGen: Enhancing Object Detection by Training on Synthetic Dataset"
description: "原创中文详解 InstaGen 如何把 Stable Diffusion、实例级 grounding head 与新类自训练组合成检测数据合成器。"
tags: ["CVPR 2024", "目标检测", "合成数据", "开放词汇检测"]
---

# InstaGen: Enhancing Object Detection by Training on Synthetic Dataset

**论文**：[CVPR 官方页面](https://openaccess.thecvf.com/content/CVPR2024/html/Feng_InstaGen_Enhancing_Object_Detection_by_Training_on_Synthetic_Dataset_CVPR_2024_paper.html)  
**官方代码**：[fcjian/InstaGen](https://github.com/fcjian/InstaGen)

## 一句话总结

InstaGen 把 Stable Diffusion v1.4 微调成复杂场景图像生成器，再接入类似 GroundingDINO 的 instance-level grounding head，使生成过程同时输出图像和实例框，最终用无需人工标注的合成数据扩展检测类别并补足稀缺真实数据。

## 研究背景与问题

扩充检测数据通常卡在框标注而非图像数量：扩散模型能按类别造图，却常生成孤立大物体、简单背景，也不天然给出可靠边界框。开放词汇检测依赖 CLIP 等外部语义对齐，而长尾或数据稀缺场景更希望把“生成能力”直接转成可训练的检测样本。论文因此把问题拆为图像域逼真度、实例定位以及未见类定位三层。

## 方法总览

流程先用真实检测集的随机裁剪与其中类别词微调 SDM；生成时读取扩散网络在 `t=1` 的多尺度区域特征。Grounding head 接收这些视觉特征和类别文本嵌入，输出类别分数与边界框。基础类由现成 Mask R-CNN 自动监督，新类则由 grounding head 的 EMA teacher 产生伪框进行自训练。最后将 `D_real + D_syn` 用于标准 Faster R-CNN 训练。

## 方法详解

**Image Synthesizer**不直接用整张多目标图微调 SDM，而是随机裁剪真实检测图，按裁剪内出现的类别构造文本；同类多个实例只写一次。优化仍是扩散噪声预测平方误差。该策略让生成图包含更小、更密集、遮挡更明显的实例，缩小“干净商品图”与真实检测场景之间的差距。

**Instance Grounding Head（G-head）**包含四部分：3×3 channel-compression 把特征压到 256 维；六层 feature enhancer 以 deformable self-attention、文本 self-attention 和双向跨模态注意力融合；language-guided query selection 按图文相似度选 top-N anchor，采用 DINO mixed query；六层 cross-modality decoder 继续分类与框细化。900 个 queries 通过与文本特征点积得到 sigmoid 分类分数，MLP 回归框。

基础类训练由 Mask R-CNN 在合成图上预测框，并用阈值 α 过滤，之后以二分匹配、Focal、L1 和 GIoU 训练 G-head。**Novel-category self-training**中，EMA teacher 在不启用 dropout 时给新类伪框，学生在 enhancer 与 decoder 内启用 dropout 接收扰动；阈值 β 清除低置信框，总损失为 `L_base+L_novel`。最终检测器仍使用普通 RPN 与检测头损失，因此合成器与下游架构解耦。

## 实验与证据

开放词汇 COCO 将 48 类设为 base、17 类设为 novel；G-head 每类每 epoch 用 1,250 张合成图，检测器每类用 3,000 张，评价 AP50。InstaGen 的 all/base/novel AP50 为 **52.3/55.8/42.3**；novel 超过 EdaDet 37.8、BARON 34.0。数据稀缺实验在 COCO 10%、25%、50%、75%、100% 子集上分别从 23.3/29.5/34.1/36.1/37.5 提升到 **28.5/32.6/35.8/37.3/38.5 AP**。跨数据集 COCO-base→Object365/LVIS 达到 11.4/23.3 AP50，且不需要含目标类别的额外真实数据。

组件消融显示：仅 G-head 为 novel 37.1；加 self-training 为 40.3；再微调 SDM 为 42.3。每类生成图从 1,000 增至 3,000 时 novel AP50 从 39.7 单调升至 42.3。α=0.8 最优；β=0.3 保留大量错误框，novel 仅 26.9，β=0.4 达到 42.3，β=0.5 又降至 39.2。

## 对 YOLO-Agent 的启发

InstaGen 更适合成为 YOLO 的离线数据工厂，而非塞进推理图：先为薄弱类别合成“图像+框”，通过质量门控后与真实数据按比例混训，并对真实/合成来源分别记录梯度和 AP。**Harness**：对照组设为真实数据 YOLO、原始 SD+现成 detector 自动框、G-head、G-head+self-training、完整 InstaGen；观测每类伪框 precision/recall、合成图背景多样性、novel AP50、真实域 mAP50-95 与误检率。完整方案在 novel AP50 至少提升 3 点、真实 base AP 下降不超过 1 点、抽检伪框 precision≥70% 时通过；若收益仅来自增大训练步数，或稀有类生成失败导致类别分布更偏，则失败。

## 优点

- 生成图和实例标注在同一模型内产生，减少二次人工框选。
- 基础类监督与新类自训练闭环清晰，可扩展任意文本类别。
- 在开放词汇、数据稀缺、跨数据集三种协议上均有证据。

## 局限

- 扩散生成慢，3,000 张/类的规模带来显著离线成本。
- 论文承认合成图仍可能缺乏真实遮挡、形变和长尾类别覆盖。
- 结果对 β 很敏感，低质量伪框会造成灾难性性能下降。

## 评分

- **创新性**：9/10
- **实验充分度**：9/10
- **工程可迁移性**：7/10
- **YOLO-Agent 参考价值**：9/10
