---
title: "LMM-Det: Make Large Multimodal Models Excel in Object Detection"
description: "LMM-Det 不外挂检测头，而以伪标签补全、坐标置信度文本化和逐类别推理提升大多模态模型的检测召回。"
tags: ["ICCV 2025", "目标检测", "大多模态模型", "LMM-Det", "伪标签", "指令微调"]
---

# LMM-Det: Make Large Multimodal Models Excel in Object Detection

**论文**: [arXiv](https://arxiv.org/abs/2507.18300)  
**代码**: [360CVGroup/LMM-Det](https://github.com/360CVGroup/LMM-Det)  
**任务**: 无专用检测模块的大多模态模型目标检测

## 一句话总结

LMM-Det 用 OWLv2-ViT、线性 projector 和 Vicuna-1.5-7B 保持标准视觉语言架构，通过 Salience-DETR 伪标签补全 COCO/Object365 框分布、直接生成坐标与置信度 token，并在推理时逐类别多轮询问来把召回率拉到可用水平。

## 研究背景与问题

LMM 在描述、VQA 和视觉指代上表现强，但直接检测时 AP 远低于专用模型。作者先系统检查 LLaVA-7B：COCO 零样本仅 0.2 AP；加入 COCO、Object365 并把分辨率从 336 提到 644 后可到 38.7 AP，仍低于同数据规模的 specialist detector 49.2 AP。可视化发现许多所谓“误检”其实是 COCO 未完整标注的真实物体，预测框数量又紧贴训练集平均每图约 7 个框，因此核心瓶颈被定位为训练分布导致的提前停止和低召回，而非 LMM 完全不会定位。

## 方法总览

架构由 1008×1008 的 OWLv2-ViT visual encoder、逐 token 线性 VL projector、最大序列长度 16,000 的 Vicuna-1.5-7B 构成，没有 region proposer 或外接检测器。训练数据侧先用 Salience-DETR 生成高质量 pseudo labels，与原真值经 NMS 合并，再把类别、坐标和置信度组织成多轮 class-specific conversations。推理不一次生成全图所有框，而是针对每个类别独立提问，汇总各轮输出作为 proposals。

## 方法详解

Data Distribution Adjustment 有三步。第一，Salience-DETR 为不完整标注补充候选，真值置信度设为 1，伪标签置信度沿用专用检测器。第二，伪标签与真值通过 NMS 去重，使每图目标数量从约 7 扩展到推理时平均约 31 个。第三，坐标和 confidence 不新增专用词表，而直接由原 tokenizer 生成普通 token；实验显示虽然序列更长，却比扩充 vocabulary 更准确。训练每图按可见类别生成正指令，并从不存在类别中采样等量负指令，顺序与框次序每 epoch 随机化。

Inference Optimization 为召回牺牲计算：一次只询问一个 category，模型输出该类全部框与分数，再遍历类别集合。训练也采用相同的 class-specific instruction，减少训练推理格式差异。默认 AP 计算不执行 NMS，因为 proposals 仍远少于 Salience-DETR 的 900 个；可视化时才使用 score 0.5 和 NMS 0.5。三阶段训练依次为 projector 图文对齐、Object365 检测预训练、COCO 重组指令微调；可选 Stage IV 再混入 665K LLaVA 数据恢复通用对话能力。

重组指令对每张图构造等量正负轮次：若图中有 n 个可见类别，就为这些类别生成 n 条要求输出框的正指令，再从其余类别随机抽 n 条“图中不存在该类”的负指令；COCO 与 Object365 的最大轮数分别限制为 80 和 365。每个 epoch 都随机打乱对话轮次和答案中的框顺序，以减轻固定序列过拟合。视觉 token 不做压缩而是一对一送入线性 projector，配合 1008 分辨率保留局部细节，这也是最大文本长度需要扩到 16K 的原因。

## 实验与证据

训练总计使用 595K 图文对和 1.86M 图像，在 6 个节点、每节点 8 张 H800 上耗时 176 小时。COCO zero-shot 中，两阶段 LMM-Det 达 24.5 AP/46.6 AR@100，高于 InternVL-2.5 的 11.8/27.5。完整微调后为 47.5 AP、66.5 AP50、51.1 AP75、34.7 APS、63.6 AR@100；不外挂 specialist 的 Griffon v2 为 38.5 AP，重训 LLaVA* 为 38.7。它仍低于 Salience-DETR 的 57.3 AP 和 RT-DETR 的 55.3，但已接近专用检测器区间。

关键消融以 CLIP-ViT LLaVA* 的 38.7 AP/50.5 AR 为基线，换 OWLv2-ViT 后为 42.1/51.3；加入 Data Distribution Adjustment 为 44.2/56.0；再加 Inference Optimization 达 47.5/63.6。逐类别推理贡献 3.3 AP 和 7.6 AR，直接指向召回瓶颈。代价是单图约 4.0 秒，论文结论也明确把推理延迟列为主要限制。

探索实验还给出分辨率与数据的交互：336 分辨率下，仅 COCO 为 16.0 AP，加入 Object365 反而是 15.6；升到 644 后，仅 COCO 为 17.7，而 COCO+Object365 跳到 38.7，说明大规模检测数据只有在视觉细节足够时才被利用。三阶段超参数分别训练 1、5、12 epoch，global batch 为 192、480、288，学习率为 1e-3、2e-5、2e-5，使用 AdamW、bf16 与 ZeRO-2/3。Stage IV 得到的 LMM-Det† 为 47.1 AP，几乎保留检测性能，同时恢复 caption 与 VQA 能力，证明分布扩展不必完全牺牲原多模态能力。

细粒度指标显示方法对小目标改善尤其明显：LMM-Det 的 APS 为 34.7，而 LLaVA* 只有 20.1；APM/APL 为 51.8/60.3。零样本版本 APS 也达到 15.4，超过其他无专用模块 LMM。另一方面，最终 47.5 AP 仍比 Salience-DETR 低 9.8 点，且伪标签正是由 Salience-DETR 生成，这意味着论文证明了“标准 LMM 架构可学检测”，却尚未摆脱强教师与高算力训练的依赖。

作者还比较坐标置信度的表示方式：直接输出普通数字 token 比扩展专用 vocabulary 更好，虽然前者增加序列长度，却不需要从头学习新 embedding。推理采样方面，beam search 的 beam=2 精度最好但更慢，最终系统约四秒一图。由于每个类别独立询问，负类别轮次也会产生开销；当类别集合扩大时，应先由 YOLO 或开放词汇分类器缩小候选类别，再调用 LMM 精查。

这一点决定了它更适合作为离线补检器。

## 对 YOLO-Agent 的启发

YOLO-Agent 可把 LMM-Det 视作“类别规划与长尾补召回器”，而不是替换实时 detector。Harness 对照组应有 YOLO 独立检测、LMM 一次性全类生成、LMM 逐类别生成、YOLO proposals 作为提示、两者分数融合；伪标签实验还需对照原 COCO 标签、Salience-DETR 补全和 YOLO 自训练补全。观测指标包含 AP、AR@100、平均 proposals、未标注真物体人工核验率、每类询问次数、首 token 到最终框延迟与 GPU 小时。通过阈值要求融合后 AR 至少 +5、AP 不下降、长尾类 AP 至少 +2，且候选复核成本可控。失败判断是单图总时延超过业务预算十倍、伪标签误差使 AP75 降 1 点以上，或增益仅来自遍历已知闭集类别，此时不得上线。

## 优点

- 先用数据规模、分辨率和框分布实验定位低召回根因，再提出对应改造。
- 架构不增加专用检测头，能直接检验 LMM 自身的坐标生成能力。
- DDA 与 INO 对 AP、AR 的逐项增益明确，附录还给出三阶段训练细节。

## 局限

- 逐类别生成造成约 4 秒单图延迟，远不能替代实时检测器。
- 训练消耗 48 张 H800、176 小时，且伪标签依赖强大的 Salience-DETR。
- 闭集类别循环适合 COCO，但开放词汇或数千类别时推理轮数会迅速膨胀。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★☆☆☆
- **YOLO-Agent 参考价值**: ★★★★★
