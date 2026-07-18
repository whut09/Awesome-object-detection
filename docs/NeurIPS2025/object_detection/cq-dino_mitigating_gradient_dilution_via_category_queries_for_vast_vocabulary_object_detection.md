---
title: "CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection"
description: "CQ-DINO 将万类检测的固定分类头改写为对象查询与可学习类别查询的对比匹配，并用图像引导的 Top-K 类别选择缓解正样本与困难负样本梯度稀释。"
tags: ["NeurIPS 2025", "目标检测", "V3Det", "Category Query", "DINO", "大词表检测"]
---

# CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection

**会议**：NeurIPS 2025  
**论文**：[NeurIPS 官方页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/21168020249eccddcc1736ec79bd096f-Abstract-Conference.html)  
**官方代码**：[FireRedTeam/CQ-DINO](https://github.com/FireRedTeam/CQ-DINO)  
**任务**：13K 级大词表目标检测

## 一句话总结

CQ-DINO 不再让每个 object query 对 13,204 个独立分类 logit 承受海量负类梯度，而是把类别表示成可学习的 **Category Queries**，先由 **Image-Guided Query Selection** 从整套类别中选出每图最相关的 Top-K，再在 GroundingDINO 式跨模态解码器中完成对象—类别对比匹配。

## 研究背景与问题

论文针对 V3Det 这类超过一万类别的闭集检测，明确提出两种优化障碍。其一是 **Positive Gradient Dilution**：正类梯度被随类别数线性增长的负类梯度总量淹没；其二是 **Hard Negative Gradient Dilution**：真正有区分价值的相似负类，又被数量更大的简单负类稀释。作者在训练初期测得，V3Det 上 Focal Loss 的正负梯度比仍约为 0.5，而 COCO 约为 1.0，说明仅调 Focal Loss 不能根治万类分类头的结构性失衡。

现有文本提示检测器虽然能把分类改成视觉—语言对齐，但面对上万类别往往需要拆分词表、多次推理；生成类别名称的方法又难以控制“car/vehicle/sedan”这类粒度。CQ-DINO 因而选择闭集但可扩展的路线：类别仍受数据集词表约束，却不再固化为巨型 FFN 分类矩阵。

## 方法总览

输入图像先经 Swin 图像编码器得到 `Fimg`，全部类别由 OpenCLIP 文本嵌入初始化为 `Qcat`。类别查询可通过 V3Det 的树结构显式注入父子关系，或在无层级数据集上用 self-attention 学习隐式相关性。随后 `Qcat` 作为 query、图像特征作为 key/value 进入两层多头 cross-attention，按激活值选出 `C'` 个类别；V3Det 默认 100 个、COCO 默认 30 个。选中类别和图像特征进入 **Feature Enhancer**，再经 **Language-Guided Query Selection** 生成 object queries，最终由 **Cross-Modality Decoder** 输出框，并以对象查询和类别查询的对比相似度完成分类。

## 方法详解

**Image-Guided Query Selection** 是核心。它把负类空间从 `C` 压缩到 `C'`，理论上的正负梯度重平衡倍数约为 `C/C'`；当类别超过一万而 `C'=100` 时约有百倍改善。被保留的负类并非随机采样，而是与当前图像响应较高的类别，因此同时形成隐式困难负样本挖掘。选择器由 **Asymmetric Loss** 监督，检测阶段则使用选中查询，避免对完整词表逐类解码。

对 V3Det，**Hierarchical Tree Construction** 从叶节点向上融合语义：父节点查询由自身查询与子节点均值加权得到，权重 `αv` 随子节点数量自适应变化，默认超参数 `w=0.3`。若某子类出现在标注中，其父类会从分类损失中屏蔽，避免“car 为正而 vehicle 被当负类”的层级冲突。COCO 没有同样的官方类别树，因此采用 8 头 self-attention 更新类别查询。

训练目标包含 Asymmetric classification loss、对象—类别 contrastive Focal Loss、L1 框损失和 GIoU，权重依次为 1、1、5、2，并沿用 GroundingDINO 的 Hungarian matching。训练分两段：先用 10 个 epoch 预训练类别查询、图像编码器和选择器以建立类别召回，再微调整条检测链；作者承认这种两阶段协同可能不如理想的端到端联合优化。

## 实验与证据

主实验使用 V3Det（13,204 类，183K 训练图、30K 验证图）和 COCO（80 类，118K/5K）。V3Det 上 Swin-B-22k 的 CQ-DINO 达到 **52.3 AP / 57.7 AP50 / 54.6 AP75**，超过 Prova 的 50.3 AP；Swin-L 达 53.0 AP。COCO 上以 Swin-L、24 epoch 得到 **58.5 AP**，与 DINO、Rank-DETR 等强 DETR 基线竞争。

关键消融显示：无关系编码且无图像选择时仅 47.3 AP、0.7 FPS；只加入选择器达到 51.1 AP、10.8 FPS；再加入层级树得到 **52.3 AP、83.3 ARC、10.6 FPS**。self-attention 在 V3Det 仅到 51.3 AP 且 ARC 降为 75.5，说明 13K 类关系更适合显式树；在 COCO 上它仍带来 58.3→58.5 AP。查询数从 50/100/200 变化时，AP 为 51.8/52.3/52.2，证明更高类别召回不必然转化为更高精度。调优 DINO 的 Focal Loss 最好可从 43.4 提到 47.4 AP，仍落后 CQ-DINO 4.9 点，且最优参数不跨骨干稳定。

扩展性测试同样有实际含义：在 A100-40G、Swin-B-22k 和单张 800×1333 图像下，CQ-DINO 每类参数约 0.8K、CUDA 内存约 2.7KB，可支持约 130K 类；DINO 对应为 2.1K、8.9KB 和 100K 类。失败分析则指出，类别召回低于 83.3% 的集合中，CQ-DINO 对稀有类 AP 为 20.5、common 类为 32.1；小、中、大目标分别为 14.1、23.1、38.9。这表明类别检索并没有消除长尾和尺度问题，只是相对经过 Focal Loss 调优的 DINO 仍保持优势。

开放式生成基线也反衬了闭集类别查询的必要性：GenerateU 和 ChatRex 在 V3Det 零样本匹配中仅有 0.4 与 1.3 AP，微调后的 GenerateU 也只有 21.8 AP。作者据此认为，大词表检测不仅需要语义覆盖，还必须精确服从数据集规定的类别粒度。

## 对 YOLO-Agent 的启发

YOLO-Agent 可把“完整类别头”与“候选类别检索”拆成两个可独立搜索的阶段：先用轻量图像摘要预测候选类，再让检测头只计算候选子空间。若数据存在 taxonomy，应把父子屏蔽和层级查询融合设为数据集能力，而不是把所有数据统一套 self-attention。还应同时记录候选类别召回、最终 AP 与真实吞吐，防止选择器只提升速度却漏掉稀有类。

### Harness

对照组设为同一 YOLO 骨干、同一训练轮数的完整 `C` 类 sigmoid/Focal 分类头；实验组依次加入“Top-K 类别选择”“Top-K+层级查询”，参数量差异控制在 2% 内。观测 overall mAP、rare/common AP、候选类别召回 ARC、每图分类计算量、batch=1 FPS 和峰值显存。通过阈值建议为：在 ARC≥85% 时 mAP 至少提升 1.0 点，或 mAP 不降超过 0.3 点且 FPS 提升至少 30%；若 rare AP 下降超过 1.0、ARC 低于 80%，或增大 K 仍不能恢复漏检，则判定候选检索不适合当前词表。

## 优点

- 从梯度来源解释大词表检测失败，而不是只报告显存不足。
- 类别选择同时改善优化、困难负样本质量和推理复杂度。
- V3Det 层级屏蔽处理了父子标签之间真实存在的监督冲突。
- 最高可测试到 130K 类，展示了比 DINO 更明确的扩展边界。

## 局限

- 两阶段训练增加流程复杂度，选择器与最终检测器不是完全联合收敛。
- 83.3 ARC 仍意味着部分真类不会进入解码器，稀有类和小目标尤其脆弱。
- 低召回类别中 CQ-DINO 的 rare AP 仅 20.5，小目标 AP 仅 14.1。
- 方法仍是预定义词表检测，不能直接等同于开放词汇或开放世界识别。

## 评分

- **创新性**：9/10
- **实验充分度**：9/10
- **工程可迁移性**：8/10
- **对 YOLO-Agent 的价值**：9/10
