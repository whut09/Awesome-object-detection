---
title: "[论文解读] OpenRSD：面向遥感图像开放提示词目标检测"
description: "解析 OpenRSD 的文本/图像提示构造、Alignment Head、Fusion Head 与三阶段训练，并复盘七数据集开放提示检测证据。"
tags: ["ICCV 2025", "遥感检测", "开放词汇", "多模态提示", "旋转框", "自训练"]
---

# OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images

**论文**：[arXiv 官方页面](https://arxiv.org/abs/2503.06146)  
**官方代码**：论文只承诺后续发布 codes and models，当前文稿没有可核验的 OpenRSD 仓库地址。  
**发表**：ICCV 2025

## 一句话总结

OpenRSD 在 RTMDet 上同时支持文本与图像 prompt，以快速的 **Alignment Head**和交互更深的 **Fusion Head**兼顾开放类别、水平/旋转框与实时性，再用 pretraining、fine-tuning、self-training 提升跨数据集泛化。

## 研究背景与问题

遥感数据集的类别命名、粒度、分辨率和框形式差异很大；开放词表检测还要兼顾小目标、任意朝向与实时性。OpenRSD 因此接受类别名、描述或示例图，并切换速度/精度 head。

**Prompt Construction**离线编码 SkyCLIP 文本与 DINOv2 图像并映射到 256 维。每类生成 10–15 条文本描述；图像框扩大 1.25 倍、裁成 224×224，再筛选最高置信的 100 张。训练时随机选一种 prompt，采样正负类，每类输入 3–7 个 embedding。

## 方法总览

OpenRSD 以 RTMDet 为视觉骨干，把离线编码的 SkyCLIP 文本 prompt 与 DINOv2 图像 prompt 映射到统一维度，并联合训练两个可切换检测头。Alignment Head 只计算预测 embedding 与 prompt 的归一化相似度，承担开放词表和快速扩类；Fusion Head 用三层双向 MHCA 深度交换图像与 prompt 信息，并借 learnable class embedding 区分跨数据集的类别粒度。两个 head 共享特征，却提供不同速度—精度档位。

## 方法详解

三阶段数据流为：七个标注集与 Million-AID → 仅训练 detection modules 的 pretraining；十个不同粒度数据集 → 全模型 fine-tuning；模型自标注 → self-training。预训练用 SAM 生成 proposal，DINOv2 embedding 聚类后以簇均值作 prompt；有向框同时生成外接水平框，兼容 HBB/OBB。自训练可冻结 backbone 或全参数更新。

## 实验与证据

论文在 **DIOR-R、DOTA-v1.0、DOTA-v2.0、FAIR1M-2.0、WHU-Mix、SpaceNet、HRSC2016** 上比较，并以 VEDAI、CORS、DOSR、SODA-A 做跨域测试。基线包括 RTMDet、YOLO-World、Grounding-DINO、MTP、CastDet。摘要报告其比 YOLO-World 平均精度高 **8.7 个百分点**、速度 **20.8 FPS**；HBB 上分别领先 4.0、6.5、13.8、4.6 个百分点，并约为 Grounding-DINO-T 的四倍速度。

组件消融中，仅 Alignment Head 在 DIOR-R/DOTA-v2.0/FAIR1M-2.0 为 71.5/68.8/45.4；加入 Fusion Head 为 73.1/69.5/45.3；再加入 class embeddings 后为 **72.9/70.4/45.9**，同时 HRSC2016 从 73.1 恢复到 **84.6**，说明类粒度冲突需要显式任务身份。联合训练后单独用 Fusion Head 的七集平均为 **68.2**，Alignment 为 67.0。prompt 数量消融显示 DOTA-v2.0 单文本 prompt 已有 70.1，单图像 prompt 65.7；图像 prompt 墠到 10 个后为 68.2，符合多示例降低随机性的解释。

训练策略消融更关键：仅 fine-tuning 的五集平均 64.0，加 pretraining 后 65.5；self-training 冻结 backbone 为 65.7，全参数更新达 **66.9**。额外加入 25 万张 Million-AID 图反而只有 65.3，说明更多无标注数据并不自动带来收益。跨域测试中相对 RTMDet-L 平均提升 2.2 个百分点，SODA-A 提升 5.2。

## 对 YOLO-Agent 的启发

开放提示检测应拆成轻量 embedding 对齐与深交互融合两个运行档。**对照组**：固定 YOLO backbone/neck，依次比较闭集分类头、text-only Alignment、image-only Alignment、dual-prompt Alignment、Alignment+Fusion，以及再加入 class embedding；同时保留“仅 fine-tuning、加 pretraining、冻结/不冻结 backbone 的 self-training”四种训练策略。**指标**：在训练内数据与未见 SODA-A 上分别报告 HBB/OBB AP、APs、20.8 FPS 对应的实际吞吐、prompt 扩展延迟、embedding margin、类别冲突率和多图像 prompt 方差。**失败判断**：若 Fusion 只抬高域内 AP 而跨域不升，class embedding 未降低同名异粒度类别冲突，图像 prompt 增多仍不稳定，或 prompt 数增长使 Alignment 档失去实时优势，就拒绝将 Fusion 设为 YOLO-Agent 默认路径。

## 优点

- 文本、图像 prompt 与 HBB/OBB 统一，任务覆盖面广。
- 双 head 给出明确的速度—精度选择，而非单一重模型。
- 三阶段训练和未见数据均有消融。

## 局限

- 原文尚未提供可确认官方代码，复现风险较高。
- GPT-4 文本生成、SAM proposal 和多数据集采样引入复杂依赖。
- 图像 prompt 单样本稳定性明显弱于文本，细粒度收益依赖示例质量。

## 评分

- 问题重要性：★★★★★
- 方法独特性：★★★★☆
- 实验证据：★★★★★
- 工程可迁移性：★★★☆☆
- YOLO-Agent 参考价值：★★★★★
