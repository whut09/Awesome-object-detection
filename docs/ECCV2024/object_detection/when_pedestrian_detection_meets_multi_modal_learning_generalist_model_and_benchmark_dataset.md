---
title: "When Pedestrian Detection Meets Multi-Modal Learning: Generalist Model and Benchmark Dataset"
description: "解析 MMPedestron 如何以 Modality-Aware Adapter、统一融合和模态随机缺失训练一个跨传感器行人检测器。"
tags: [ECCV2024, pedestrian-detection, multimodal-learning, RGB-IR, event-camera]
---

# When Pedestrian Detection Meets Multi-Modal Learning: Generalist Model and Benchmark Dataset

## 论文与代码

- 论文：[ECCV 2024 官方页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6487_ECCV_2024_paper.php)
- 代码：论文没有提供 MMPedestron 的独立官方仓库；MMDetection、Dual ViT 与 Co-DINO/Faster R-CNN 只能界定依赖来源，MAA、MMF、模态缺失训练及大规模预训练配方仍缺少作者实现可供核验。

## 一句话总结

MMPedestron 用共享视觉编码器接收 RGB、红外、深度、事件和 LiDAR 投影图，通过 Modality-Aware Adapter 识别当前传感器属性、用统一融合模块聚合任意可用模态，并以 modality dropout 训练，使同一模型无需按数据集微调即可处理单模态、双模态和模态缺失的行人检测。

## 研究背景与问题

传统多模态行人检测往往为 RGB-IR、RGB-depth 或 RGB-event 分别设计双流网络，融合位置、参数规模和训练配方都绑定某一传感器组合；现实系统却可能白天只有 RGB、夜间依赖红外，事件相机或 LiDAR 还会临时掉线。已有“通用多模态”模型多聚焦分类或分割，难以应对拥挤遮挡下的实例级框回归。论文同时指出，各公开行人数据集标注格式、类别与测试协议分裂，无法公平判断一个模型是否真能跨模态泛化，因此构建 Multi-Modal Pedestrian Detection（MMPD）统一基准并提出 MMPedestron。

## 方法总览

模型以 Dual ViT 为默认编码器。各模态先经过共享 patch embedding，再插入 Modality-Aware Adapter（MAA）：它从输入 token 估计模态条件，并用轻量残差变换修正共享特征，避免为每个传感器复制整套 backbone。随后 Multi-Modal Fusion（MMF）对可用模态 token 做对齐和注意力聚合，输出统一金字塔特征，接 Co-DINO 或 Faster R-CNN head。训练分两阶段：先在大规模合并 RGB 行人数据上预训练，再在 MMPD 的多模态样本上联合训练；以 0.3 概率随机丢弃模态，使网络学习单模态退化路径。

## 方法详解

MAA 不是简单在输入通道拼接传感器数据，而是在 Transformer block 内根据当前模态生成适配残差；因此 RGB、IR、depth、event 和 LiDAR 共享主干注意力权重，模态差异集中在小型 adapter。MMF 接收一组存在掩码，先把不同模态映射到相同 token 空间，再让主模态查询其他模态的互补线索。独有数据流是“任意模态图像→共享 patch tokens→逐层 MAA 条件修正→带存在掩码的 MMF→统一多尺度特征→同一检测头”，测试时不需要预先固定模态数量。

MMPD 汇集纯 RGB 与 RGB-IR、RGB-depth、RGB-event、RGB-LiDAR 数据；新增 EventPed 经过 RGB/事件时间同步、单应性配准和共同视野裁剪，分辨率为 960×512。训练分布以 RGB 为主，其他模态约占 2%，测试则更均衡，这迫使模型依靠第一阶段 RGB 表征并从少量配对样本学习跨模态互补。Faster R-CNN 版本还把候选框比例扩展到 0.5、1、1.5、2、2.5、3，并将每图 proposals 提高到 2000 以覆盖拥挤人群。

## 实验与证据

基准包含 LLVIP、STCrowd、InOutDoor、EventPed 等不同传感器组合，并在 COCO-Persons、CrowdHuman 等纯 RGB 数据上验证通用性。多模态融合表中，Early-Fusion、FPN-Fusion、ProbEN、HRFuser、CMX 参数量为 41M–150M；MMPedestron+Faster R-CNN 仅 41M，却在 LLVIP/STCrowd/InOutDoor/EventPed 获得 66.8/66.4/66.1/75.3 AP。Co-DINO 版本为 62M，前四项达到 72.6/74.9/65.7/79.0，显著超过 CMX 的 59.6/61.0/62.3/58.0。

关键消融围绕 MAA、MMF 与 modality dropout：共享编码器但无模态适配时不同传感器互相干扰；加入 MAA 后单模态与融合结果均改善；MMF 优于早期拼接和 FPN 直接相加；训练时随机缺失模态能明显提高测试缺失传感器时的稳定性。代价不容忽视：RGB 预训练使用 64 张 V100、约 27,648 GPU 小时，多模态阶段再用 32 张 V100 训练 550k iterations、约 1,056 GPU 小时。

## 对 YOLO-Agent 的启发

面向车载或安防多传感器时，可保留单个 YOLO backbone，在 C2f 后插入 Modality-Aware Adapter（MAA），并于 PAN 前接入带存在掩码的 Multi-Modal Fusion（MMF）。LLVIP、STCrowd、InOutDoor、EventPed 四套数据应交叉验证；每种模态独立 YOLO、通道拼接、双流 FPN 相加、共享 backbone+MAA、MAA+MMF、再加 modality dropout 依次充当**对照组**。验收**指标**包含各单模态 AP、融合 AP、夜间与遮挡 AP、参数量、延迟，以及主动屏蔽 RGB/IR/Event 后的退化曲线和 1–3 帧错位敏感度。对于**失败判断**，完整模型在模态齐全时即使上涨，只要缺失任一支后比最佳单模态低 2 AP 以上便视为过度依赖；若 MAA 不能降低跨数据集 AP 方差，或轻微同步偏差导致误检激增，也禁止自动编排该融合方案。

## 优点

- 一个模型覆盖多种传感器组合，避免维护多套检测器。
- 41M Faster R-CNN 版本已超过多个更大融合基线，参数效率突出。
- EventPed 和统一 MMPD 协议把“跨模态泛化”变成可测问题。

## 局限

- 预训练算力极高，难以按论文规模完整复现。
- 多模态训练数据仍以 RGB 为绝对多数，稀有传感器的长尾能力受限。
- 配准误差、时间不同步和不同视场在真实系统中会破坏 token 对齐。

## 评分

- 创新性：8/10
- 实证充分性：9/10
- 工程可迁移性：7/10
- 对 YOLO-Agent 价值：8/10
