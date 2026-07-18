---
title: "[论文解读] Plain-Det: A Plain Multi-Dataset Object Detector"
description: "解析数据集专属冻结分类头、Class-Aware Query Compositor 与 Hardness-indicated Sampling。"
tags: ["ECCV 2024", "目标检测", "多数据集训练", "DETR", "开放词汇"]
---

# Plain-Det: A Plain Multi-Dataset Object Detector

**论文**：https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/763_ECCV_2024_paper.php  
**官方代码**：https://github.com/ChengShiest/Plain-Det  
**会议**：ECCV 2024

## 一句话总结

Plain-Det 不强行统一 COCO、LVIS、Objects365、OpenImages 的冲突标签空间，而是共享编码器、解码器和类无关框头，为每个数据集保留由 CLIP 文本向量构成的冻结分类头，再用 Class-Aware Query Compositor 和 Hardness-indicated Sampling 实现可扩展联合训练。

## 研究背景与问题

同一实例在一个数据集可标为 dolphin，在另一个数据集中却是背景；把标签拼成超大统一词表会持续积累冲突，新增数据集还需重建映射。DETR 的 top-K 查询初始化又依赖当前数据集分类器，强数据集先验会让共享 decoder 学成互不迁移的规则；完全共享随机查询则忽略语义和图像内容。

Plain-Det 追求新增数据集方便、各数据集性能稳定、训练高效，并能移植到不同 query-based detector，而不是设计更复杂的全局 taxonomy。

## 方法总览

图像经共享 `Enc` 得到特征。样本来自 `Dm` 时只启用其分类头 `Hc^m`；分类器由冻结 CLIP 文本编码器生成并做空字符串基向量校正，框头类无关共享。Class-Aware Query Compositor 用当前数据集分类器生成弱语义查询，再以全局图像特征调制后送入 decoder。Hardness-indicated Sampling 周期记录各数据集框损失与规模，在线调整采样比例。

## 方法详解

每个标签名用提示“the photo is [class name]”编码为 `Wm`，再计算 `Norm(Wm-Enctext(NULL))` 去掉 CLIP 基础偏置。分类头独立优化，因此 COCO 未标注的 LVIS 类不会被硬塞入统一负类；相似概念仍通过共同语义空间向共享视觉模块传递知识。

Query Compositor 先将校正分类器送入 MLP 得到 `Qb`，再对编码器特征全局 Max-Pool，经 MLP 得到图像权重 `W`，最终 `Qc=WQb`。它避免从数据集专属得分图直接挑 top-K 局部特征。在线采样器用周期框损失 `Lm` 表示难度，并结合图像数 `Sm` 计算权重；高损失、小规模数据会被更多采样，各数据集内部仍可保留 RFS 等策略。

## 实验与证据

训练组合从 LVIS+COCO 扩展到 Objects365、OpenImages，并在 13 个下游数据集和 ODinW 验证。以 Def-DETR 为底座，Plain-Det 在 COCO val 达 51.9 AP，单独 Def-DETR 为 46.9；ScaleDet 为 47.1，且经历 192 个 COCO 等价 epoch，Plain-Det 仅 36。移植 Sparse R-CNN 后 COCO AP 从 43.0 升至 46.1。

ODinW 五数据集零样本中，L+C+O+D、3.6M 数据的 Plain-Det 为 46.1 mAP，高于同数据量 ScaleDet 39.2，也超过 5.5M 数据 GLIP-T 的 44.0。消融在 C+L 上从 31.1 mAP 开始，加入 Class-Aware Query 达 36.5，再加入文本校正达 37.8；分区头自身约损失 0.5 点，但换来扩展性。C+L+O 的在线采样使平均 mAP 从 31.9 升到 35.0，并按预期过采样 LVIS、下采样 Objects365。

## 对 YOLO-Agent 的启发

YOLO 可借鉴“共享定位、隔离分类语义”：保持 backbone/neck 和类无关 box/objectness 共享，为各业务数据集挂独立词表适配器。查询合成器不能原样移植，但可转化为文本原型与全局图像向量调制分类特征的门控。

### Harness

对照组为单数据集 YOLO、标签并集联合训练、数据集专属头、专属头+CLIP 校正、再加语义门控与在线采样。使用 COCO+LVIS+Objects365，固定各集实际见图数，并保留 ODinW。观测各集 AP、平均 AP、尾类 AP、新增数据集后的旧集退化、训练迭代和采样轨迹。通过条件：平均 AP 比标签并集高至少 2 点，任一旧集退化不超过 1 点，ODinW 提升至少 3 点；若收益来自额外 COCO 暴露，或某数据集采样长期低于 10% 导致遗忘，则失败。

## 优点

- 无需人工统一 taxonomy，新增数据集改动局部。
- 冻结 CLIP 分类器兼顾语义共享与标签隔离。
- 在 Def-DETR、Sparse R-CNN、ODinW 上验证泛化。
- 在线采样直接利用训练难度。

## 局限

- 继承 CLIP 偏见与类别命名敏感性。
- 多分类头增加部署词表管理复杂度。
- 迁移到密集 YOLO 需重新设计调制位置。

## 评分

| 维度 | 评分 |
|---|---:|
| 方法完整性 | 9.0/10 |
| 扩展性 | 9.0/10 |
| 实验证据 | 9.0/10 |
| YOLO-Agent 价值 | 8.5/10 |
| 综合 | 8.9/10 |
