---
title: Progressive Exploration-Conformal Learning for Sparsely Annotated Object Detection in Aerial Images
publication: NeurIPS
year: 2024
description: "解析 PECL 如何用保形置信、类别原型与序列奖励筛选航拍稀疏标注中的伪框，并评估其对旋转小目标检测的实际增益。"
paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/47a287298e7887d1c25d4aabb918bd54-Abstract-Conference.html
code: https://github.com/SAOD-research/PECL
tags: [稀疏标注, 航拍目标检测, 强化学习, 保形预测, 旋转目标检测]
---

# Progressive Exploration-Conformal Learning for Sparsely Annotated Object Detection in Aerial Images

## 一句话总结

PECL 将航拍伪标签筛选改写为序列决策：Conformal Pseudo-label Explorer 逐候选决定选或不选，Multi-clue Selection Evaluator 用熵变化与置信边际给奖励，筛出的伪框再反向更新旋转检测器。

## 研究背景与问题

稀疏标注目标检测只标每张图中的少量实例，能以相同框数量覆盖比半监督学习更丰富的场景，但未标实例会被检测器误当背景。航拍图又比 COCO 更密集：论文给出的 DOTA 每图平均目标数为 68.4，COCO 为 7.7；飞机等大目标置信度高，小车辆等目标置信度低，因此固定阈值会系统性丢掉困难类别。

Co-mining、Region-based、Calibrated Teacher 等方法可挖伪框，但仍主要依靠固定或校准后的单点置信度。PECL 要利用类别原型、候选自身质量和同图实例上下文，学习一个可随候选序列变化的选择策略，而非再设计一条静态阈值规则。

## 方法总览

系统先用稀疏真值预训练 ReDet、OR-CNN 或 S2A-Net，并通过在线聚类得到每类 K 个原型。检测器对图像产生候选后，**Conformal Pseudo-label Explorer** 按顺序读取候选，输出二维动作分布，动作 1 接纳伪框、动作 0 拒绝。**Multi-clue Selection Evaluator** 估计状态—动作累计奖励，经验存入 replay pool 更新 explorer 与 evaluator。完成一幅图的探索后，稀疏真值与最终伪标签共同优化检测器，形成“检测器生成候选—策略筛选—伪标签更新检测器”的闭环。

## 方法详解

**Conformal Pseudo-label Explorer 的数据流**是：当前候选 `x̃` → 计算预测概率 `pro`、与同类原型最大余弦相似度 `fsim`、保形置信水平 `AP` → 拼接当前候选特征、原稀疏标注实例平均特征和此前已选伪标签平均特征 → 三层全连接策略网络 `π` → 采样 select/reject 动作。`AP` 由候选非一致性分数 `1-pro` 与同类 K 个原型的非一致性分数比较得到，因而同样的原始置信度可在不同类别分布下获得不同决策依据。

**Multi-clue Selection Evaluator 的数据流**是：探索状态与动作 → 三层全连接 Q 网络 → 预测累计奖励。即时评价 `ψ` 由信息熵变化 `ΔH` 与置信边际变化 `ΔU` 加权组成；若 explorer 接纳了令 `ψ≤0` 的候选，或拒绝了令 `ψ≥0` 的候选，奖励为 +1，反向情况为 -1。目标值为即时奖励加折扣后的下一状态 Q 值，损失一方面拟合目标值，另一方面最大化累计奖励；经验回放缓解策略估值过高。

**Progressive Detector Updating 的数据流**将最终伪框与稀疏真值分别计算分类和 SmoothL1 回归损失，真值分支额外使用 prototype loss；背景权重降为 0.3，减轻未标前景被当成负样本。每处理新图，候选分布、策略经验、类原型和检测器都会继续变化，所以 progressive 指在线互相强化，而不是多模型互挖或离线多轮阈值扩标。

## 实验与证据

数据集仅为 **DOTA** 与 **HRSC2016**。DOTA 有 2,806 张高分辨率图、188,282 个旋转实例、15 类，按类别随机保留 1%、2%、5%、10% 框；HRSC2016 用 5%、10% 标注率。基线检测器是 **S2A-Net、OR-CNN、ReDet**，同时比较 SOOD、Unbiased Teacher、Calibrated Teacher、BRL、Co-mining、Region-based，指标为 OBB/HBB mAP。

DOTA 1% OBB 上，ReDet 从 **48.10** 提升到 **63.72**，OR-CNN 从 49.36 到 56.91，S2A-Net 从 40.14 到 50.39；5% 标注率下 ReDet+PECL 达 **67.06**，高于 Unbiased Teacher 64.74、Co-mining 65.35、Region-based 65.71。HRSC2016 10% OBB 上三种基线加入 PECL 后分别达到 80.65、85.50、87.29。

关键消融以 DOTA 1% ReDet 为准：仅 explorer 为 **60.84**，加 evaluator 为 **61.31**，再加经验回放为 **63.72**。探索特征只缺置信水平、只缺特征相似度、只缺预测概率时分别为 60.90、61.17、62.40，三者齐全为 63.72；二动作空间也优于单一选择概率输出的 60.81。小车辆和船舶的提升明显大于棒球场、足球场，支持同图上下文对密集小目标更有效的主张。

## 对 YOLO-Agent 的启发

1. 将伪标签接纳建模为有状态的逐实例决策，让已标框和已选框影响后续候选，而非独立阈值判断。
2. 为每类维护多个在线原型，用保形排名和原型相似度修正 YOLO 类别间置信度失衡。
3. 把选择轨迹写入经验池，使 agent 能审计“为何选中该框”及奖励是否真正对应检测收益。

**YOLO-Agent Harness（PECL）**：在 DOTA 按类别保留 1%、2%、5%、10% 旋转框，并把 HRSC2016 作为船舶域外复核。**对照组**依次设置为固定置信阈值、只输入 `pro` 的 explorer、使用 `pro+fsim+AP` 的 explorer、追加 Multi-clue Selection Evaluator，以及再启用 replay pool 的完整 PECL；同时保留 Co-mining 与 Region-based 外部基线。**指标**覆盖 OBB/HBB mAP、small-vehicle/ship 等类别 AP、伪标签 precision/recall、逐类接纳率、Q 值校准误差和单位新增真阳性的训练开销。**失败判断**不只看总分：完整闭环若未超过最强外部基线，若保形置信与原型相似度不能改善小车辆或船舶的伪标签召回，若累计奖励升高却伴随 precision 连续下降，或经验回放带来的成本没有换来稳定增益，均应拒绝接入。

## 优点

- 将固定阈值问题转化为包含上下文的明确决策过程。
- 保形置信、类别原型和强化学习奖励职责清楚，消融可逐项对应。
- 同时覆盖一阶段、两阶段和 OBB/HBB 检测器。

## 局限

- 强化学习与逐候选决策增加训练时间，论文也明确承认这一点。
- 方法偏向密集目标场景，在普通稀疏场景可能不占优势。
- 奖励由熵和边际的代理量构成，并不直接等同于伪标签真实正确性。

## 评分

- 创新性：8.5/10
- 技术完整性：8.5/10
- 实验说服力：9/10
- 工程可迁移性：7.5/10
- 对 YOLO-Agent 价值：9/10
