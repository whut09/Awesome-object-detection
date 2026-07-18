---
title: "T-norm Selection for Object Detection in Autonomous Driving with Logical Constraints"
description: "MOD-ECL 将自动驾驶多标签检测与 243 条常识约束连接起来，系统比较 12 类 t-norm，并以探索—利用算法和 λ 调度器控制精度与违规率。"
tags: ["NeurIPS 2025", "目标检测", "自动驾驶", "神经符号", "T-norm", "YOLO"]
---

# T-norm Selection for Object Detection in Autonomous Driving with Logical Constraints

**会议**：NeurIPS 2025  
**论文**：[NeurIPS 官方页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7dbdf006424e7749c8a35913d3574c4e-Abstract-Conference.html)  
**官方代码**：[pudumagico/MOD-ECL](https://github.com/pudumagico/MOD-ECL/)  
**任务**：带逻辑约束的自动驾驶多标签目标检测

## 一句话总结

论文构建 **MOD-ECL（Multilabel Object Detection with Enhanced Constrained Loss）**，把 YOLOv8n/YOLO11n 的多标签框预测代入 ROAD-R 与 ROAD-Waymo-R 的命题逻辑规则，用 12 种 t-norm 计算可微违规损失，并通过自适应 t-norm 选择和递增 `λ` 调度寻找精度—合规折中。

## 研究背景与问题

自动驾驶框不仅要定位，还可能同时带 agent、location、action 多种标签；例如 pedestrian、crosswalk、walking、push object 可以属于同一框。ROAD-R 又附加 243 条常识约束，如“vehicle lane 不能同时是 parking lot”。普通检测损失不会阻止模型输出逻辑矛盾，而把规则转成 constrained loss 时，具体采用 Product、Gödel 或 Łukasiewicz 等 t-norm 会产生完全不同的梯度形态。

以往工作通常静态指定少数 t-norm，并把约束权重 `λ` 固定。本文的实验表明，某个算子在一个数据集上减少违规，换到另一个数据集却可能同时损伤精度和合规；同一算子在训练早晚期也未必同样有效。因此问题不是“加不加规则”，而是如何选择规则的模糊逻辑实现，以及何时让规则损失主导训练。

## 方法总览

MODYOLO 先把 YOLOv8n 或 YOLO11n 改为 n-hot 多标签训练，并在推理时按唯一 agent 标签执行 NMS。每个至少有一个标签置信度超过 0.5 的候选框，其多标签概率被代入约束集合；强否定用 `1-a`，所有框和规则的平均值形成 `Lcl`。总损失为 `Lcomb=Lnn+λLcl`。MOD-ECL 可静态使用一个 t-norm，也可从 12 个算子集合中逐 batch 自适应选择；另一个 scheduler 在 warm-up 后指数增大 `λ`。

## 方法详解

框架实现 Product `TP`、Gödel `TG`、Łukasiewicz `TL`、Drastic `TD`、Hamacher Product `THP`、Nilpotent Minimum `TNM`，以及 Schweizer-Sklar、Hamacher、Frank、Yager、Sugeno-Weber、Aczél-Alsina 等参数化家族。不同算子对低置信度合取的惩罚强度差异很大，所以作者没有把它们视为可互换的数学细节，而是逐一训练比较。为避免背景候选制造大量无意义违规，低于 0.5 的全标签低置信框不参与规则计算。

论文用“Only pedestrians or cyclists can wait to cross”解释损失构造：规则写成 `Pedestrian ∨ Cyclist ∨ ¬Wait2X`，再把预测置信度经强否定与 t-norm 聚合。若 pedestrian、cyclist、wait-to-cross 分别为 0.9、0.2、0.7，Product 形式得到 0.056 的约束项。这个例子也揭示风险：相同概率换用取最小值或截断型算子后，惩罚与梯度会显著改变，进而影响模型选择降低哪个标签来满足规则。

**Adaptive T-norm Selection** 类似多臂老虎机。以概率 `β` 随机探索一个 t-norm，否则选当前得分最高者；计算该算子当前 constrained loss 与上次损失的归一化改善 `ni`，再用折扣因子 `δ` 更新历史得分，下一迭代切换到最高分算子。论文给出在平稳、有界得分等假设下的次优选择次数上界，但也坦言真实训练不一定满足全部条件。实际频率统计中，某些 Yager 变体会在保守探索配置下被选中超过 40%。

**λ Scheduler** 先保持 `λ0` 到 warm-up epoch `tw`，之后每个 epoch 执行 `λt=γ·λt-1`。其意图是让模型先学习检测，再逐步强化规则，而非一开始就用高权重把模型推向“宁可不识别也不违规”的保守解。实验设置中 `tw=3`，ROAD-R/ROAD-Waymo-R 的 `λ0` 分别为 50/100，并比较 `γ=1.1` 到 2.0。

## 实验与证据

ROAD-R 基于 Oxford RobotCar，含 22 段约 8 分钟视频、122K 帧、41 类和 243 条约束；ROAD-Waymo-R 基于 Waymo，规模更大、场景更复杂。作者因官方测试标签不可见，自行划分训练/测试视频。指标为 IoU 0.5 的 frame-mAP（f-mAP@0.5）和 **Mean Per Box Violation（MPBV）**，模型为 YOLOv8n、YOLO11n，并与 3D-RetinaNet、I3D 结果对照。

静态算子没有统一赢家。ROAD-R 的 YOLOv8n 基线为 57.55 mAP、40.45% MPBV，Product 在 `λ=50` 下达到 **61.00 mAP、35.66% MPBV**，实现双赢；Yager 变体可把 MPBV 降到 29.48%，但 mAP 仅 53.03。YOLO11n 上 Łukasiewicz 得到 61.74 mAP，却把 MPBV 推高到 67.61%。ROAD-Waymo-R 中同一 Yager 变体反而恶化到 47.53%/49.27% MPBV，直接证明 t-norm 不可跨场景照搬。

自适应算法在各 `(β,δ)` 配置下都降低 ROAD-R 违规，但通常牺牲部分精度；例如 YOLOv8n 的 `(0.1,0.25)` 为 55.10 mAP、30.20% MPBV，而 `(0.5,0.5)` 保留 58.93 mAP、MPBV 为 36.34%。`λ` 调度器展示更细的控制：ROAD-R YOLO11n 在 `γ=1.5` 时 MPBV 仅 0.019%，但 mAP 从 61.43 降到 56.17；过强调度在 ROAD-Waymo-R 甚至使合规重新变差。

附录三种子开销表显示，ROAD-R YOLOv8n 基线训练约 1176 秒，Product 约 2027 秒，自适应约 2800 秒；FLOPs 保持 4.12G，且没有额外推理开销。这说明规则计算的主要成本发生在训练期，但在线选择仍比固定算子昂贵。置信区间也提示部分 mAP 差异不大，不能只依据单次训练排名。

## 对 YOLO-Agent 的启发

YOLO-Agent 可以把业务规则编译成训练期可微约束，但不能只优化“违规数最低”。代理应把规则算子、规则权重与启用时机纳入联合搜索，并绘制 Pareto 前沿。对不同规则类型还可分配不同 t-norm：互斥标签、先验共现和空间关系的最佳松弛未必相同，统一 Product 只是最简单基线。任何合规提升都要同时检查预测框数量，防止模型通过少报目标规避规则。

### Harness

对照组为原始多标签 YOLO；实验组包含静态 Product、静态最佳候选、自适应选择、递增 `λ` 四组，数据划分和置信度阈值固定。观测 f-mAP@0.5、MPBV、每条规则违规频率、空预测率、每类召回、训练时间以及精度—违规 Pareto 面积。通过标准设为 MPBV 相对下降至少 15%，同时 mAP 不下降超过 0.5；若两者同时改善则优先接受。若 MPBV 的降低主要来自预测框数下降超过 10%，任一安全关键类别 AP 下跌超过 2 点，或跨随机种子 Pareto 排序翻转，则判定约束学习产生退化捷径。

## 优点

- 把 t-norm 选择从默认超参数提升为被系统验证的研究变量。
- 同时覆盖静态算子、在线选择和约束权重调度。
- 使用 MPBV 与检测精度共同呈现真实权衡，而非只报告规则损失。
- MODYOLO 与 YOLOv8n/YOLO11n 的结合便于在实际检测器上复验。

## 局限

- 自适应得分只看 constrained loss 改善，没有直接纳入检测精度。
- 自建测试划分降低了与公开 leaderboard 的直接可比性。
- 某些极低 MPBV 伴随明显 mAP 损失，仍需人工定义可接受折中。
- 研究只覆盖两个自动驾驶数据集，尚未证明可迁移到医疗等规则密集领域。

## 评分

- **创新性**：8/10
- **实验充分度**：8/10
- **工程可迁移性**：7/10
- **YOLO-Agent 规则学习价值**：9/10
