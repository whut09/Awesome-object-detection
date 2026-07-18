---
title: "Partial Weakly-Supervised Oriented Object Detection"
description: "解析 PWOOD 如何以少量水平框或点标注训练 OS-Student，并通过动态 CPF 过滤充分利用未标注旋转检测数据。"
tags: ["CVPR 2026", "旋转目标检测", "PWOOD", "弱监督学习", "伪标签"]
---

# Partial Weakly-Supervised Oriented Object Detection

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Partial_Weakly-Supervised_Oriented_Object_Detection_CVPR_2026_paper.html)  
**官方代码**: [VisionXLab/PWOOD](https://github.com/VisionXLab/PWOOD)  
**任务**: Partial Weakly-Supervised Oriented Object Detection

## 一句话总结

PWOOD 用 Orientation Learning 从翻转/旋转一致性恢复角度、用 Scale Learning 从高斯重叠与 Voronoi-Watershed 约束恢复尺寸，形成 OS-Student，再通过 Gaussian Mixture Model 驱动的 Class-Agnostic Pseudo-Label Filtering 动态接收未标注图像伪旋转框。

## 研究背景与问题

旋转框标注昂贵；纯 WOOD 假设所有图都有水平框或点，SOOD 又要求一小部分完整旋转框。论文提出更弱的 Partial Weakly-Supervised 设置：训练集中只有 10%、20% 或 30% 图像带 HBox 或单点，其余完全无标注。难点在于 HBox 没有方向，点同时没有方向和尺度，而 teacher-student 伪框若使用固定阈值，在不同数据集和训练阶段会剧烈波动。

PWOOD 的目标不是直接把弱标注当旋转框，而是先让学生从几何变换和空间布局学习缺失变量，再让该能力经 EMA 传给 teacher；teacher 产生的旋转伪框又反过来监督 student，形成方向、尺度和无标注数据互相增强的闭环。

## 方法总览

OS-Student 与 teacher 共享 FCOS 式 backbone、FPN 和旋转检测头。预训练阶段只用部分弱标注，Orientation Learning 与 Scale Learning 建立方向/尺寸监督；到 burn-in 时把 student 权重复制给 teacher。之后未标注图分别做弱增强给 teacher、强增强给 student，teacher 框分数经 CPF 过滤后形成伪标签，student 用监督损失与无监督分类、centerness、框回归损失联合更新，teacher 用 EMA 跟随。

## 方法详解

Orientation Learning 采用 symmetry learning：原图与垂直翻转或随机旋转视图分别预测角度，输出必须满足已知变换关系。翻转分支约束变换前后角度和，旋转分支约束角度差等于施加的旋转角，均使用 Smooth-L1；弱监督对与自监督预测对共同工作，使 HBox 条件下也能学习 orientation。

Scale Learning 面向点标注。上界把每个预测 OBB 表示为高斯分布，以 Bhattacharyya coefficient 惩罚不同框的重叠，避免尺寸无限膨胀；下界用点作为前景 marker、Voronoi ridge 作为背景 marker，经 watershed 得到对象 basin，再按当前预测角旋转区域，用 Gaussian Wasserstein Distance 回归宽高。OS-Student 总监督损失还包含 FCOS focal classification、centerness 交叉熵和 IoU box loss，角度/重叠/分水岭权重默认 `0.2/10/5`。

CPF 对 teacher 全部伪框置信度拟合正、负两个一维高斯混合，正均值初始化为最高分、负均值为最低分，EM 估计后验，再从正样本后验决定动态阈值 `Td`。因此早期可接受较低但相对可靠的框，后期阈值随 teacher 质量提高。无监督分支用 teacher/student 的 class、centerness、四边距离输出计算 BCE 与 Smooth-L1，并按点置信度加权。

HBox 与 Point 两条监督路线共享 teacher-student 框架，但缺失信息不同。HBox 已给出近似中心和轴对齐尺度，主要需要从变换一致性中恢复角度；Point 只给中心，若没有 Gaussian overlap 上界，预测框可能扩张覆盖多个对象，若没有 watershed 下界，又可能收缩为极小框。OS-Student 将这两种退化方向分别约束，才使后续 teacher 有能力生成可用旋转伪框。

EMA 正反馈也存在启动条件。burn-in 前 teacher 不能凭空比 student 更懂方向和尺度，因此论文先用弱标注训练 OS-Student，再复制权重；之后 weak view 的 teacher 输出经 CPF 变成 strong view 的监督，student 的改进再由 EMA 回流。若过早加入未标注数据，GMM 看到的只是大量低质量单峰分数，动态阈值也无法挽救伪标签。

DOTA-v2.0 增加了大量小目标，PWOOD-HBox 相对 partial-RBox baseline 的优势比 DOTA-v1.0 更明显。这个现象提示收益不仅来自降低标注成本：弱标注几何约束与未标注伪框可能为密集小实例提供额外一致性监督。但点标注路线仍明显低于 HBox，说明缺失尺度信息的代价没有被完全消除。

论文还初步尝试 RBox、HBox、Point 多格式联合训练。在 DOTA-v1.5 的 20% 标注预算下，用一部分 HBox 替换昂贵 RBox 只产生约 `0.08 mAP` 差异，说明框架有潜力按标注成本动态组合监督。不过这部分证据规模小于主实验，尚不能证明任意比例和任意数据集都保持同样性价比。

噪声表中 Point 路线并非所有轻噪声都单调下降，例如少量扰动可能偶然提供尺度扩张信号。这提醒复现者不能只看最终 mAP，还应直接检查角度、宽高和伪框匹配质量，避免把随机标注抖动带来的偶然收益误判成稳定鲁棒性。

不同弱标注类型还应分别维护采样比例，防止便宜点标注在批次中压过更可靠的水平框监督。

## 实验与证据

- 数据集为 DOTA-v1.0/v1.5/v2.0 和 DIOR；基座 FCOS-ResNet50-FPN，比较 H2RBox-v2、Point2RBox-v2，以及 SOOD、Dense Teacher、PseCo、ARSL、SOOD++、MCL 和 partial-RBox Vanilla Baseline。
- DOTA-v1.0 在 20% 标注时，PWOOD-HBox 为 `62.93 mAP`，H2RBox-v2 为 `54.38`，partial-RBox baseline 为 `62.82`；PWOOD-Point 为 `45.01`，Point2RBox-v2 为 `40.39`。
- DOTA-v1.5 的 10/20/30% HBox 结果为 `52.87/59.36/61.58`；Point 为 `35.33/41.54/43.02`。20% HBox 下超过 PseCo 的 `55.28` 和 Vanilla Baseline 的 `58.28`，接近或超过多种完整 RBox 半监督方法。
- DOTA-v2.0 的 HBox 三档为 `31.03/36.39/40.27`，相对 partial-RBox baseline 的 `24.77/34.03/37.30` 在小目标更密集场景收益更明显。
- 固定阈值从 `0.02` 改为 `0.03` 可导致 `3.09 mAP` 下跌；CPF 在 HBox/Point 设置分别把最佳静态阈值的 `58.03/41.25` 提到 `59.36/41.54`。
- 噪声消融中，DOTA-v1.0 20% HBox 加 30% 标注噪声时 H2RBox-v2 从 `54.38` 降至 `41.75`，PWOOD 从 `62.93` 降至 `55.32`；说明未标注闭环与动态过滤提高了鲁棒性。

## 对 YOLO-Agent 的启发

对遥感 YOLO，PWOOD 给出按缺失标注类型选择监督的策略：只有 HBox 时启用几何等变角度约束，只有点时额外开启尺度上下界；不要把两者混成同一伪框生成器。Agent 还应监控 teacher 分数分布是否真呈双峰，再决定 CPF 是否可信。

**Harness**：在 DOTA-v1.5 20% 设置比较 H2RBox-v2/Point2RBox-v2、partial-RBox teacher-student、OS-Student 无未标注、固定阈值 PWOOD、完整 CPF-PWOOD。观测 mAP50、角度误差、宽高相对误差、小目标 AP、伪框 precision/recall、GMM 两分量间距和阈值轨迹。通过条件为 HBox 组相对对应 WOOD 至少 `+5 mAP`、Point 组至少 `+3 mAP`，角度误差下降且 CPF 对 `0.01` 阈值扰动的性能波动小于 `1 AP`；若 GMM 长期单峰、watershed 造成框膨胀，或未标注数据使伪框 precision 降低超过 `5%`，则失败。

## 优点

- 首次系统覆盖“少量弱标注+大量无标注”的旋转检测成本区间。
- OS-Student 对角度与尺度采用不同、可解释的几何监督。
- 在四个遥感数据集、两种弱标注和噪声条件下验证充分。

## 局限

- GMM 双分布假设在极弱 teacher 或类别极不均衡时可能不成立。
- Voronoi-Watershed 对密集、重叠和非凸目标的尺度估计存在偏差。
- 训练流程包含预训练、burn-in、双增强、EMA 与多项损失，复现复杂度较高。

## 评分

- **创新性**: ★★★★★
- **证据强度**: ★★★★★
- **工程可用性**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★★
