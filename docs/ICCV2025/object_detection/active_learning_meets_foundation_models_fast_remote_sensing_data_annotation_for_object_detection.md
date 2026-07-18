---
title: "[论文解读] Active Learning Meets Foundation Models: Fast Remote Sensing Data Annotation for Object Detection"
description: "原创中文解读 AL4FM-OD：以 SAM 框和掩码特征联合驱动不确定性、样本多样性与冷启动半自动标注。"
tags: ["ICCV 2025", "主动学习", "遥感标注", "SAM", "目标检测"]
---

# Active Learning Meets Foundation Models: Fast Remote Sensing Data Annotation for Object Detection

**论文**：[CVF 官方论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Burges_Active_Learning_Meets_Foundation_Models_Fast_Remote_Sensing_Data_Annotation_ICCV_2025_paper.html)  
**代码**：[官方训练代码与界面](https://github.com/mburges-cvl/ICCV_AL4FM)  
**发表**：ICCV 2025  
**类别**：Active Learning, Foundation Models, Data Annotation

## 一句话总结

AL4FM-OD 预先用 SAM 为未标图像提取掩码与框，以 SAM 框和检测框的一致性估计定位不确定性，用掩码后的基础模型特征估计对象多样性，并在冷启动阶段用 SAM 框替换检测器的劣质定位结果加速人工确认。

## 研究背景与问题

遥感目标检测标注不仅要判类别，还要精确画框。主动学习可减少样本数，但早期检测器分类与定位都不稳定：纯熵采样只看类别置信度，可能忽视框偏移；全图特征的多样性又会被大面积背景主导；半自动标注若直接展示冷启动检测框，会让用户花时间修正。本文利用两个基础模型能力：SAM 提供对象边界候选，遥感/通用预训练骨干提供可聚类特征，让样本选择和框建议都围绕对象而非整图。

## 方法总览

每轮先对未标集合离线运行 SAM，缓存 masks 与其 bounding boxes；最新 RT-DETRv2 输出检测框。Dual-Source Uncertainty Estimation（DSUE）忽略类别，以 Hungarian matching 对齐 SAM 框与检测框，把匹配代价指数映射为质量 `σ_match`，再与检测置信度 `σ_OD` 做调和平均，得到对象不确定度，图像内取平均；先选出预算 n 的 θ 倍候选。Mask-guided Diversity Estimation（MDE）用已标框提示 SAM 得到对象掩码，将掩码作用于冻结骨干特征，按类别以 k-means++ 建立 5 个原型及背景原型；候选对象按余弦相似度归类，用最近与次近原型间隔和类内最大距离评估多样性，选最终 n 张。Dynamic Box Switching（DBS）在标注界面中保留检测器类别，却用重叠最高的 SAM 框替换定位。

## 方法详解

DSUE 把“检测器自己很自信但框与 SAM 边界不一致”的对象判为高不确定。对象分数为 `1 - 2σ_ODσ_match/(σ_OD+σ_match+ε)`，因此任一来源质量低都会提高采样优先级。两阶段预算扩展避免直接在全未标集做昂贵多样性计算，默认 θ=6。

MDE 的关键是先以 mask 剔除框内背景再聚类。候选掩码与类别原型匹配后，图像得分来自各预测类别内部的最大两两距离均值，优先选择已标集合中欠覆盖的外观。DBS 只替换几何框，不篡改检测器类别；它针对“预训练骨干使分类先变好，而框回归仍冷启动”的时间差，约 50 张标注后收益衰减，应允许关闭。

## 实验与证据

实验覆盖 DIOR、DOTAv2、FAIR1M、HRSC2016 与 363 张 WaffleHomes，主模型为 RT-DETRv2，默认 FMOW 预训练 ResNet50，骨干冻结。AL4FM-OD 在最终轮 DIOR 为 10.68 mAP，PPAL 为 4.89；HRSC2016 为 50.52，高于随机 47.79；WaffleHomes 为 42.05，高于 DivProto 36.04。DOTAv2 上 PPAL 9.24、本文 7.93，显示 SAM 对极小目标的限制。DIOR 消融在 300 张标注时，随机 4.0、熵采样 1.3、DSUE 5.4、DSUE+CoreSet 4.8、DSUE+MDE 10.7。预算扩展 θ=6 达 10.7，θ=7 计算时间从 45 秒增至 87 秒。DBS 在首轮使 WaffleHomes R@100 提升约 4 倍、HRSC2016 约 2 倍，但 50 张后优势消失。8 人一小时用户研究中，本文平均完成 235 个标注、接受 51.75 个建议框，随机流程为 209 和 13.33；最终 AP50/AP75/AR100 为 0.34/0.17/0.43，对照为 0.28/0.11/0.35。

## 对 YOLO-Agent 的启发

该工作适合做数据闭环 Agent，而非改检测网络本身。YOLO 可替代 RT-DETRv2，但必须区分“采样收益”“预标框收益”和“基础模型缓存成本”，避免把人工交互节省与模型 AP 混为一项。

**机制特定 Harness**：**对照组**固定 5 张与 10 张两种冷启动、每轮图像预算和总人工分钟，依次比较随机采样、熵采样、DSUE、DSUE+全图特征多样性、DSUE+MDE，以及带 Dynamic Box Switching 的完整 AL4FM-OD。**指标**按目标面积、边界清晰度、旋转角、每图实例数和空图比例统计学习曲线 AUC、每分钟新增正确框、预标框接受率、修改点击数、SAM Recall@IoU0.5/0.75，并单测 SAM 网格密度、θ 与 50 张后关闭 DBS 的时机。**失败判断**以标注闭环而非单轮 AP 决策：完整流程的 AUC 相对随机提升不满 2%，或单位正确框人工时间节省少于 15%，或极小目标连续两轮入选率低于随机，或 DBS 框接受率比检测器预标框低 10 个百分点以上，即停止使用该采样与切换策略。

## 优点

- 不确定性同时利用类别置信度与独立边界来源，适配检测任务双重误差。
- 掩码特征降低遥感大背景对多样性估计的污染。
- 有真实标注员研究，验证的不只是离线 mAP。

## 局限

- 全流程依赖 SAM 对目标的可分割性，极小或上下文定义目标会级联失败。
- 用户研究仅 8 人、单次一小时，统计规模有限。
- SAM 离线缓存和候选扩展仍有显著计算/存储成本。

## 评分

**8.8/10**。方法围绕标注流程的真实瓶颈设计，消融与用户研究互补；最大风险是 SAM 在极小、模糊和语义边界目标上的可靠性。
