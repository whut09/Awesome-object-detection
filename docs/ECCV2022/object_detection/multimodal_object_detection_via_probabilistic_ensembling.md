---
title: "[论文解读] Multimodal Object Detection via Probabilistic Ensembling"
description: "ProbEn 用 Bayes 规则融合 RGB 与热成像类别概率，并通过不确定性加权完成框融合。"
tags: ["ECCV 2022", "多模态检测", "ProbEn", "RGB-T", "概率融合"]
---

# Multimodal Object Detection via Probabilistic Ensembling

**论文**: [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1448_ECCV_2022_paper.php)  
**代码**: 论文提供开源实现，链接见官方论文页面  
**任务**: RGB—热成像多模态目标检测

## 一句话总结

ProbEn 不训练额外 fusion network，而在条件独立假设下用 Bayes 规则把 RGB 与 thermal detector 的后验概率相乘并除以类别先验，再以定位方差加权融合框，同时通过边缘化处理某一模态漏检。

## 研究背景与问题

RGB 在白天细节丰富，热成像在夜间更可靠；early/mid fusion 需要严格配准和成对标注，late fusion 若仅 pool+NMS 会丢掉低分模态证据，简单平均又会把两个一致高分压低。FLIR 等数据还存在模态不对齐和只在一侧标框的问题。

## 方法总览

两个单模态 detector 独立产生类别分布、box 与不确定性。候选先按跨模态重叠匹配；匹配框的类别后验按 $p(y|x_1,x_2)\propto p(y|x_1)p(y|x_2)/p(y)$ 合成，两个模态一致时分数上升。框坐标根据各自预测方差做 precision-weighted averaging。未匹配候选通过对缺失模态积分而保留，不要求两个 detector 同时触发。

## 方法详解

ProbEn 可叠加在 early/mid fusion 模型输出之上，说明它是校准后的决策层而非特定骨干。论文也比较 NMS、score average、mixture of experts 和 learned fusion。box fusion 使用概率回归视角：方差小的模态贡献更大，避免固定平均在热/RGB错位时拉偏框。

## 实验与证据

实验覆盖对齐的 KAIST 与未对齐 FLIR。ProbEn 相对已有多模态方法取得超过 13% 的相对性能提升，并在 day/night 子集都优于单 RGB、单 thermal、pooling、NMS 与平均分数。消融显示 score fusion 是主要收益，加入 box fusion继续改善定位；即使条件独立假设不严格成立，融合已有 learned-fusion 输出仍能增加 AP。

## 对 YOLO-Agent 的启发

对照应包含 RGB-only、thermal-only、NMS late fusion、score average、ProbEn-score、ProbEn-score+box。按昼夜、配准误差和模态缺失率报告 mAP、AP75、校准 ECE 与每帧延迟。若融合分数升高但可靠性恶化，Bayes 组合可能重复计算相关证据；若人工平移热图后 box fusion 比 NMS 更差，应增大该模态定位方差或停止坐标融合。还要验证单模态失效时边缘化不会抑制可靠候选。

## 优点

- 无需重新训练融合网络。
- 能处理模态漏检和输入不对齐。
- 概率公式提供可解释的类别与框融合。

## 局限

- 条件独立和置信度校准在现实中不完全成立。
- 候选匹配仍依赖跨模态空间重叠。
- 两个 detector 的总计算成本较高。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★★
- **工程价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★★
