---
title: "Improving Single Domain-Generalized Object Detection: A Focus on Diversification and Alignment"
description: "原创中文详解 DivAlign 的单源域多样化、分类与定位跨视图对齐，以及域外校准证据。"
tags: ["CVPR 2024", "目标检测", "域泛化", "校准"]
---

# Improving Single Domain-Generalized Object Detection: A Focus on Diversification and Alignment

**论文**：[CVPR 官方页面](https://openaccess.thecvf.com/content/CVPR2024/html/Danish_Improving_Single_Domain-Generalized_Object_Detection_A_Focus_on_Diversification_and_CVPR_2024_paper.html)  
**官方代码**：[msohaildanish/DivAlign](https://github.com/msohaildanish/DivAlign)

## 一句话总结

DivAlign 先用 ImageNet-C 与 Fourier 扰动把单一源域扩成多种视觉域，再强制原图和扰动图在同一组 proposals 上保持分类分布与框坐标一致，从而同时提升未知域检测 mAP 与 D-ECE 校准。

## 研究背景与问题

Single-DGOD 训练时只有一个带标注源域，目标域及其文本描述都不可见。单纯增强能阻止检测器依赖纹理捷径，但原图与增强图会经 RPN 产生不同候选框，无法像分类任务那样直接对齐一个 logits 向量；而只看 mAP 也会忽略安全场景中的过度置信。论文因此把“域不变检测”定义为分类概率相等且对应框 IoU 差异为零。

## 方法总览

每个 mini-batch 同时送入干净图 `x_s` 和随机视觉腐蚀图 `x_φ`。两者共享 Faster R-CNN。为建立一一对应，方法不使用增强图自己的 proposals，而把干净图 RPN 产生的 `O_s` 同时作用在干净特征 `F_s` 与增强特征 `F_φ` 上，经 RoIAlign 得到空间一致的 proposal features，再计算 classification alignment loss 与 regression alignment loss。

## 方法详解

**Diversifying Single Source Domain**从 ImageNet-C 和 Fourier corruption 中选取 22 种、5 个强度等级，包含 blur、noise、digital 与 Fourier 组；为避免偷看目标天气，排除 weather corruption，同时去掉会破坏实例级结构的 constant amplitude。保留 phase scaling、high-pass filter 等频域变化，与高斯/运动模糊、噪声、JPEG、亮度和像素化共同模拟域偏移。

**Aligning classification（Lcal）**以干净 proposals `O_s` 对齐两条特征流：`A_s=RA(O_s,F_s)`，`A_φ'=RA(O_s,F_φ)`，然后最小化对应 proposal 分类概率的 KL 散度。**Aligning localization（Lral）**最小化两视图回归框坐标的 L2 平方距离。总目标为 `L_det + αL_cal + βL_ral`；它不要求当前预测正确，而要求同一对象在风格变化下输出稳定。

该设计也可放到 FCOS：核心不是 RoI 本身，而是为两视图建立相同空间锚点，再对齐分类与定位。分类对齐还能抑制增强图上的高置信错误，因此论文观察到没有显式 temperature scaling 也能降低域外 D-ECE。

## 实验与证据

Real-to-Art 使用 VOC2007+2012 的 16,551 张 trainval 作源域，在 Clipart1k、Watercolor2k、Comic2k 测试；Urban Scene 从 Daytime Sunny 训练，在 Night Clear、Night Rainy、Dusk Rainy、Daytime Foggy 测试。Faster R-CNN+ResNet101 与 FCOS+ResNet50-FPN 均训练 18k iterations，指标为 mAP@0.5，校准用 D-ECE。

Faster R-CNN 基线在 Clipart/Watercolor/Comic 为 25.7/44.5/18.9；仅 diversification 为 34.2/53.0/24.2；完整 `Lcal+Lral` 达到 **38.9/57.4/33.2**。只加 Lcal 为 36.2/53.9/28.7，只加 Lral 为 35.0/53.8/28.7，说明两任务均不可少。FCOS 上从 24.4/44.3/15.4 提升到 **37.4/55.0/31.2**。增强消融中 Fourier-only 为 36.1/54.8/29.1，ImageNet-C-only 为 36.8/55.5/30.6，联合为 38.9/57.4/33.2。域外 D-ECE 也明显降低，例如 Night Clear 从基线 27.9、diversification 28.9 降到 15.8。

## 对 YOLO-Agent 的启发

YOLO 可将同一标签的 clean/aug 图成对前向，并在 assignment 后按同一 GT 或同一 grid 索引对齐类别分布、DFL 与框坐标；训练日志应同时报告 AP 和可靠性。**Harness**：对照组为原始 YOLO、仅腐蚀增强、仅分类一致性、仅定位一致性、双一致性；观测源域与三个未知域 mAP50、ECE、clean-aug 分类 KL、匹配框 IoU。双一致性相对仅增强在至少两个目标域提升 2 mAP、ECE 降低 15% 且源域下降不超过 2 点即通过；若 alignment 只让置信度变低却不提高召回，或增强图 assignment 大量错配导致框 IoU 恶化，则失败。

## 优点

- 不需要目标域图像、目标域文字或额外生成网络。
- 同时处理分类稳定性、定位稳定性与检测校准。
- 对两阶段和单阶段检测器均给出独立验证。

## 局限

- 腐蚀集合仍是人工设计，难覆盖几何、传感器或语义层面的未知偏移。
- 双视图前向增加训练计算量，且强增强可能破坏小目标。
- 用干净 proposals 对齐增强图，依赖变换不改变几何位置。

## 评分

- **创新性**：8/10
- **实验充分度**：9/10
- **工程可迁移性**：9/10
- **YOLO-Agent 参考价值**：9/10
