---

# VLPD: Context-Aware Pedestrian Detection via Vision-Language Semantic Self-Supervision
title: "VLPD：基于视觉—语言语义自监督的上下文感知行人检测"
description: "解析 VLS 语义分割与 PSC 原型语义对比学习如何借助 CLIP 伪标签，在无额外上下文标注的条件下改善小尺度、重遮挡行人检测。"
tags:
  - 行人检测
  - 视觉语言模型
  - 自监督学习
  - 语义分割
  - 对比学习
  - CVPR2023
---

- **论文链接**：https://openaccess.thecvf.com/content/CVPR2023/html/Liu_VLPD_Context-Aware_Pedestrian_Detection_via_Vision-Language_Semantic_Self-Supervision_CVPR_2023_paper.html
- **官方代码**：https://github.com/lmy98129/VLPD

## 一句话总结

该方法以无锚点行人检测器 CSP 为主体，CLIP 预训练的 ResNet-50 从 S3、S4、S5 阶段提取视觉特征，经反卷积和拼接形成 Detection Features。其 VLS（Vision-Language Semantic）分支同时运行一个冻结的 CLIP 教师：图像经过冻结视觉编码器并保留投影后的像素级特征，类别文本通过提示句“A picture of [CLS].”进入冻结文本编码器；每个像素特征与每个语言向量计算余弦相似度，产生上下文类别伪分割图。可训练的检测骨干一边接受 CSP 的行人框监督，一边用 Smooth L1 损失拟合这些伪分割图，预测结果还会输入检测头作为显式语义上下文。

PSC（Prototypical Semantic Contrastive）进一步处理 VLS 伪标签粗糙、行人局部可能被误判成其他类别的问题。VLS 的非人类类别分数图先上采样并经过带温度的 Softmax，再对 Detection Features 做逐像素加权和空间求和，得到 ground、building、tree、traffic sign 以及 car、bicycle、bus、truck 等负原型；CSP 中表示行人中心的二维高斯图以同样方式聚合出正原型。标注行人位置上的每个像素特征保留为查询，损失将其拉近正原型、推离整个 mini-batch 内的跨图像负原型，而不反向扰动 VLS 分数图。

实验只使用行人框，在 CityPersons 与 Caltech 上以对数平均漏检率 MR⁻² 为主指标，即在 FPPI 从 10⁻² 到 10⁰ 的区间统计漏检率，数值越低越好。CityPersons 上，PyTorch 重实现 CSP 的 Reasonable／Small／HO 分别为 10.96／16.05／40.59；仅换成 CLIP 初始化后为 10.13／12.59／38.97，加入 VLS 后为 9.70／12.57／36.50，再加入 PSC 达到 9.41／10.93／34.88。相对“CLIP 初始化即可”的控制组，完整方案在重遮挡 HO 上下降 4.09 个百分点，说明收益来自显式上下文学习与语义判别，而非单纯更强的预训练。

## 研究背景与问题

城市环境中的交通标志、杆体等物体可能呈现近似人体的轮廓，导致假阳性；小行人或被车辆、建筑物遮挡的行人又缺乏完整外观，容易漏检。已有方案存在两类限制：SMPD 依赖 CityScapes 的人工分割标注，成本高且难迁移；FC-Net 只学习邻域隐式特征，EGCL 则在局部候选框间做对比，均无法明确回答“上下文是什么”。

本文的关键判断是：目标框内部信息不足时，需要先以视觉—语言模型自动识别全图语义类别，再把这些类别转化为检测器可利用、可区分的上下文。其核心并非用 CLIP 直接检测行人，而是将 CLIP 的跨模态映射转化为自生成的像素级监督。

## 方法总览

VLS 采用 Compacted Class Policy，避免直接照搬 CityScapes 的细碎类别。road 与 sidewalk 合并为 ground，building、wall、fence 合并为 building，vegetation 与 terrain 合并为 tree，person 与 rider 合并为 human，pole、traffic light、traffic sign 合并为 traffic sign。vehicle 因内部差异过大而被拆开，只保留出现较频繁的 car、bicycle、bus、truck，舍弃 motorcycle 与 train。

类别策略不是无关紧要的标签整理。完整 CityScapes 类别得到 10.40／12.87／37.14，全部粗粒度压缩得到 10.47／13.30／40.49，均不如所提策略的 9.70／12.57／36.50。删除车辆四类后恶化到 10.61／13.61／38.84，删除 ground 后为 10.51／13.53／37.48，表明交通参与者与道路区域是判断行人上下文的重要参照。

## 方法详解

总损失为 `LDet + λ1·LVLS + λ2·LPSC`，其中 VLS 权重 λ1=100。检测头沿用 CSP：卷积降维后分别预测 Center Heatmap、Scale Map 与 Offset Map；高度由 Scale Map 给出，宽度采用固定宽高比 0.41，Offset Map 修正水平和垂直定位，最终组合为行人框。

训练图像尺寸在 Caltech 上为 336×448，在 CityPersons 上为 640×1280；优化器为 Adam。Caltech 使用一张 RTX 3090，CityPersons 使用两张 RTX 3090，测试均在单张 3090 上完成。所有测试保持原始 1×图像尺度，不使用可见框、头部框或额外语义标注。

## 实验与证据

CityPersons 上，完整方法取得 Reasonable 9.4、Heavy 43.1、Partial 8.8、Bare 6.1、Small 10.9。它与上下文相关方法相比明显更强：SMPD 为 9.9／45.6，EGCL 为 10.9／46.4，FC-Net 为 13.9／46.8；在另一套遮挡划分中，R+HO 与 HO 达到 21.7 和 34.9，也优于 EGCL 的 24.8 和 39.3、PRNet++ 的 25.4 和 40.9。

Caltech 上 Reasonable、All、Heavy 分别为 2.3、52.4、37.7；其中论文正文给出的精确 Reasonable 值为 2.27%，低于 PedHunter 的 2.31%。Heavy 从 CSP 的 45.8 降到 37.7，显示显式非人类上下文尤其有利于处理跨类别遮挡。

## 对 YOLO-Agent 的启发

若在 YOLO-Agent 中复现这一思想，不应把它简化为“加入 CLIP 文本特征”。应增加冻结的 CLIP 伪标签路径、可训练的 VLS 分割路径，以及作用于 YOLO 多尺度融合特征的 PSC 路径。VLS 输出既作为辅助损失，也应拼接或注入检测头；PSC 的正原型由 GT 行人区域或中心热图聚合，负原型由非 human 语义分数加权聚合，并保留跨图像负样本。

接口上需要记录类别提示集合、CLIP 视觉投影层、特征层分辨率和梯度截断位置。关键约束是 VLS 分数用于生成 PSC 权重时停止梯度，否则对比损失可能通过修改语义图来“投机”，破坏原本的上下文自监督。

## 优点

建议严格设置五组 CityPersons 控制实验：原始 YOLO/CSP；仅换 CLIP 初始化；CLIP+VLS；CLIP+VLS+PSC；完整模型但允许 PSC 梯度回传 VLS。统一报告 Reasonable、Small、HO 的 MR⁻²，并补充 human-like traffic sign 假阳性数与重遮挡漏检数。类别策略还应对比 Full CityScapes、Full Compacted 和论文的 Compacted Policy。

具体失败判据：若完整模型相对“仅 CLIP 初始化”在 Small 与 HO 上均未降低 MR⁻²，或 Reasonable 的改善小于 0.5 个百分点，则不能声称上下文模块有效；若加入 PSC 后未优于 VLS-only，首先检查正原型区域、跨图像负原型和停止梯度是否与论文一致。

## 局限

方法的优势在于无需额外语义标注，却能把“道路、建筑、树、交通标志、车辆”等概念显式引入行人检测；PSC 又把粗糙语义图转换成全局类别原型，计算量低于全像素两两对比。两部分分别回答“场景里有什么”和“如何将行人与它们分开”。

局限在于伪标签质量受 CLIP 与文本提示影响，类别集合需要依据城市数据分布人工压缩；human 类与行人框并不完全等价，VLS 也无法显式修复边界噪声。固定 CSP 检测头及 0.41 宽高比还限制了对姿态变化和非常规人体比例的建模。

## 评分

最重要的结论不是视觉—语言预训练能够替换 ImageNet，而是跨模态相似度可以成为无需人工分割标注的上下文教师。仅 CLIP 初始化已经改善小尺度结果，但 VLS 与 PSC 才持续降低重遮挡漏检。该路线适合迁移到现代检测器：保留主检测监督，用冻结视觉—语言模型产生显式类别分数，再以检测标注构造正原型、以语义上下文构造负原型，实现检测任务内部的语义自监督。
