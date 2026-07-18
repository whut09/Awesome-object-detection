---
title: "[论文解读] SIOD: Single Instance Annotated per Category per Image for Object Detection"
description: "DMiner 用 SPLG 挖掘同类漏标实例，并以 PGCL 抵抗伪标签噪声。"
tags: ["CVPR 2022", "目标检测", "SIOD", "DMiner", "部分标注"]
---

# SIOD: Single Instance Annotated per Category per Image for Object Detection

**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Li_SIOD_Single_Instance_Annotated_per_Category_per_Image_for_Object_CVPR_2022_paper.html)  
**代码**: [solicucu/SIOD](https://github.com/solicucu/SIOD)  
**任务**: 每图每类仅标一个实例的检测

## 一句话总结

SIOD 规定一张图中每个出现类别只框一个实例；DMiner 先用 Similarity-based Pseudo Label Generating 从已标实例特征寻找同类候选，再以 Pixel-level Group Contrastive Learning 降低错误伪框对表示的破坏。

## 研究背景与问题

WSOD 只有图像级标签，定位先验太弱；SSOD 把整张图分成有标和无标，域差会影响 teacher。SIOD 将问题缩成同一图像内的已标/漏标差异：类别存在和一个真实位置已知，人工成本又远低于全框标注。难点是普通 detector 会把其余同类实例当背景。

## 方法总览

DMiner 的 SPLG 从已标 box 提取 prototype，在 proposal/像素特征空间搜索相似区域，并结合 detector score 生成 latent instance boxes。PGCL 把像素按前景组和背景组构造对比目标：同一伪实例内部聚合，不同实例和背景分离；低质量伪框边缘不会像硬框监督那样全部视为正样本。人工框与伪框共同训练检测头。

## 方法详解

SPLG 利用“同图同类外观通常相近”，但不直接接受所有高相似区域，而通过空间与置信筛选减少重复。PGCL 的 group 设计使伪框只需提供大致区域，模型可在区域内部寻找一致像素；因此对框偏移、包含背景和漏掉部分物体具有容忍度。

## 实验与证据

MS COCO 上，仅标约 40% 实例时，DMiner 已取得接近 fully supervised detector 的结果，并持续优于把未标实例当背景、WSOD 和 SSOD 改造基线。消融显示 SPLG 解决召回，PGCL 在伪标签精度不高时进一步增加 AP；去掉 PGCL 后，提高伪标签数量会更快导致性能下降。

## 对 YOLO-Agent 的启发

构造 COCO-SIOD 时必须按 image-category 保留一个随机实例，并多次种子重复。对照 ignore 未标区域、teacher pseudo-label、SPLG、SPLG+PGCL，报告 AP、伪框 precision/recall、每图漏标同类数和拥挤子集 AP。若增益依赖选择最大实例作为唯一标注，随机选择后消失，则不通过；若伪框召回增加但 AP75 下滑，应降低硬框权重并检查 PGCL 像素组。

## 优点

- 标注协议介于图像标签与全框之间，实际可执行。
- 同图已标实例提供可靠类别和外观锚点。
- PGCL 对伪框几何噪声具有针对性。

## 局限

- 同类外观差异大时相似度挖掘会漏实例。
- 每类至少要有一个人工框。
- 极密集目标的像素组容易互相污染。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★☆
- **工程价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
