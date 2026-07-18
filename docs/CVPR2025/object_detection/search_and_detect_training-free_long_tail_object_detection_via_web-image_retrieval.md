---
title: "Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval"
description: "原创中文详解 SearchDet 的网页正负样例检索、DINOv2 查询、SAM 区域筛选与热图联合定位。"
tags: ["CVPR 2025", "目标检测", "训练免除", "长尾检测"]
---

# Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval

**论文**：[CVPR 官方页面](https://openaccess.thecvf.com/content/CVPR2025/html/Sidhu_Search_and_Detect_Training-Free_Long_Tail_Object_Detection_via_Web-Image_CVPR_2025_paper.html)  
**官方代码**：[Mankeerat/SearchDet_Long_Tail_Object_Detection](https://github.com/Mankeerat/SearchDet_Long_Tail_Object_Detection)

## 一句话总结

SearchDet 不微调检测器，而是为待检类别实时搜索网页正样例与由 Phi-3 生成的负概念图，用 DINOv2 构造与当前输入相关的类别查询，再联合 SAM proposals 和 patch similarity heatmap 输出长尾目标框。

## 研究背景与问题

开放词汇检测器遇到罕见品牌、专有物体或模糊类别名时，继续预训练和下游微调代价高，模型内部知识也可能没有该概念。网页搜索却能即时提供视觉实例，但正样例经常夹带共现背景，例如 surfboard 搜索图同时出现 waves。SearchDet 把动态网页样例当成推理时外部记忆，并用负概念从表示中减去干扰因素。

## 方法总览

给定输入图和类别名，Google 分别抓取 top-10 正样例；Phi-3-mini-4k-instruct 根据类别生成常见视觉干扰词，再检索负样例。DINOv2 对输入、支持图、SAM mask crop 与输入 patch 编码。系统先用正负支持构成 adjusted query 选择 SAM 区域，再独立生成像素热图，最后取被选 mask 与二值热图的交集形成检测框。

## 方法详解

**Attention-based Query Generation**计算输入 embedding 与每个正、负支持 embedding 的余弦相似度，各自 softmax 得权重，分别加权求和为 `A_pos`、`A_neg`，最终 `q_adjusted=A_pos-A_neg`。这比简单均值后相减更能保留与当前图像上下文匹配的实例特征。

**SAM Region Proposals**用 SAM-HQ 的均匀网格提示产生无类别 masks。每个 mask 外区域置零后取 DINOv2 CLS token，与多组 adjusted queries 比较。Frequency-based Automatic Thresholding 将 query-mask 距离排序分箱；某 mask 占一个 bin 超过 80% 才成为候选，并以支持 query 之间的参考距离分布做三倍标准差验证，拒绝不一致区域。

**Heatmap Generation**用 DINOv2 patch features 与正负查询做余弦相似度，热图缩放到 0–100，并以 50 二值化。Joint Object Grounding 取候选 SAM mask 与热图交集；SAM 提供边界，热图可补救漏分割或包含多个物体的粗 mask。

## 实验与证据

在 COCO val、LVIS Minival-1203、ODinW-35、Roboflow100 上以 IoU 0.5 评估。SearchDet 的 mAP 分别为 **59.3、43.6、33.1、27.9**；GroundingDINO-L 为 48.4、27.4、26.1、8.3。LVIS rare 类 AP 达 **55.9**，明显高于 T-Rex2 Visual-G 的 43.8。10-shot COCO 上 SearchDet 为 **61.4 mAP**，DE-ViT 为 52.9。

支持图数量从 1/3/5/7/10 增加时，COCO mAP 为 49.70/54.10/56.20/58.50/59.34。组件消融：完整方法 59.34；仅正样例 45.80；不做 heatmap RoI refinement 为 51.07；支持图 mean-pooling 为 55.47。平均首见类别查询中，网页抓取 2.9 秒、SAM 2.5 秒并行，后续距离筛选 0.57 秒、热图 0.68 秒并行，最大平均约 3.58 秒；类别缓存后可跳过抓取。

## 对 YOLO-Agent 的启发

SearchDet 可作为 YOLO-Agent 的低频“未知类专家”：常见类仍由 YOLO 实时检测，用户查询长尾概念时调用网页检索与分割支路，并把成功样例缓存为本地 prototype 库。**Harness**：对照组为 GroundingDINO 文本提示、仅正图 prototype、正负均值、attention 正负查询、完整 SAM+heatmap；观测 rare-class mAP50、首查/缓存时延、网页样例污染率、不同日期检索方差、错误类别名敏感度。完整方案相对文本基线提升≥8 mAP、缓存推理≤1 秒、跨三次检索 mAP 标准差≤2 点才通过；若负词误删目标属性、搜索结果污染超过 30%，或无网络时完全不可用，则失败。

## 优点

- 不训练即可吸收网页中的最新或极罕见视觉概念。
- 正负支持、区域 proposal 与像素热图各有明确消融贡献。
- 对网页结果变化进行稳定性分析，并支持查询缓存。

## 局限

- 首次查询约数秒，无法直接替代实时全类别检测器。
- 依赖搜索引擎、网络和类别名称质量，存在内容漂移与隐私风险。
- 多个冻结基础模型叠加，系统资源占用和许可证管理较复杂。

## 评分

- **创新性**：9/10
- **实验充分度**：8/10
- **工程可迁移性**：7/10
- **YOLO-Agent 参考价值**：9/10
