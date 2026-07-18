---
title: "[论文解读] Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration"
description: "解析 SAOD 任务、图像级不确定性、LaECE 校准指标与自感知检测器基线。"
tags: ["CVPR 2023", "目标检测", "不确定性", "校准", "OOD"]
---

# Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/CVPR2023/html/Oksuz_Towards_Building_Self-Aware_Object_Detectors_via_Reliable_Uncertainty_Quantification_and_CVPR_2023_paper.html)  
**代码**：官方 PDF 声明提供 SAOD 代码，但本任务未给出官方代码链接  
**会议**：CVPR 2023

## 一句话总结

论文提出 Self-Aware Object Detection（SAOD）统一评测：检测器既要用图像级不确定性拒绝 OOD 场景，又要让检测置信度同时匹配分类正确率与定位 IoU，还要承受域偏移；作者据此设计 LaECE、IDQ、DAQ，并把 Faster R-CNN、ATSS、RS-RCNN、Deformable DETR 转换为 SAOD 基线。

## 研究背景与问题

安全场景中的检测器不能只在同分布验证集追求 AP。它需要知道整幅场景何时超出训练域，对被接受图像给出可信置信度，并在腐蚀或域变化下维持性能。既有 OOD 工作常在 detection level 判断未知框，但“背景”和“未知物体”在没有像素级标注时难以区分；既有 ECE 又只比较置信度与分类准确率，忽略框是否准确。

论文还指出 top-k 会保留远多于真实物体数的低分框。COCO 每图平均 7.3 个 GT，而 ATSS 在 top-100 后平均产生 86.4 个检测；这些框不影响 AP，却能人为降低校准误差，让模型看起来更可靠。

## 方法总览

SAOD 检测器输出 f_A(X)={â,{ĉ_i,b̂_i,p̂_i}}：â 决定接受或拒绝图像，p̂_i 必须是分类与定位联合校准后的置信度。基线以检测不确定性 1-p̂_i 的 top-3 均值作为图像级不确定性，以伪 OOD 验证集选择接受阈值 ū；以每类 LRP 最优阈值 v̄ 过滤检测；再用线性回归校准分数。

## 方法详解

### 1. 图像级不确定性

聚合函数 G:X→R 输出场景不确定性，若 G(X)<ū 则接受。作者比较分类熵、Dempster-Shafer、1-p̂_i，以及概率检测器框协方差的行列式、迹和熵；聚合比较 sum、mean、min 和最确定 top-m 检测的均值。实验最终选用 1-p̂_i 的 mean(top-3)。阈值不是用真实 OOD 调参，而是在 ID 验证图上构造 pseudo-OOD set 做交叉验证。

### 2. Localisation-aware ECE

理想检测置信度应满足：

$$
P(ĉ_i=c_i | p̂_i) · E[IoU(b̂_i,b_{ψ(i)}) | p̂_i] = p̂_i.
$$

第一项是分类正确概率，第二项是真阳性框的平均定位质量。每类置信度被划分为 J=25 个等宽 bin：

$$
LaECE_c = Σ_j (|D̂_j^c| / |D̂^c|) · |p̄_j^c - precision_c(j) · IoŪ_c(j)|.
$$

p̄_j^c 是 bin 内平均分数，precision 计入 FP，IoŪ 计入 TP 定位质量，再对类别平均。因此 LaECE 同时惩罚错分类、误检与框不准；若只保留分类项，它才退化为常规分类校准。

### 3. 阈值、校准与统一指标

论文不用 top-k，而在验证集为每类选择使 LRP 最小的 v̄。LRP 同时计数 FP、FN 和 TP 的连续定位误差；ATSS、F-RCNN 的最优阈值约为 0.30、0.70，过滤后都约为每图 6 个框。校准器令 TP 的目标分数为匹配 GT 的 IoU，FP 的目标分数为 0；比较 linear regression、histogram binning、isotonic regression，基线采用 LR。

Balanced Accuracy 衡量接受/拒绝；IDQ 是 1-LRP 与 1-LaECE 的调和平均；域偏移集得到 IDQ_T；最终 DAQ 是 BA、IDQ、IDQ_T 的调和平均，任一可靠性维度过低都会拉低总分。

## 实验与证据

SAOD-Gen 用 COCO train/val 训练验证，以 Objects365 的 Obj45K 为 ID 测试及腐蚀源，OOD 由 Objects365、iNaturalist、SVHN 组成；SAOD-AV 用 nuImages train/val，BDD45K 为 ID 与腐蚀集。每个 use case 的 ID、三档腐蚀和 OOD 合计约 155K 图像。模型包括 Faster R-CNN、RS-RCNN、ATSS、Deformable DETR，以及 NLL-RCNN、ES-RCNN 概率检测器。

图像不确定性实验中，mean(top-3) 的 AUROC 在 SAOD-Gen 上为 F-RCNN 94.1、RS-RCNN 94.8、ATSS 94.2、D-DETR 94.7；far-OOD 明显比 Objects365 near-OOD 容易。校准方面，SAOD-Gen 的 F-RCNN LaECE 从 **43.3% 经 LR 降到 17.7%**，RS-RCNN 从 32.0 降到 17.4；ATSS 未校准已为 15.7。

完整评测中，SA-D-DETR 在 SAOD-Gen 取得最高 **43.5 DAQ**，其 BA 88.9、IDQ 41.7；SAOD-AV 中 SA-ATSS 为 44.7 DAQ。消融以 SA-F-RCNN 为例：普通阈值、无校准、固定 TPR 阈值时 DAQ 36.0；加 LRP 阈值为 36.5，再加 LR 为 39.1，最后加入伪集选择 ū 达 39.7。即使最佳 DAQ 仍低，说明现有检测器离安全自感知尚远。

## 对 YOLO-Agent 的启发

接入分三点：在后处理前保留每框最终置信度，以最确定 top-3 生成 scene score；在验证集按类别搜索 LRP-optimal 阈值替代固定 conf/top-k；训练后用线性回归把 TP 分数拟合 IoU、FP 拟合 0。对照组应为原 YOLO、仅图像拒绝、仅 LRP 阈值、仅 LR、完整 SAOD，并在 ID、腐蚀 1/3/5、near/far OOD 上报告 BA、LaECE、LRP、DAQ，而不能只看 AP。论文的 SA-F-RCNN 完整基线相对裸方案提升 3.7 DAQ，可把失败阈值设为：DAQ 增益低于 2、LaECE 不下降至少 5 个百分点、或 ID AP 下降超过 0.5；任何一项触发都应回滚对应组件。

## 优点

- 把场景拒绝、检测校准、域偏移和检测准确度统一到一套任务与指标。
- LaECE 显式融合 precision 与 IoU，修复分类 ECE 不适合检测的问题。
- 揭示 top-k 对校准的误导，并用 LRP-optimal 阈值替代。
- 大规模 near/far OOD 与腐蚀测试比小型 OOD 集更接近部署。

## 局限

- DAQ 是作者设计的复合指标，未覆盖所有应用的风险成本。
- 图像级拒绝会牺牲一张图中仍可识别的 ID 目标，不能定位具体未知框。
- 基线主要是后处理与校准组合，并未从训练阶段解决域偏移。

## 评分

- **创新性：9/10**——核心贡献是检测可靠性任务和度量体系，而非单一模型模块。
- **实验充分性：9.5/10**——两类应用、155K 级测试集、多检测器和完整消融。
- **可复现性：8.5/10**——协议与公式清楚，PDF 声明有代码但任务未提供链接。
- **YOLO-Agent 适配度：9/10**——可直接作为可靠性评测与后处理层，且给出明确失败指标。

