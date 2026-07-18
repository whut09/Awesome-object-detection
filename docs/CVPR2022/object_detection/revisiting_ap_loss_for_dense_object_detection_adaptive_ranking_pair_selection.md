---
title: "[论文解读] Revisiting AP Loss for Dense Object Detection: Adaptive Ranking Pair Selection"
description: "解析 ARPS 如何扩展正样本排序对并以归一化排名/定位分数改进 AP Loss。"
tags: ["CVPR 2022", "目标检测", "排序损失", "样本选择"]
---

# Revisiting AP Loss for Dense Object Detection: Adaptive Ranking Pair Selection

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Revisiting_AP_Loss_for_Dense_Object_Detection_Adaptive_Ranking_Pair_CVPR_2022_paper.html)  
**代码**：论文条目未提供官方代码链接  
**发表**：CVPR 2022  
**主题**：AP Loss、Adaptive Pairwise Error、Adaptive Ranking Pair Selection

## 一句话总结

论文把 AP Loss 拆成距离函数、平衡常数和排序对选择三部分，实验证明真正限制精度的是排序对选择；ARPS 将定位较差的正样本重新视为自适应假阳性，并用实例内归一化的排名分数与 IoU 经 GMM 划分正负簇，形成更完整、准确的 APE 排序监督。

## 研究背景与问题

密集检测最终由分类分数排序并经 NMS 选框，因此 AP Loss 直接把分类学习写成正负样本的 pairwise ranking。对正样本 `u`，原 AP 精度误差统计分数不低于它的负样本数 `FP`，再除以 `TP+FP`；这种设计能缓解正负极度不均衡，但默认所有正样本之间无需排序。

论文指出这一默认与检测目标冲突：同一实例周围会产生多个正样本框，其中漏掉目标局部的框显然不应与精确框同级。另一方面，基于预设 anchor 与真实框 IoU 的样本划分不看图像内容，背景与前景位置可能获得相同优先级；直接把分类损失和回归损失送入 PAA，又会因两者尺度分布不同而偏向分类。

作者先替换 AP Loss 的分段距离函数、Error-Driven Update 和平衡常数，结果几乎不变；把 IoU 阈值选样换成 ATSS，AP 却从 37.3 提升到 39.2。由此提出 Adaptive Ranking Pair Selection（ARPS）和对应的 Adaptive Pairwise Error（APE）。

## 方法总览

检测器仍由 FPN、ranking subnet 与 localization subnet 组成。对每个正样本 `u`，ARPS 从原正样本集合 `P` 中找出 IoU 低于 `u` 的样本，并把它们加入该样本专属的负集合 `A_u`；原负样本集合 `N` 也并入 `A_u`。APE 对 `u` 与 `A_u` 中各样本计算成对排序误差。

随后，论文对每个实例的 ranking score 与 localization score 分别做 0-1 归一化，把二维点送入双分量 GMM。两个簇的所有跨簇组合构成排序对，使样本选择同时依赖图像内容分数和定位质量，而不是依赖固定 anchor 几何。

## 方法详解

原 AP Loss 对正样本 `u` 的误差可写成 `1-Precision(u)=FP/(TP+FP)`，其中样本对距离由分段函数 `H(P_v-P_u)` 近似；`P_u、P_v` 是 sigmoid 前排名 logits。论文将一般 pairwise error 概括为距离函数 `D`、平衡常数 `BC` 和被选中的 `(u,v)`。

ARPS 的 APE 为

\[
L_{APE}(P_u,P)=-\frac{1}{BC}\sum_{v\in A_u}D(P_v-P_u).
\]

`BC` 仍取 `rank^+(u)+rank^-(u)`，`D` 使用 sigmoid 后交叉熵 `CE(S(·),0)`。对每个 `u∈P`，若另一正样本 `v` 的预测框 IoU `I_v<I_u`，则 `v` 被标记为 adaptive False Positive（aFP）并加入 `A_u`；最后 `A_u←A_u∪N`。因此 APE 的 AP 解释变成 `(FP+aFP)/[(TP-aFP)+(FP+aFP)]`：正标签不再保证排序正确，定位较差者会被更好者压低。

为减少错误排序对，作者用每个实例内的最小—最大归一化统一 ranking score 和 IoU 的尺度。归一化后，理想正负样本分别聚集在 `(1,1)` 与 `(0,0)` 附近，PAA 的 GMM 将其划成两簇，簇间样本组合产生训练对。这样排名与定位共同决定样本身份，且无需给二者手工配权。

总训练包含 APE ranking loss 与 GIoU localization loss。FCOS 实现去掉 centerness 分支，并用类别预测加权 GIoU、用 IoU 加权 APE；RetinaNet-512 使用 48 epoch，FCOS-800 使用 24 epoch。

## 实验与证据

实验在 COCO 上进行，主消融使用 ResNet-50-FPN。RetinaNet 中 AP Loss 基线为 37.3 AP/38.9 AP75；换 APE 后为 38.3/41.1，AP75 提升 2.2。APE+ATSS 达 39.9 AP，完整 APE+PAA* 达 41.1 AP、59.6 AP50、43.7 AP75。排名分数与 IoU 的 PCC、SCC、KCC 从 0.3742、0.3790、0.2538 提升到 0.4940、0.5034、0.3449，说明分数与定位质量更一致。

PAA 输入消融中，Focal Loss+GIoU Loss 只有 39.4 AP；未归一化的排名+定位分数为 40.5；归一化后为 41.1。单独使用归一化排名或定位分数分别为 38.9、33.9，二者关联带来 2.2 AP。RetinaNet-512 最终为 41.1 AP，FCOS-800 为 41.5 AP；后者没有 centerness 分支。

与分类/排序损失比较，APE 在 PAA、800 输入、24 epoch 下为 41.5 AP。加入多尺度训练、ResNeXt-101-FPN-DCN 后 test-dev 为 49.0 AP，Res2Net-101-FPN-DCN 为 50.0 AP。训练效率表明 800 分辨率、0.01 学习率下 24 epoch 即可完成，而原 AP Loss 设置为 512 分辨率、0.001 学习率、100 epoch；论文同时强调加速很大部分来自更高学习率和分辨率，并非全部归功于 APE。

## 对 YOLO-Agent 的启发

可在 YOLO-Agent 的 matcher 输出之后增加 `arps_pairs`：保留原正负分配，但对每个真实实例按当前预测 IoU建立正样本内部有向排序对，再把实例内归一化的 objectness/class ranking score 与 IoU 输入二分 GMM。对照组应是现有分类损失、仅正样本 APE、APE+现有分配、APE+归一化 GMM，避免把 PAA 收益误算给损失。

指标应记录 COCO AP、AP75、NMS 前分数—IoU 的 PCC/SCC/KCC、每实例排序对数量和 aFP 比例。论文关键参照是 37.3→38.3 AP（仅 APE）与 37.3→41.1 AP（完整选择）。若 AP75 增益低于 0.5，或 PCC 不升反降超过 0.02，应判定正样本内部排序无效；若单实例排序对超过 100,000 或 aFP 比例长期高于 80%，则触发采样上限，否则二次复杂度会压垮训练并把大量未收敛正样本误判为 aFP。

## 优点

- 通过系统替换实验定位到“排序对选择”这一主因，而不是直接堆叠新损失。
- 把正样本定位差异纳入排名，目标与 NMS 选框更一致。
- 归一化 ranking/IoU 后再聚类，避免两种信号尺度不协调。

## 局限

- 排序对随候选数量快速增长，需要最大 pair 数限制和高效实现。
- IoU 与排名分数在训练早期都不稳定，GMM 可能产生错误簇。
- 训练加速与最终精度同时受输入分辨率、学习率和分配器影响，不能只归因于 APE。

## 评分

**8.9 / 10**：论文对 AP Loss 的诊断严谨，ARPS 把定位质量直接写入排序关系；其工程挑战是大规模成对计算与早期聚类稳定性。
