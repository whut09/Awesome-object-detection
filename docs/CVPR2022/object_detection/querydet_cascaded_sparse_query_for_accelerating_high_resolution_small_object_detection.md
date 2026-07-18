---

# QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection
title: "QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection"
description: "通过级联稀疏查询定位高分辨率特征中的小目标，以极少精度损失显著加速特征金字塔检测头。"
tags:
  - 小目标检测
  - 高分辨率检测
  - 稀疏卷积
  - 特征金字塔
  - CVPR2022
---

- **论文链接**：[CVPR 2022 官方论文](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html)
- **官方代码**：[ChenhongyiYang/QueryDet-PyTorch](https://github.com/ChenhongyiYang/QueryDet-PyTorch)

## 一句话总结

该方法的关键不是裁剪原图，而是让 **Cascade Sparse Query（CSQ）** 在特征金字塔内部执行由粗到细的空间查询。FPN 先产生 \(P_2\) 至 \(P_7\)；从查询起始层开始，Query Head 在低分辨率特征上输出小目标概率热图，超过阈值的网格成为 query keys。每个位于 \(P_l\) 的 key 被映射为 \(P_{l-1}\) 上四个相邻位置，这些位置从高分辨率特征中抽取 query values，组成稀疏张量，再交给稀疏分类头、回归头和下一层 Query Head。查询逐层级联至 \(P_3\)、\(P_2\)，因此高分辨率检测头只处理小目标附近，而非整幅特征图。

## 研究背景与问题

Query Head 与 RetinaNet 的分类、回归分支并行，输出单通道位置概率。训练时，论文把尺度小于当前层最小锚框尺度的实例定义为该层小目标，并依据特征点到小目标中心的距离生成二值 query map，以 Focal Loss 监督；分类仍使用 Focal Loss，框回归沿用 Smooth L1。加入 \(P_2\) 后样本数量会被高分辨率层主导，因此作者以随金字塔层级变化的 \(\beta_l\) 重平衡各层损失。推理时默认阈值为 0.15、从 \(P_4\) 开始查询，稀疏卷积直接复用四层密集检测头的卷积权重，不依赖 RoIAlign，也不在粗查询阶段回归候选框。

## 方法总览

实验覆盖 COCO 与小目标占比更高的 VisDrone，并以 RetinaNet 为主要基线，同时验证 FCOS、Faster R-CNN、MobileNet V2 和 ShuffleNet V2。COCO 上，RetinaNet 得到 37.46 AP、22.64 \(AP_S\)、13.60 FPS；加入高分辨率 \(P_2\) 的密集 QueryDet 提升到 38.53 AP、24.64 \(AP_S\)，但速度降至 4.85 FPS；启用 CSQ 后仍有 38.36 AP、24.33 \(AP_S\)，速度恢复到 14.88 FPS。消融中，加入 Query Head 使重平衡后的模型从 38.11 AP、23.06 \(AP_S\) 提升至 38.53 AP、24.64 \(AP_S\)，即增加 0.42 AP 和 1.58 \(AP_S\)。

## 方法详解

问题源于 FPN 检测头的计算分布极不均衡：分辨率每提高一级，空间面积约增至四倍。RetinaNet 的 \(P_3\) 检测头已占接近一半 FLOPs；继续增加 \(P_2\) 会令 \(P_2+P_3\) 占检测头约 75% 的计算，并把论文环境中的速度从 13.6 FPS 降到 4.85 FPS。与此同时，小目标在图像中通常稀疏分布，大量卷积实际落在纯背景区域。CSQ 利用“小目标位置稀疏”这一结构性先验，把低分辨率层视为廉价的位置预测器，把高分辨率层变成按位置访问的精细检测器。

## 实验与证据

组件消融说明，简单增加 \(P_2\) 会因训练样本分布偏移使 AP 从 37.46 降至 36.10；加入层间损失重平衡后恢复至 38.11，证明高分辨率训练不能只改网络结构。查询起始层对速度和召回同样敏感：从 \(P_6\) 开始仅有 37.91 AP，从 \(P_4\) 开始达到 38.36 AP、14.88 FPS，而从 \(P_3\) 开始虽有 38.45 AP，速度却降至 11.51 FPS。替代方案 Crop Query 和 Complete Convolution Query 分别只有 10.49、8.73 FPS，均慢于 CSQ 的 14.88 FPS。上下文实验表明，查询位置周围约 \(5\times5\) 的区域已能提供足够信息，继续扩大上下文只带来极小 AP 增益并持续降低速度。

## 对 YOLO-Agent 的启发

对 YOLO-Agent，最值得吸收的是把“是否启用高分辨率计算”建模为逐层空间决策，而不是固定地让所有输入经过完整高分辨率检测头。可在较粗尺度的 YOLO 特征层增加独立小目标查询分支，将高置信位置映射到下一高分辨率层，仅激活这些位置及有限上下文；分类、回归与查询共享主干特征，但查询损失应单独监督。Agent 的动作空间可限定为查询阈值、起始层和上下文范围，依据场景密度选择速度—召回工作点。该设计尤其适合无人机画面，但不能直接把 objectness 当作 query score：查询标签应针对“下一层需要处理的小尺度实例”，否则大目标会误激活高分辨率区域。

## 优点

VisDrone 上，RetinaNet 为 26.21 AP、2.63 FPS；密集高分辨率版本达到 28.35 AP，但仅有 1.16 FPS；CSQ 保留 28.32 AP 并提升到 2.75 FPS，说明目标越小、输入越大，空间稀疏化越有价值。FCOS 上，CSQ 版本取得 39.49 AP、24.81 \(AP_S\)、14.40 FPS，相比其密集高分辨率版本的 40.05 AP、25.52 \(AP_S\)、7.92 FPS 明显更快。Faster R-CNN 的 RPN 也可采用同一机制，速度由 17.57 提升至 19.03 FPS，但 \(AP_S\) 从 22.98 降至 22.23，收益弱于密集单阶段检测头。

## 局限

复现实验应设置三组核心对照：原始 RetinaNet \(P_3\!-\!P_7\)、加入 \(P_2\) 但保持密集计算的 QueryDet、加入相同 \(P_2\) 与 Query Head 并启用 CSQ 的完整模型；三者必须使用相同输入尺度、骨干、训练日程、混合精度和硬件。报告 COCO AP、\(AP_S\)、\(AP_M\)、\(AP_L\)、FPS，并在 VisDrone 报告 AP、各档 AR 与 FPS。进一步固定完整模型，分别比较 \(P_3/P_4/P_5/P_6\) 起始层、CQ/CCQ/CSQ 查询方式以及 \(1\times1\) 至 \(11\times11\) 上下文。若 CSQ 相对密集 \(P_2\) 模型未达到至少两倍 FPS，或 AP 下降超过 0.3、\(AP_S\) 下降超过 0.5，则判定稀疏查询实现失败，并检查 key 映射、上下文激活、阈值和稀疏卷积权重复制。

## 评分

主要失败模式有两类：Query Head 已正确激活小目标附近位置，但后续检测头仍无法准确定位；以及大目标或背景被错误激活，使稀疏张量膨胀并拖慢推理。其速度因此取决于目标密度和查询误报率，在极密集场景中未必保持高加速比。整体而言，这项工作的核心贡献是把高分辨率小目标检测从“全图密集计算”改写为“低分辨率找位置、高分辨率做局部精检”，并以 CSQ 将这一过程递归应用于 FPN。
