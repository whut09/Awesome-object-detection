---
title: "Unknown-Aware Object Detection: Learning What You Don't Know From Videos in the Wild"
description: "STUD 从野外视频的时空候选中蒸馏未知物体，并用能量不确定性损失校准检测器的开放集边界。"
tags:
  - CVPR 2022
  - out-of-distribution detection
  - unknown-aware detection
  - video
  - energy score
---

# Unknown-Aware Object Detection: Learning What You Don't Know From Videos in the Wild

- 论文：[CVPR 2022 官方页面](https://openaccess.thecvf.com/content/CVPR2022/html/Du_Unknown-Aware_Object_Detection_Learning_What_You_Dont_Know_From_Videos_CVPR_2022_paper.html)
- 代码：[STUD 官方仓库](https://github.com/deeplearning-wisc/stud)

## 一句话总结

STUD 利用训练视频里未标注但真实存在的路牌、建筑等区域，为每个已知目标合成一个“难度适中”的未知特征，再训练能量分支拒绝测试时的开放集物体。

## 研究背景与问题

常规检测器只在封闭类别上优化分类与回归，遇到鹿、岩石等未见类别时，仍可能以极高置信度输出 pedestrian 或 car。对象级 OOD 比整图 OOD 更难：同一张道路图像里，车辆可以是 ID，而广告牌、桥梁或动物是 OOD；同时为未知类别逐框标注既昂贵也不可能覆盖开放世界。论文选择视频作为无标注未知源，因为连续帧天然混合已知与未知实例。

## 方法总览

`Spatial-Temporal Unknown Distillation (STUD)` 与 Faster R-CNN 从头联合训练。关键帧中的已知框提供锚点，邻近参考帧的 proposal 先通过 energy percentile 筛选；小型编码器把 proposal 特征映射后，计算它们与已知对象的 L2 不相似度，再用 softmax 权重对多个候选特征加权求和，得到蒸馏未知对象。多个参考帧被拼接后执行同样过程，以扩大未知分布。ID 特征与蒸馏未知特征共同进入 logistic uncertainty branch，推理时该分支直接给每个预测框 ID/OOD 分数。

## 方法详解

空间蒸馏不是选“最远的一个框”。对关键帧对象 `h(x0,bi)`，参考帧中 objectness 超阈值的对象先经过两层 `3×3` 卷积和平均池化编码；两两 L2 距离 `s(i,j)` 经指数归一化为权重 `α(i,j)`，原始 proposal 特征按权重线性组合为 `o_i`。组合多个候选的目的，是让一个未知特征覆盖更丰富的局部模式，而非记住单一负样本。

时间维度随机从 `[t0-R,t0+R]` 采 T 帧，把各帧候选拼成集合后蒸馏。正式配置为 `T=3、R=9`。并非所有 proposal 都合适：低能量候选很可能仍是 ID，高能量极端背景又过于简单；因此作者按分类 logits 的 energy 排序，只取 `40%–60%` 区间的“中等困难”对象。

总损失为 `L_det + βL_uncertainty`。`L_uncertainty` 把 ID 与未知对象的 energy 输入带可学习斜率的 logistic regressor，要求 ID 概率高、未知概率低。BDD100K 使用 `β=0.05`，Youtube-VIS 使用 `0.02`。测试阶段先由 Faster R-CNN 产生框，再按 uncertainty threshold 判断 ID/OOD；阈值通常校准到 95% 的 ID 真阳性率，因而论文主指标采用 FPR95 和 AUROC。

## 实验与证据

训练 ID 数据为 BDD100K 与 Youtube-VIS 2021，OOD 测试集为去除语义重叠后的 MS-COCO 和 nuImages，骨干统一 ResNet-50。BDD100K→COCO 时，STUD 的 FPR95/AUROC 为 `52.18/85.67`，优于此前最佳 CSI 的 `69.38/80.85`；对 nuImages 为 `77.57/75.67`。与此同时 ID mAP=`30.5`，接近只做检测的 `31.0`。Youtube-VIS→nuImages 的 FPR95=`76.93`，也显著优于 CSI 的 `84.85`。训练成本约 10.1 小时，和普通检测 9.1 小时同量级，明显低于 CSI 的 15.3 小时。

几组消融指出收益来自“候选质量、时空多样性、损失形态”的组合。只用最远对象 AUROC=`83.04/71.38`，随机对象为 `79.61/70.42`，负 proposal 为 `80.94/72.92`，完整 STUD 达 `85.67/75.67`。去掉 energy 筛选时 FPR95=`62.23/83.54`，选择 40%–60% 后降到 `52.18/77.57`。`T=3` 相比 `T=1` 在 COCO 上增加 5.24 AUROC；但 T 继续增大反而引入冗余。把 logistic loss 换成 hinge loss，COCO AUROC 从 `85.67` 跌到 `74.32`；设为 K+1 类更只有 `59.40`。

时间窗口也不是越远越好。在固定三张参考帧时，`R=9` 附近效果最佳；当 R 扩大到从整段视频随机取帧，COCO AUROC 从 `85.67` 降到 `80.35`。作者的解释是，相邻但不完全相同的帧提供“接近决策边界”的困难未知，过远场景则产生容易区分、对边界塑形帮助较小的样本。正则权重同样呈中间最优：BDD100K 上 β=`0.03/0.04/0.05/0.06/0.07` 时，COCO FPR95 分别为 `63.52/59.52/52.18/57.37/55.03`，证明提升不是简单把未知损失不断加大。

从对象检测精度看，替代方案各有代价。像素空间 Mixup 的 AUROC 为 `81.76/70.17`，但 ID mAP 降至 `27.6`；GAN 合成的 mAP 为 `30.1`，BDD100K→COCO 的 FPR95 却有 `67.95`。STUD 保持 `30.5` mAP 的同时取得更低误报，因此真正的证据是“拒识与闭集检测的联合帕累托改进”。若迁移到 YOLO 后只看到 AUROC 上升，却伴随明显 ID AP 损失，就没有复现论文所声称的平衡。

评测前必须做语义去重：例如训练类别与 OOD 集中的同义类别、父子类别若未剔除，FPR95 会把“正确识别训练语义”误算成拒识失败。阈值只能在 ID 验证集上校准到 95% 真阳性率，不能浏览 OOD 测试分数后挑阈值；三次运行还应报告标准差，以对应论文的统计方式。

所有拒识框还应保留原始类别与能量，便于逐例审计。

## 对 YOLO-Agent 的启发

对 YOLO-Agent，STUD 的可迁移单元应是“proposal 特征上的开放集校准器”，而不是简单降低分类置信度。可从视频训练批次的多尺度 neck 特征中抽取候选框特征，按能量中段过滤后跨帧加权混合，再接二分类 uncertainty head；部署时保留原检测类别头，只额外返回 `known_score`。

**对照组**分成 Vanilla YOLO、Vanilla+后处理 Energy、YOLO+单帧随机负框正则、YOLO+STUD 三组，所有组使用相同 BDD100K 类别、训练步数、视频采样和 ID 验证阈值。STUD 组再拆 `T=1/3/5`、无筛选/20%–40%/40%–60%/60%–80% 与 logistic/hinge/K+1。**指标**以目标级 FPR95、AUROC 为主，附 ID mAP、ECE、每帧额外延迟，并分别在 COCO、nuImages 和自建道路未知集报告。**失败判断**不同于普通 AP 实验：只要任一 OOD 集 FPR95 未比后处理 Energy 下降至少 5 个点，ID mAP 损失超过 1 点，跨三个种子 AUROC 方差大于 1.5，或未知分支把夜间/雨天的已知车辆大量拒绝，便认定该蒸馏策略没有形成可用开放集边界。

## 优点

- 未知监督来自真实视频对象，而非纯噪声或像素级生成器。
- 时空蒸馏、能量筛选和不确定性损失均有独立消融支撑。
- ID 检测精度基本保持，说明拒识并非靠普遍压低置信度获得。
- 推理只增加一次轻量不确定性判断，不需要 MC-Dropout 多次前向。

## 局限

- “未知”由当前 ID 类集合定义，数据去重和类别边界变化会显著影响结果。
- 方法依赖视频及邻近帧；纯图片训练集无法直接提供同样的时间候选。
- 中段 energy percentile、R、T、β 都与数据域有关，跨场景需要重新校准。
- 线性混合发生在特征空间，蒸馏对象未必对应可解释的真实实例，错误候选也可能污染边界。

## 评分

- 问题重要性：**9.0/10**
- 方法可信度：**8.5/10**
- 部署可行性：**7.5/10**
- 综合：**8.5/10**
