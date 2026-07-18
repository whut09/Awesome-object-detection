---
title: "Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation"
description: "解析 Point2RBox-v2 的 Point-Axis 表示、空间约束、尺度投影与跨视图一致性如何完成点监督旋转检测。"
tags: [ECCV2024, oriented-object-detection, point-supervision, point-axis, weak-supervision]
---

# Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation

## 论文与代码

- 论文：[ECCV 2024 官方页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4051_ECCV_2024_paper.php)
- 代码：[官方 Point2RBox-v2 仓库](https://github.com/VisionXLab/point2rbox-v2)

## 一句话总结

Point2RBox-v2 不从单点直接猜完整旋转框，而把实例表示成中心点加两条经过中心的主轴，通过正交、空间边界、尺度投影和仿射前后一致性逐步确定轴方向与长度，再从两轴还原 OBB。

## 研究背景与问题

点监督旋转检测只给每个实例一个类别点，缺少宽、高、角度。已有方法依赖外部分割、合成纹理或复杂增强，尤其在桥梁、港口、跑道等边界模糊目标上容易把背景扩进伪框。论文观察到，旋转矩形可以由中心、长轴、短轴唯一确定；若先约束轴而不是同时回归五个框参数，点标注可以提供中心，图像边界和跨视图变化则可分别提供长度与方向证据。

## 方法总览

Point-Axis Representation（PAR）以中心 c、轴向量 u、v 表示框。Spatial Constraint 用中心到实例或邻域边界的距离限制轴端点，避免轴越过目标。Scale Projection 把沿轴采样的特征投影成一维响应，根据前景到背景的转折估计半轴长度。Consistency Constraint 对同一图像施加旋转、翻转和尺度变换，要求变换前后的中心与轴经过同一仿射矩阵后对应。最终由两轴端点恢复四个角点，并把生成框作为下游旋转检测器的伪标签。

## 方法详解

空间约束首先利用点所属对象的局部显著区域：长轴和短轴必须正交，两个端点应停留在支持该类别响应的区域内；当附近存在同类密集点时，以点间分割边界阻止一条轴跨入邻居。它处理的是“框不能伸到哪里”。尺度投影则处理“轴应伸多长”：沿候选轴双向采样多层特征，将二维特征压到轴坐标上，依据响应衰减确定正负方向长度，因而允许中心点不严格位于几何中心。

跨视图一致性是角度学习的核心。原图预测的轴经过已知仿射变换后，应与增强图预测轴重合；对于矩形的轴交换和 180° 周期，匹配时采用等价集合中的最小误差，避免角度边界不连续。独有数据流是“点标注与图像→中心热图/轴预测→空间约束筛除越界轴→轴向尺度投影修正长度→仿射一致性校准方向→PAR 转四点框→训练标准 OBB detector”。三个约束分别提供中心、尺度和方向证据，构成无需人工框的闭环。

## 实验与证据

实验覆盖 DOTA-v1.0/v1.5/v2.0、DIOR-R、HRSC2016。仅用点标注训练的 Point2RBox-v2 在 DOTA-v1.0 达到 72.61 mAP，明显超过 Point2RBox 的 67.42；在 DOTA-v1.5 达到 65.73，在 DOTA-v2.0 达到 51.89，DIOR-R 达到 61.95，HRSC2016 的 VOC07/VOC12 指标达到 89.32/95.58。作者还报告用生成伪框训练全监督检测器后可进一步提高，说明 PAR 输出不仅适合端到端弱监督，也能成为标注扩增器。

关键消融逐项加入 Spatial Constraint、Scale Projection、Consistency Constraint，完整组合最佳；去掉尺度投影时细长目标的轴长偏差明显，去掉一致性时方向随机或受纹理干扰，去掉空间约束时密集场景出现跨实例大框。证据还包括伪框可视化：桥梁、船舶、车辆的轴端点更贴近真实边界，密集小车之间的重叠框减少。论文也与 H2RBox-v2、Point2RBox 等弱监督基线比较，Point-Axis 在多套遥感数据上保持优势。

## 对 YOLO-Agent 的启发

该方法可作为“廉价点标注→YOLO-OBB 训练框”的离线代理：训练 Point-Axis Representation（PAR）生成器，对未标框航拍图输出带置信度的 OBB，再按轴约束残差筛选。以 DOTA-v1.0 与 DIOR-R 的小规模真实 OBB 子集验收时，中心点扩固定尺寸框、Point2RBox 伪框、分别移除 Spatial Constraint/Scale Projection/Consistency Constraint 的 PAR、完整 PAR 共同组成**对照组**。伪框 IoU、中心误差、角度 MAE、长短轴相对误差、下游 YOLO mAP50/75 是主**指标**，且必须按细长目标、密集同类、点偏移和弱边界分桶。Scale Projection 若未让长轴误差至少降低 10%，Consistency Constraint 若未降低旋转增强后的轴等变误差，或完整 PAR 训练出的 YOLO 比固定尺寸点框低超过 1 mAP，均满足**失败判断**并停止自动扩标；轴互换频繁或正交残差过大的样本转人工复核。

## 优点

- 表示与监督缺口匹配：点给中心，图像证据补轴，而非直接回归完整框。
- 生成的 OBB 可复用给任意旋转检测器，不绑定特定部署网络。
- 对密集、细长和任意方向目标提供了机制级约束。

## 局限

- 轴向响应依赖可辨认边界，阴影、遮挡和纹理相似背景仍会误导尺度。
- 训练链路包含伪框生成与二阶段检测器训练，数据工程较长。
- 点位置若系统性偏离目标主体，会同时污染空间约束和尺度投影。

## 评分

- 创新性：9/10
- 实证充分性：8/10
- 工程可迁移性：8/10
- 对 YOLO-Agent 价值：9/10
