---
title: "[论文解读] AlignDet: Aligning Pre-training and Fine-tuning in Object Detection"
description: "AlignDet 把预训练拆成 image-domain 与 box-domain 两阶段，使 backbone 之外的分类、回归和查询模块也在无标注数据上学习，从而缩小预训练与检测微调在数据、结构和任务上的差距。"
tags: ["ICCV 2023", "目标检测"]
---

# AlignDet: Aligning Pre-training and Fine-tuning in Object Detection

**论文**：[官方论文](https://arxiv.org/abs/2307.11077)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

AlignDet 把预训练拆成 image-domain 与 box-domain 两阶段，使 backbone 之外的分类、回归和查询模块也在无标注数据上学习，从而缩小预训练与检测微调在数据、结构和任务上的差距。

## 研究背景与问题

传统自监督预训练通常只初始化 backbone，检测头、RPN、回归分支仍从零开始；有些方法预训练区域表示，却没有为不同检测器构造一致的分类与定位任务。AlignDet 的目标是用一个框架覆盖 FCOS、RetinaNet、Faster/Mask R-CNN 和 DETR。

## 方法总览

第一阶段在图像域训练 backbone，学习整体视觉表征。第二阶段利用无标注图像中的候选框与实例特征建立 box-domain 任务，同时训练检测器其余模块：分类部分学习实例语义，回归部分学习位置概念。微调时整套参数直接迁移。

## 方法详解

### 1. 执行常规自监督 image-domain 预训练，得到可迁移 backbone。

### 2. 在 COCO 无标签图像上生成候选区域，构造框级视图和对应关系。

### 3. 按目标检测器的结构预训练 neck、分类头、回归头或 query decoder，而不是强行共享一种头。

### 4. 微调阶段保留完整初始化，并在不同数据比例与日程下测试收敛速度。

## 实验与证据

仅用 COCO 进行 12 epoch 的 box-domain 预训练，AlignDet 在多种检测器上稳定增益。论文摘要报告短日程下 FCOS、RetinaNet、Faster R-CNN、DETR 分别提升 5.3、2.1、3.3、2.3 mAP。跨数据迁移到 VOC 时，Faster R-CNN 从 53.5 AP 提到 57.8，DETR 从 52.1 提到 58.2；5% COCO 数据下，Mask R-CNN 过长训练会从 19.1 降至 15.2 mAP，而 AlignDet 仍带来 1.4 mAP。

## 对 YOLO-Agent 的启发

开关 image_pretrain、box_pretrain、pretrain_head、pretrain_regression，并在 1%、5%、10%、100% 标注比例上运行。主要失败信号是：只迁移 backbone 就得到全部收益，或 AP75 没有随回归预训练改善。低数据实验必须同时报告 12k 与 90k 迭代，防止把缓解过拟合误写成最终精度优势。

## 优点

真正覆盖检测器全部模块，并验证了单阶段、双阶段和 query 检测器。

## 局限

需要额外的无标注预训练阶段，收益会受候选框质量与预训练数据域影响。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

