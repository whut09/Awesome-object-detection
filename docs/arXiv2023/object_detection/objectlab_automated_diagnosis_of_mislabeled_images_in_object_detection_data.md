---
title: "[论文解读] ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data"
description: "ObjectLab 用已有检测器给每张图的标注质量打分，分别寻找漏标目标、位置错误框和类别错标，并把最可疑图像优先送给人工复核。"
tags: ["arXiv 2023", "目标检测"]
---

# ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data

**论文**：[官方论文](https://arxiv.org/abs/2309.00832)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

ObjectLab 用已有检测器给每张图的标注质量打分，分别寻找漏标目标、位置错误框和类别错标，并把最可疑图像优先送给人工复核。

## 研究背景与问题

检测数据的一张图可同时含多个框，错误也有不同形态。单纯按训练损失或整图 mAP 排序，无法区分“模型没学会”和“标注本身有问题”。ObjectLab 因而从预测框与给定标注的匹配关系构造对象级错误分数，再汇总为图像级 label quality。

## 方法总览

算法输入是交叉验证得到的离样本预测。高置信预测若找不到匹配标注，形成 overlooked score；同类预测与标注能匹配但边界重叠差，形成 badly located score；空间匹配而类别概率冲突，则形成 swapped-class score。三个分数经校准后聚合，分值越低越应优先审核。

## 方法详解

### 1. 对每个标注框寻找类别一致、IoU 合理的预测，建立一对一对应。

### 2. 检查未被标注覆盖的高置信预测，估计图中可能漏掉的实例。

### 3. 利用匹配框的几何差和类别概率，分别评估框位置与类别标签是否可信。

### 4. 将对象级风险合成为图像级分数；排序只用于找错，不改检测器结构。

## 实验与证据

COCO-bench 含 2171 张 COCO2017 图像，只评 person、chair、cup、car、traffic light 五类，并利用额外重标结果确定真实错误；SYNTHIA-AL 则可精确注入漏标、框偏移和类别交换。COCO-full 的 Precision@100 中，Detectron-X101 预测驱动的 ObjectLab 达到 0.71，而基于 mAP 的质量分数为 0.22；使用 Faster R-CNN 时分别为 0.57 和 0.17。论文还以 Average Precision、Precision@100 和 Lift@100 评价“前排图像是否真的有错”。

## 对 YOLO-Agent 的启发

先冻结同一检测器，产生五折离样本预测，导出 overlooked、location、class 三个风险及总分。人工审核最低分的前 100 张，计算 Precision@100 与 Lift@100；修正后用完全相同配方重训。若排序前 100 的真实错误率不高于损失排序，或修正数据后 AP 没有改善，就停止接入。

## 优点

能同时覆盖三类常见框标注错误，而且不要求更换训练代码。

## 局限

质量上限受基础检测器制约；模型系统性漏检的类别也可能被误判为“标注正确”。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

