---
title: "[论文解读] Optimal Correction Cost for Object Detection Evaluation"
description: "OC-cost 用最优传输估计把检测结果人工修正成真值所需的图像级成本。"
tags: ["CVPR 2022", "目标检测", "OC-cost", "评测指标", "最优传输"]
---

# Optimal Correction Cost for Object Detection Evaluation

**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Otani_Optimal_Correction_Cost_for_Object_Detection_Evaluation_CVPR_2022_paper.html)  
**代码**: 论文未提供官方仓库  
**任务**: 目标检测评估

## 一句话总结

Optimal Correction Cost 把预测框与真值框看作两组质量，求解带匹配、删除、添加和类别/位置修正代价的 optimal transport，以“把这一张图修正确需要多少操作”补充 mAP 的排序视角。

## 研究背景与问题

mAP 聚合全数据集的置信排序，同一图像内的假阳性、漏检和框偏移并不按实际修正工作量计价；一张很差的图也可能被大量容易样本掩盖。对于人工审核、自动标注和逐图质量控制，更关心每张图需要删除几个框、补几个框、移动多远。

## 方法总览

OC-cost 构造 prediction-to-GT transport matrix。把预测搬运到真值产生 localization/class correction cost；多余预测流向虚拟删除节点，对应 false-positive cost；缺失真值从虚拟添加节点补充，对应 false-negative cost。求解最小总运输代价后按图像归一化，再对数据集平均，因此每幅图权重相等。

## 方法详解

匹配不是先按 IoU 贪心再计数，而是全局选择总代价最低的对应关系。代价权重可根据下游业务调整，例如漏检比误报更昂贵；这也使 OC-cost 不宣称替代 mAP，而是提供面向修正行为的另一坐标。论文用人工偏好实验检验指标是否符合人对单图优劣的判断。

## 实验与证据

人类比较实验中，OC-cost 对单图检测质量的排序与人工偏好一致性高于 mAP 式分数。对不同数据划分，模型按 OC-cost 的排名比 mAP 更稳定。作者还构造相同 mAP 但错误类型不同的例子，展示 OC-cost 能区分大量漏检与少量定位偏差，并分析 false-positive/false-negative 权重变化。

## 对 YOLO-Agent 的启发

Harness 可在 COCO AP 外增加平均/分位数 OC-cost，并为自动标注场景设置漏检、删除、移动框的真实人时权重。应比较两个模型在 mAP 相近时的 p90 correction cost 和最差图片。若 OC-cost 排名对任意小权重变化都反转，说明业务成本未校准；若只汇报均值仍掩盖灾难帧，应强制 p95 与错误操作分解。不可用 OC-cost 替代标准 benchmark AP。

## 优点

- 明确对应人工修正行为。
- 同时处理漏检、误报、分类和定位错误。
- 图像等权，适合质量审核。

## 局限

- 操作成本权重具有业务主观性。
- 最优传输计算高于常规 AP 统计。
- 与社区历史榜单不可直接比较。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★☆
- **工程价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★★
