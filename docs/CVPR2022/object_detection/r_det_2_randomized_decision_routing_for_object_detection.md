---
title: "[论文解读] R(Det)2: Randomized Decision Routing for Object Detection"
description: "R(Det)2 把软决策树嵌入检测头，以随机路由、节点选择和关联损失学习多样决策。"
tags: ["CVPR 2022", "目标检测", "R(Det)2", "软决策树", "检测头"]
---

# R(Det)2: Randomized Decision Routing for Object Detection

**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Li_RDet2_Randomized_Decision_Routing_for_Object_Detection_CVPR_2022_paper.html)  
**代码**: 论文未提供官方仓库  
**任务**: 检测头决策结构设计

## 一句话总结

R(Det)2 将 soft decision trees 插入分类/回归 head，用窄分支生成节点 routing probabilities 与 masks，并以 node-selective、associative losses 让不同叶节点形成互补预测，接入现有 detector 可提高 1.4–3.6 AP。

## 研究背景与问题

常规检测头由共享卷积后直接线性输出，决策选择和预测值耦合在同一特征中。Mixture-of-experts 可产生多分支，却常出现所有分支学成相同函数或某一路垄断。论文借用树结构，把“样本走哪条路径”和“叶节点预测什么”显式分开。

## 方法总览

主检测特征进入若干 narrow routing branches，每个内部节点输出软概率和随机 mask；样本沿多条带权路径到叶节点，叶预测按路径概率加权。Node selective loss 鼓励节点针对不同样本作出明确选择，associative loss 约束相关节点的信息联系并稳定优化。该 head 可替换 RetinaNet、Faster R-CNN 等原决策层。

## 方法详解

Randomized routing 在训练时改变可用节点/路径，类似结构化 dropout，迫使叶节点承担不同决策区域；推理时整棵软树按概率集成。窄路由分支控制额外 FLOPs，预测值分支不必复制完整 head。树深、节点数和随机率共同决定容量与多样性。

## 实验与证据

COCO 上，R(Det)2 装配到多种 one-stage/two-stage detectors 后带来 1.4–3.6 AP 增益。消融分别去掉 randomized mask、node-selective loss、associative loss，并比较普通加深 head、并行 ensemble 和软树；完整组合最好，说明收益不是单纯增加参数。节点可视化显示不同路径偏向尺度、类别或几何不同的实例。

## 对 YOLO-Agent 的启发

在 YOLO head 上应比较同参数加深卷积、普通 MoE、soft tree、R(Det)2，并记录 APs/APl、路由熵、叶节点负载、额外延迟和导出算子。若大部分样本集中到单叶，或随机路由关闭后性能不变，说明多样性未形成；若 AP 增益小于同参数卷积且 TensorRT 出现动态分支开销，应判失败。还需检查分类树与回归树是否需要分开。

## 优点

- 将决策路径与叶预测解耦。
- 可插入多类现有检测器。
- 有明确损失抑制路由塌缩。

## 局限

- 软树增加实现与部署复杂度。
- 树深和随机率需要调节。
- 动态 mask 对部分推理后端不友好。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **工程价值**: ★★★☆☆
- **YOLO-Agent 参考价值**: ★★★★☆
