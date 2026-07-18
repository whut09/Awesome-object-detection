---
title: "[论文解读] Deep Equilibrium Object Detection"
description: "DEQDet 把 query 的多层迭代写成固定点方程，用共享隐式 decoder 表示近似无限次 refinement，并以 RAG、RAP 稳定检测训练。"
tags: ["ICCV 2023", "目标检测"]
---

# Deep Equilibrium Object Detection

**论文**：[官方论文](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Deep_Equilibrium_Object_Detection_ICCV_2023_paper.html)  
**代码**：论文未给出可确认的独立官方仓库

## 一句话总结

DEQDet 把 query 的多层迭代写成固定点方程，用共享隐式 decoder 表示近似无限次 refinement，并以 RAG、RAP 稳定检测训练。

## 研究背景与问题

普通 query detector 堆叠多个 decoder layer，每层都重复相似的查询更新，显存随深度增加。直接套用通用 DEQ 又忽略了框查询逐步变准的过程。

## 方法总览

DEQ decoder 求解 z*=f(z*,x)。作者使用 two-step unrolled equilibrium equation 显式表示相邻 refinement，并提出 Refinement-Aware Gradient 在不精确隐式反传中保留更新方向；Refinement-Aware Perturbation 则沿求解轨迹加入深监督。

## 方法详解

### 1. 用共享 decoder 根据图像特征更新 query、类别和框。

### 2. 固定点求解器反复调用同一模块，直到残差或迭代上限满足。

### 3. RAG 利用两步展开结果修正隐式梯度，使训练感知 query refinement。

### 4. RAP 对优化路径施加扰动与深监督，减少求解不稳定和过拟合。

## 实验与证据

MS COCO 上，ResNet-50、300 queries、24 epoch 的 DEQDet 达到 49.5 mAP 和 33.0 APs，优于对应 AdaMixer 基线；论文同时报告更快收敛和更低训练显存。消融分别验证 two-step equilibrium、RAG、RAP，并比较固定点迭代数。

## 对 YOLO-Agent 的启发

记录 solver_max_iter、fixed_point_residual、RAG、RAP、forward/backward 收敛率。与参数共享的显式 AdaMixer 及相同查询数基线比较 AP、APs、显存和训练时间。若残差下降但检测损失不降，或 RAG/RAP 任一关闭都无影响，应视为固定点实现或监督路径有误。

## 优点

以隐式深度增加 query refinement，不需要线性堆叠 decoder 参数和激活。

## 局限

求解器容差直接影响速度和精度，导出到 TensorRT/ONNX 比普通固定层网络困难。

## 评分

- 创新性：8/10
- 证据完整度：8/10
- 工程迁移价值：7/10
- 综合：7.7/10

