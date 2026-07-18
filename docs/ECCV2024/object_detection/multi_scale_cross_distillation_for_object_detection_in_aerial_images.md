---
title: "Multi-scale Cross Distillation for Object Detection in Aerial Images"
description: "解析 MSCD 如何让不同输入尺度与旋转视图的单尺度检测器互相蒸馏，并以 PMBA 融合参数而不增加推理成本。"
tags: [ECCV2024, aerial-detection, knowledge-distillation, multi-scale, oriented-detection]
---

# Multi-scale Cross Distillation for Object Detection in Aerial Images

## 论文与代码

- 论文：[ECCV 2024 官方页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6619_ECCV_2024_paper.php)
- 代码：ECCV 页面和论文附件均未列出 MSCD 官方仓库；现有证据只能确认 PMBA、像素级 ACSKD 的算法描述与表中结果，三尺度专家训练、融合权重求解及单模型导出脚本不可直接核验。

## 一句话总结

MSCD 先分别训练偏向不同图像尺度的旋转检测器，再让同一航拍图的不同缩放与旋转视图在像素特征和实例预测层面双向蒸馏，最后用 Performance-sensitive Multi-scale Balance of Averaging 合并分支参数，使部署仍只运行一个单尺度模型。

## 研究背景与问题

航拍图中的同类目标尺度跨度极端：DIOR-R 的飞机可从十余米到数百米，DOTA 同图既有小车又有机场。多尺度训练能提高鲁棒性，但单个参数集合容易在大目标与小目标之间折中；多尺度测试虽有效，却要对每个尺度重复前向、旋转框 NMS 也更昂贵。普通知识蒸馏通常固定 teacher/student，并在同一尺度对齐，无法把“大尺度视图擅长小目标、低尺度视图擅长大结构”的互补知识真正交换。MSCD 的目标是训练时使用多分支，推理时回到单分支成本。

## 方法总览

第一阶段训练多个 Single-Scale（SS）模型，每个模型对应一个输入缩放策略。Performance-sensitive Multi-scale Balance of Averaging（PMBA）根据各尺度模型的验证性能而非简单平均来融合参数，得到较强初始化。第二阶段执行 Adaptive Cross-Scale Knowledge Distillation（ACSKD）：同一图像生成不同尺度并附加随机旋转，分别送入各分支；Pixel-wise Cross Distillation 对齐空间特征，Instance-wise Cross Distillation 对齐 proposal/类别/旋转框预测；不同分支轮流提供软目标，形成跨尺度、跨方向的信息交互。结束后只保留融合后的单一检测器。

## 方法详解

PMBA 先计算每个单尺度模型的检测表现，将表现更好的模型赋予更高参数融合权重，避免算术平均把专长模型拉回中庸点。它既可作为最终模型，也可作为第二阶段各分支的共同起点。ACSKD 的像素级路径把同一物体在两个尺度特征图上的对应位置映射后，以通道与空间响应约束；实例级路径使用预测框将区域对齐，再蒸馏类别分布和 OBB 回归，因此不会要求整张特征图在不同尺度上逐像素完全相同。

独有数据流是“同一航拍图→尺度 s1/尺度 s2 与独立随机旋转→两个共享初始化的检测分支→像素对应蒸馏+实例 proposal 蒸馏→双向更新→PMBA/权重收敛→导出一个 SS detector”。随机旋转并非普通增强附带项，而是让交叉蒸馏同时交换尺度和方向知识。方法可接 RoI Transformer、Oriented R-CNN、S2A-Net、ReDet 等，不改变其检测头定义。

## 实验与证据

数据集包括 DOTA-v1.0（2,806 张图、188,282 个实例、15 类）、DOTA-v2.0（约 11,286 张图、180 万实例、18 类）和 DIOR-R（23,463 张图、190,288 个实例、20 类）。在 RoI Transformer 上，DIOR-R 的单尺度第一阶段为 63.10 mAP，第二阶段 MSCD 后 66.37；普通多尺度训练为 65.63，经 MSCD 后 67.93；PMBA 初始化为 68.95，MSCD 后达到 70.22。DOTA-v1.0 对应结果为 75.63→78.86、78.10→79.43、79.77→80.49。

ACSKD 消融以 DOTA-v1.0 单尺度 75.63 为基线：仅多一个无交互训练阶段为 76.64，Single-Scale Cross Distillation 为 76.35；跨尺度蒸馏达到 78.25，加入随机旋转的完整设置为 78.86。多尺度测试约耗单尺度 7 倍时间，而 MSCD 导出的单模型推理时间与原 SS 模型相同，精度接近多尺度模型。可视化中，机场、服务区、棒球场等尺度跨度大的类别得到更完整 OBB，小目标漏检也减少。

## 对 YOLO-Agent 的启发

YOLO-Agent 可把 MSCD 实现为训练编排器：先训练 0.5×、1.0×、1.5× 三个 YOLO-OBB 专家，以验证集估计 PMBA 权重，再进行含随机旋转的 ACSKD 双向蒸馏并导出单权重。DOTA-v1.0 复现实验用单尺度、常规 multi-scale augmentation、多尺度测试、简单权重平均、PMBA、PMBA+像素蒸馏、完整 ACSKD 形成阶梯式**对照组**，DIOR-R 用于检验类别尺度跨度迁移。总体 mAP、APs/APm/APl、角度 MAE、尺度跨度分桶 AP、跨尺度同实例框 IoU 与类别 KL、单图延迟和峰值显存构成**指标**集合。若 ACSKD 相对普通多尺度训练提升不足 0.7 mAP，或 APs 的收益以 APl 下降超过 1 点为代价，记为知识失衡的**失败判断**；导出模型延迟高于单尺度基线则说明训练期分支未被彻底移除，同样不能交付。

## 优点

- 把多尺度计算留在训练期，部署保持单模型速度。
- 同时蒸馏像素与实例，能覆盖纹理、分类和旋转框定位。
- 在多种初始训练策略和多种 OBB 检测器上均有提升。

## 局限

- 需训练多个尺度专家和第二阶段分支，训练总成本显著增加。
- PMBA 依赖代表性的验证集，尺度分布迁移后权重可能失效。
- 不同尺度的对应关系和 proposal 匹配对密集小目标容易含噪。

## 评分

- 创新性：8/10
- 实证充分性：9/10
- 工程可迁移性：8/10
- 对 YOLO-Agent 价值：8/10
