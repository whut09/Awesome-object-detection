---
title: "[论文解读] DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model"
description: "详解 DI-MaskDINO 的残差双重 token 选择与 BATO，如何缓解 MaskDINO 解码器早期检测落后于分割的不平衡。"
tags: ["NeurIPS 2024", "目标检测", "实例分割", "MaskDINO", "多任务学习"]
---

# DI-MaskDINO: A Joint Object Detection and Instance Segmentation Model

**论文**：[官方论文页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html)  
**代码**：[论文声明的官方代码](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO)  
**发表**：NeurIPS 2024

## 一句话总结

DI-MaskDINO 针对 MaskDINO 解码器首层“框检测 AP 明显落后于掩码 AP”的失衡，通过 De-Imbalance 模块生成 balance-aware query，再由 Balance-Aware Tokens Optimization 用框、掩码双指导 token 重整约两万个 encoder token，使检测和实例分割从解码器入口就更均衡。

## 研究背景与问题

MaskDINO 在 DINO 上增加 mask 分支，用同一组 query 联合预测类别、框和实例掩码。论文观察到一个具体中间现象：解码器开始层的检测性能落后于分割，且这种差距会限制最后一层上限。原因之一是框回归需要跨 patch 的全局几何关系，而 dense mask supervision 更偏局部像素相似性，直接按分类分数选 top token 并不能专门补强检测。

DI-MaskDINO 不采用重新加权多任务损失的常见做法，而是在 decoder 之前改造 Query 与 Key/Value。其目标是让 query 先经历面向框定位的 token 交互和二次筛选，同时保留原始 token 的细节，再将平衡信息反向注入 encoder tokens。

## 方法总览

Feature Tokens Extractor 使用 backbone 与六层 multi-scale deformable attention encoder 产生初始 token `Ti`。De-Imbalance（DI）执行 first selection、两层 MHSA、second selection 和残差式 MHCA，输出 300 个 `Qbal`。Balance-Aware Tokens Optimization（BATO）从 `Qbal` 生成 box guiding token 与 mask guiding token，融合后通过 MHCA 优化约 20k 个 `Ti`，得到 `Tbal`。

Transformer Decoder 以 `Qbal` 为 Query、`Tbal` 为 Key&Value，逐层得到 `Qref`。检测头从 `Qref` 输出类别与框；分割头结合 `Qref`、原始 `Ti` 和 1/4 分辨率 CNN 特征输出实例掩码。该数据流明确区分了“选择候选”“形成平衡查询”“优化密集特征”三个阶段。

## 方法详解

**Residual Double-Selection。** 第一次按类别分数从 `Ti` 取 top-`k1` 得到 `Ts1`，过滤大量背景。两层 MHSA 让属于同一物体的 patch token 交换几何、语义与上下文信息，再从交互后的 token 取 top-`k2` 得到 `Ts2`。最后以 `Ts2` 和 `Ti` 做 MHCA，补回筛选丢失的信息并生成 `Qbal`；这不是简单 top-k，而是“筛选—交互—再筛选—残差融合”。

**BATO 与 GTG。** `Qbal` 分别通过 mask MLP 和 box MLP 得到 `Tg_mask`、`Tg_box`，相加形成 Guiding Token Generation（GTG）的 `Tg`。以 `Ti` 查询 `Tg` 的 MHCA 将同一实例相关的局部 token 向共同指导中心聚合，输出 `Tbal`，因此 Key&Value 同时携带细粒度像素和更强前景结构。

**联合预测。** decoder 不丢弃原 MaskDINO 的检测、分割头。框分支直接读取 refined query；掩码分支仍保留 `Ti` 与高分辨率 CNN 特征，避免 BATO 的前景聚合损伤像素边界信息。

## 实验与证据

实验在 COCO 和 BDD100K 上进行，使用 ImageNet-1K 预训练 ResNet-50 或 ImageNet-22K 预训练 Swin-L。COCO R50、12 epoch 时，MaskDINO 的 `APbox/APmask` 为 45.7/41.4，DI-MaskDINO 为 46.9/42.3；24 epoch 时从 48.4/44.2 提至 49.6/44.8。Swin-L、12 epoch 下也从 52.2/47.2 提升至 53.3/47.9，代价是 FPS 从 3.4 降到 3.0。

主模块消融在 BDD100K/COCO 上一致：DI、BATO 都关闭时为 27.8/24.4 与 45.6/41.2；仅 DI 为 28.8/25.2 与 46.4/42.1；仅 BATO 为 28.3/24.9 与 46.2/41.8；两者同时启用达到 29.5/25.7 与 46.9/42.3。

细粒度消融中，用 `Ti`、`Ts1`、`Ts2`、`Qbal` 指导 BATO，COCO `APbox/APmask` 依次为 46.2/41.8、46.6/42.0、46.7/42.2、46.9/42.3，验证二次筛选和残差融合均有效。去除 GTG 后 COCO 为 46.5/42.2，加入后为 46.9/42.3。失衡耐受测试中，位置 token 约束使 MaskDINO 框 AP 下降 21.8%，DI-MaskDINO 下降 14.7%，证明它确实提高了对早期任务失衡的容忍度。

## 对 YOLO-Agent 的启发

对同时承担框、掩码或姿态任务的 YOLO-Agent，DI-MaskDINO 提醒我们不要只在 loss 权重上处理多任务冲突。可以在 neck 输出后设置两阶段前景 token 选择：先按分类置信筛背景，再用全局交互强化框几何，最后将框/掩码指导重新注入密集特征，分别供检测头和分割头读取。

**Harness。** 对照组为原 YOLO-seg 的共享 neck 与并行 box/mask head；实验组增加 residual double-selection 和 BATO，训练配置不变。记录首个预测层与最终层的 `APbox/APmask` 差值、最终两项 AP、边界 IoU、FPS 和显存。通过阈值：首层框掩码差距缩小至少 20%，最终 APbox 提升至少 0.8、APmask 提升至少 0.3，FPS 下降不超过 8%；若首层变平衡但最终 AP 无提升，或 mask 边界 IoU 下降超过 0.5 点，则判定模块只做了表面重分配。

## 优点

- 从解码器中间层现象出发，问题定位比泛化的“多任务冲突”更具体。
- DI 与 BATO 数据流清晰，主模块和内部 token 选择均有独立消融。
- 在 COCO、BDD100K、R50、Swin-L 和不同训练时长下均有稳定增益。
- 保留原始高分辨率特征供掩码头使用，兼顾框几何与像素细节。

## 局限

- 额外 MHSA/MHCA 带来速度下降，Swin-L 配置更明显。
- 方法围绕检测与实例分割设计，不直接适用于语义分割和全景分割。
- top-k 分类分数仍决定候选入口，低置信真目标可能在第一次筛选中被排除。
- 对“失衡”的度量主要依赖 AP 轨迹，尚缺少梯度冲突或表示分解层面的解释。

## 评分

**8.5/10。** 以 token 数据流解决联合检测分割失衡，证据链完整且易定位实现；计算开销和任务适用范围限制了通用性。
