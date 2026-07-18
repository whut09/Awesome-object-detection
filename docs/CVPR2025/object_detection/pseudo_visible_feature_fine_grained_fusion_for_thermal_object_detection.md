---
title: "[论文解读] Pseudo Visible Feature Fine-Grained Fusion for Thermal Object Detection"
description: "以 Pearl-GAN 伪可见特征、VMamba 多层增强和 Graph-Mamba 级联融合改进纯热红外推理。"
tags: ["CVPR 2025", "热红外检测", "跨模态融合", "VMamba", "图网络"]
---

# Pseudo Visible Feature Fine-Grained Fusion for Thermal Object Detection

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Pseudo_Visible_Feature_Fine-Grained_Fusion_for_Thermal_Object_Detection_CVPR_2025_paper.html)  
**任务**: 热红外目标检测 / 训练期跨模态知识补偿

## 一句话总结

PFGF 先用 Pearl-GAN 把热图转成伪可见图并保留生成器潜码，再以 VMamba 提炼热—伪可见判别特征，最后用 Graph-Mamba Fusion 在节点级过滤不可靠生成信息并从浅层到深层级联传递，测试时仍只输入热图。

## 研究背景与问题

热红外在夜间稳定，却缺少颜色、纹理和细粒度边缘。多光谱检测通常要求测试时同时具备对齐可见光，传感器成本高；热到可见翻译可以补充知识，但伪可见图含有伪色、结构漂移和背景噪声，直接拼接或注意力融合会把错误内容送入检测器。论文要解决的是：利用未配对可见图训练翻译先验，同时只保留对热检测有用的生成信息，并让低层细节逐级影响高层语义。

## 方法总览

第一阶段训练 Pearl-GAN 热到可见模型 `Φ`，输出伪可见图 `xv` 和编码潜特征 `z`。第二阶段冻结 Pearl-GAN，将热图与伪可见图拼接后输入 CSPDarknet，得到 `c3/c4/c5`；VMamba-based Multi-level Feature Enhancement（MEM）生成判别特征 `f3/f4/f5`。Graph-Mamba Fusion（GMF）把 `z` 与 `f3` 细粒度融合，并通过 Cascade Knowledge Integration（CKI）把融合知识依次传到 `f4、f5`，最后经 neck 与 YOLOX 解耦头检测。

## 方法详解

GMF 把每个尺度特征经不同池化粒度构造成 3 个节点，边由节点差值卷积得到，消息权重使用边的 Sigmoid。生成潜码与 `f3` 的对应节点先进入 Inter-Mamba：两路序列各自经过归一化、线性层、卷积和 SSM，热特征产生门控权重，对两路隐藏状态之和做选择性融合，目的是抑制伪可见噪声而非平均两种模态。

同一判别子图内，节点通过 GRU 聚合邻居消息，再经 VMamba 更新；节点拼接卷积得到 leader node。CKI 用 leader node 的全局池化注意力调制下一尺度节点，实现 `f3→f4→f5` 的级联知识传递。最后 `gi` 与原 `fi` 残差相加，送入 neck。推理虽然内部仍由 Pearl-GAN 从热图生成伪可见信息，但外部输入不需要真实可见光相机。

## 实验与证据

- 数据集为 FLIR-aligned（4,129/1,013 训练测试对、3 类）、LLVIP（12,025/3,463、单人类）和 Autonomous Vehicles（6,009/1,503、5 类）。输入 640×640，第二阶段训练 8 epoch。
- FLIR 上 PFGF 达 `47.1 mAP、84.8 mAP50、44.6 mAP75`，超过热模态 TIRDet/IDA，也以 mAP 略高于需要双模态测试的 Fusion-Mamba `47.0`。
- 消融中完整模型在 FLIR/LLVIP/AV 为 `47.1/67.3/45.9 mAP`；去掉 Pearl-GAN 为 `45.2/65.0/44.4`，去掉 GMF 为 `45.6/65.6/44.3`。Inter-Mamba 的影响大于 CKI。
- 融合策略对比中 GMF 在三数据集均最佳；FLIR 上比 Add 的 `42.6` 和 FMB 的 `45.7` 更高。将 Mamba 全换成 Transformer 后，GMTF 为 `46.2 mAP/604.1 GFLOPs`，PFGF 为 `47.1/409.0 GFLOPs`。

## 对 YOLO-Agent 的启发

- Harness 对照热图基线、热+伪图拼接、Pearl-GAN+MEM、无 Inter-Mamba、无 CKI、完整 PFGF，并保证 Pearl-GAN 权重固定一致。
- 分昼夜、低对比度、行人遮挡、伪可见翻译质量切片；总体 mAP 上升但夜间或低质量翻译切片误检增加时判失败。
- 记录翻译耗时、GMF FLOPs、端到端 FPS 与显存，不能只按“单热输入”宣称低成本。Pearl-GAN 输出结构漂移、leader node 导致高层过度激活或热-only 基线更稳时停止采用。
- 与真实双模态输入分开报告：PFGF 的价值是无需可见光传感器，不应把其结果等同于使用真实 V+T 的上限。

## 优点

- 明确区分生成特征与判别特征，并针对伪可见噪声设计门控融合。
- Graph-Mamba 同时建模节点关系、跨模态交互和跨尺度传递。
- 在三种热检测数据上有模块、融合器和效率对照。

## 局限

- 推理仍运行 Pearl-GAN，所谓纯热输入不代表低延迟。
- 翻译模型域外失真会直接污染后续特征，失败检测机制不足。
- 训练分两阶段且模块较多，复现和端侧部署复杂。

## 评分

- **创新性**: ★★★★☆
- **证据强度**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★☆☆
