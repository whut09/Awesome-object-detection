---
title: "[论文解读] LLM-Assisted Semantic Guidance for Sparsely Annotated Remote Sensing Object Detection"
description: "原创中文解读 RSST：用离线 LLM 类别提示约束密集伪标签分配，并以 AHR 稳定稀疏标注下的负样本学习。"
tags: ["ICCV 2025", "稀疏标注", "遥感检测", "大语言模型", "伪标签"]
---

# LLM-Assisted Semantic Guidance for Sparsely Annotated Remote Sensing Object Detection

**论文**：[CVF 官方论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Liao_LLM-Assisted_Semantic_Guidance_for_Sparsely_Annotated_Remote_Sensing_Object_Detection_ICCV_2025_paper.html)  
**代码**：[官方代码](https://github.com/wuxiuzhilianni/RSST)  
**发表**：ICCV 2025  
**类别**：Sparsely Annotated Detection, LLM Guidance, Oriented Detection

## 一句话总结

该框架先用适配遥感图像的 Vicuna 视觉语言模型离线判断每图可能出现的类别，再以类别提示约束教师网络的密集像素伪标签分配，并用 Adaptive Hard-Negative Reweighting 降低“未标目标被当背景”的高置信负样本损失。

## 研究背景与问题

稀疏标注检测保留同一图像中的少量实例，其余真实目标没有框。对遥感密集场景而言，未标目标会被一阶段检测器当作负样本；Dense Pseudo-Label 虽保留原始 logits 的丰富监督，却常用全局 top-k 或分层 top-k，类别数量与平均置信度不一致时，少数高置信类别会垄断伪标签，低置信但数量多的类别得不到训练。论文将问题拆成两部分：在分配前先判断图像包含哪些前景类别；在监督分支中识别可能是漏标目标的高置信“负样本”，降低其破坏性。

## 方法总览

框架基于 Multi-Branch Input、EMA teacher-student 和 Rotated FCOS。LLM-Assisted Semantic Prediction（LSP）离线把 504×504 图像送入 CLIP-ViT L/14，插值位置编码得到 1296 patches，经 MLP 将 1024 维视觉嵌入映射到 Vicuna-v1.5 7B 的 4096 维空间；结构化问题要求从 DOTA 类别集合或 none 中输出一到多个类别。稀疏标注图像的最终提示取 LLM 预测与已标类别并集。Class-Aware Label Assignment（CLA）在教师 logits 上联合三类选择：提示类别且超过阈值的像素、全局高置信 top-k、每类 top-k；稀疏标注分支再加入真值框内像素，去重后作为强增强学生的正伪标签。Adaptive Hard-Negative Reweighting（AHR）在监督分支将高置信负样本乘较小权重 w，缓解漏标目标造成的错误背景监督。

## 方法详解

对无标数据，CLA 先用类别提示过滤语义不符的预测，再保留全局强响应以免 LLM 漏类，同时做每类 top-k 平衡类别覆盖；三集合取并集而非串行裁剪，所以兼顾语义先验、整体质量与类别均衡。对稀疏标注数据，真值像素直接加入并集，保证人工监督不会被 LLM 或教师置信度覆盖。弱增强图像产生 teacher logits 与伪标签，强增强图像进入 student，EMA 更新教师，形成循环优化。

AHR 作用于原监督分支。正样本和普通负样本沿用 focal 风格项；当负样本预测前景置信度超过阈值 `thr` 时，认为其可能是未标目标或混淆背景，将损失额外乘 w。论文最优训练集配置为阈值 0.9、w=0.15，目标不是删除全部困难负样本，而是在保留背景辨别能力的同时减少错误梯度和 mAP 曲线突降。

## 实验与证据

DOTA 含 2,806 图、188,282 实例、15 类，按类别保留 1%/2%/5%/10% 标注后切为 1024×1024；HRSC2016 按 10% 标注构造。DOTA 上本文 mAP 为 56.64/59.37/63.95/65.11；Dense Teacher 在 2%/5%/10% 为 47.72/50.82/58.13，提升 11.65、13.13、6.98。HRSC2016 mAP 66.80，高于 Dense Teacher 64.86。组件消融中，基线在 2%/5%/10% 为 47.72/50.82/58.13；仅 AHR 为 55.38/61.60/62.93；加入 LSP+CLA 为 58.73/63.65/64.77；再加 MBI 为 59.37/63.95/65.11。分配消融在 5% 时，无提示 61.97、LLM 提示 63.65、真值提示上界 65.04。原始 LSP 在 21,046 张图中 exact 4,861、partly 3,771、errors 5,471；与稀疏标注并集后正确预测增加，说明提示有效但远非完美。

## 对 YOLO-Agent 的启发

该方法可迁移到旋转 YOLO 的密集 logits，但 LLM 只应作为离线类别候选器，不能替代像素定位。Agent 需要缓存提示版本、成本和错误类型，并保留无提示与真值提示上下界。

**机制特定 Harness**：**对照组**从监督稀疏训练、Dense Teacher、Dense Teacher+AHR、CLA 无提示、通用 LLM 提示到论文 LSP 逐级增加组件，并保留真值类别提示作为不可部署上界。**指标**按 1%/2%/5%/10% 标注率、类别频次、目标尺度、旋转角、图像密度及 LLM exact/partly/error 分桶，报告 OBB mAP、AP50、AP75、逐类伪标签 precision/recall、未标目标误杀率与训练曲线突降次数；提示删类、加错类、输出 none 以及 top-k、`thr`、`w` 扫描必须使用同一教师快照。**失败判断**针对语义分配链：LSP 相对 CLA 无提示在 2% 和 5% 标注率的平均增益不足 0.8 mAP，或稀有类伪标签 recall 下降超过 3 个百分点，或 AHR 使 AP75 比无 AHR 低 0.3 以上，或错误提示导致误杀率翻倍，任一发生即回退到无提示 Dense Teacher。

## 优点

- LLM 只提供图像级类别先验，定位仍由检测器 logits 完成，职责边界清楚。
- CLA 同时处理无标与稀疏标注图像，并显式加入人工真值像素。
- AHR 对漏标目标造成的错误负监督给出直接、可消融的修正。

## 局限

- LSP 错误数量仍高，提示质量依赖模型、问题措辞与遥感适配。
- 离线 7B 模型增加预处理成本，类别集合变化时需重新生成提示。
- 主要验证 DOTA 与 HRSC2016，跨传感器、跨模态和开放类别尚未覆盖。

## 评分

**8.7/10**。语义提示、密集分配和负样本稳定化形成完整训练链，低标注率增益显著；但 LLM 提示错误与离线成本必须被纳入部署判断。
