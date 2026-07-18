---
title: "InsCal: Calibrated Multi-Source Fully Test-Time Prompt Tuning for Object Detection"
description: "解析 InsCal 如何用文本引导风格变换、多源 Grounding DINO 编码器和实例校准熵完成在线检测提示适应。"
tags: ["CVPR 2026", "测试时适应", "InsCal", "Prompt Tuning", "检测校准"]
---

# InsCal: Calibrated Multi-Source Fully Test-Time Prompt Tuning for Object Detection

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Que_InsCal_Calibrated_Multi-Source_Fully_Test-Time_Prompt_Tuning_for_Object_Detection_CVPR_2026_paper.html)  
**官方代码**: 未发现论文声明的官方代码  
**任务**: Multi-source Fully Test-Time Prompt Tuning for Object Detection

## 一句话总结

InsCal 用 Text-Guide Image Augmentation（TGIA）把单张目标图变换成多个源域风格视图，再让多个冻结的 Grounding DINO 图像编码器与共享文本编码器共同优化 prompt，并以 top-1/top-2 间隔加权的 calibrated entropy 抑制普通熵最小化造成的过度自信。

## 研究背景与问题

文本驱动检测器可以识别开放类别，但在夜雨、黄昏雨、雾天和艺术风格域上，零样本 Grounding DINO 的框与类别置信度明显失真。Fully Test-Time Adaptation 要求目标样本在线到达、不能访问源图像；普通 TPT 对多增强预测做熵最小化，错误预测也会被强制变尖锐，导致 mAP 和检测校准误差 D-ECE 同时恶化。

论文进一步指出，单源模型无法覆盖所有域知识。InsCal 假设有多个分别在不同源域微调的 GDINO 模型，以及每个域的一句风格描述；适应过程中冻结全部图像/文本编码器，只更新文本 prompt，因此不会持续改写检测器权重。

## 方法总览

TGIA 输入目标图、目标风格文本和某一源域风格文本，通过方向对比损失让图像嵌入变化方向对齐“目标风格文本嵌入−源风格文本嵌入”，同时用 L2 项保持内容。每个源域生成 `N` 个视图并送入对应源图像编码器。Multi-Source TPT（MSTPT）把 `S×N` 个预测平均，过滤高熵视图，再用 InsCal 的实例校准熵更新 prompt。

## 方法详解

TGIA 不需要源图像。它对目标图随机裁剪，比较增强前后图像嵌入差 `ΔI` 与文本风格方向 `ΔT`，优化变换参数使二者同向；内容约束避免把车辆、行人结构一起改掉。水彩域可只提供类似“a drawing in watercolor style”的描述，因此仍满足 fully test-time 的数据访问约束。

MSTPT 为每个源域保留独立 image encoder，但共享 domain-agnostic text encoder。提示 `p∈R^{L×D}` 经文本编码后，与每个风格视图图像特征计算余弦相似度和温度 softmax；只保留低熵百分位视图，减少坏增强影响。Calibrated Entropy Minimization 再读取每个预测的 `p1st-p2nd`，以 `1+(p1st-p2nd)^φ` 调整熵项：类别间隔大的稳定实例获得更强权重，间隔小的模糊实例被降权，避免把不可靠框硬推向高置信。

多源并不是把模型参数平均。每个 TGIA 视图必须送入与目标风格相对应的源域 image encoder，之后才在预测分布层汇总；共享 text encoder 和 prompt 是各源知识发生交互的唯一位置。这样既保留各源模型在晴天、雨天或艺术域学到的视觉专长，又避免同时更新多个大检测器。高熵视图过滤发生在平均之前，否则一个失败的风格转换会把所有源模型的共识拉向错误类别。

检测校准比分类校准更苛刻，因为“正确”同时要求类别正确且框 IoU 超过阈值。D-ECE 因此按置信度和框属性分箱，比较每个 bin 的实际 precision 与平均 confidence。InsCal 的目标并非单纯把所有分数压低，而是让 margin 大的预测承担更多 prompt 更新、margin 小的预测少影响参数；可靠性图中置信度与真实正确率更接近，才说明校准项完成了任务。

Night Rainy 是检验方法的关键域：原始 GDINO 在该域只有 `13.6 mAP`，意味着大量初始预测本身不可靠，普通熵最小化最容易把错误固化。InsCal 提升到 `20.8` 的同时 D-ECE 降到 `12.2`，说明收益不是仅靠扩大分数；不过该数值仍远未接近白天域，表明完全测试时提示适应不能替代真正的目标域监督。

Art Image 的多源设置也检验了方法是否局限于天气变化。Clipart、Comic、Watercolor 的纹理差异无法只靠亮度或雾化增强描述，TGIA 必须依据文本方向移动高层风格表征。若源风格句子过于笼统，多个源视图可能趋同，MSTPT 的互补性会下降；因此风格描述本身应被当作可验证配置，而非免费元数据。

在线流式评估还应在每张图适应后重置提示作对照，以区分单样本适应和跨样本累积漂移。

两种模式的稳定性结论可能不同。

## 实验与证据

- DWD 包含 Day Clear、Day Foggy、Dusk Rainy、Night Rainy、Night Clear，七类目标；Art Image 使用 Clipart1k、Comic2k、Watercolor2k。指标同时报告 `mAP@0.5` 与 D-ECE。
- DWD 中 InsCal 在 Day Foggy、Dusk Rainy、Night Rainy、Night Clear 分别达到 `37.1/33.2/20.8/38.5 mAP`，D-ECE 为 `10.6/14.5/12.2/13.2`。原始 GDINO 对应 mAP 为 `34.1/29.0/13.6/29.2`。
- Night Clear 组件消融从 EM 的 `33.1 mAP、14.7 D-ECE` 出发；TPT 为 `34.7/14.2`，多源为 `37.5/13.7`，加 TGIA 为 `36.1/13.5`，完整 CEM 达 `38.5/13.2`。该表说明 TGIA 单独不保证最高 mAP，但与校准目标结合后整体最好。
- Day Foggy 开放词汇实验加入 novel 类 Traffic Light，InsCal 达 `37.1 mAP、10.6 D-ECE`，优于 TPT 的 `34.9/13.2`；新类 AP 为 `33.7`，说明 prompt 更新没有完全遗忘开放类别。
- 对比包含 UDA、SFDA、Tent、TPT、DART、C-TPT、ZS-Norm、Penalty、SaLS、O-TPT。InsCal 在多种目标域获得最低或接近最低 D-ECE，并在最困难的 Night Rainy 上把 GDINO mAP 从 `13.6` 提至 `20.8`。

## 对 YOLO-Agent 的启发

若 YOLO-Agent 挂接文本类别编码器，InsCal 提供了一个不会改动视觉骨干的在线适应方案：Agent 先根据环境元数据生成“夜雨→白天晴朗”等风格描述，构造多源视图，只更新 prompt；同时必须把 AP 与 D-ECE并列监控，防止熵下降被误认为检测变好。

**Harness**：选择 DWD Night Rainy 与 Day Foggy，设置零样本、单源 TPT、多源 MSTPT、MSTPT+TGIA、完整 InsCal；额外加入错误源风格文本和无内容保持项两组反事实。观测 mAP50、D-ECE、可靠性图、top-1/top-2 margin、每图更新耗时、提示漂移范数和 novel 类 AP。通过门槛为完整模型相对单源 TPT 至少 `+2 mAP`、D-ECE 下降 `1` 个百分点，novel AP 下降不超过 `1` 点，且单图适应延迟满足系统预算；若只降低熵却提高 D-ECE，错误风格描述不影响输出，或 TGIA 造成框位置明显漂移，则判定失败。

## 优点

- 同时解决多源知识聚合、源图像不可访问和检测置信度失准。
- 只更新 prompt，冻结检测器编码器，适合受限在线环境。
- 同时报告 mAP、D-ECE 与开放词汇结果，证据维度较完整。

## 局限

- 需要为每个源域保存独立模型，并提供可靠的源/目标风格描述。
- 每张测试图生成多源多视图并反向更新 prompt，在线成本高于一次前向。
- 校准权重依赖 top-1/top-2 间隔；当模型一致地高置信犯错时仍可能失效。

## 评分

- **创新性**: ★★★★☆
- **证据强度**: ★★★★☆
- **工程可用性**: ★★★☆☆
- **YOLO-Agent 参考价值**: ★★★★☆
