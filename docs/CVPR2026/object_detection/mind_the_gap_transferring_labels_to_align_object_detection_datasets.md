---
title: "Mind the Gap: Transferring Labels to Align Object Detection Datasets"
description: "解析 LAT 如何用特权候选生成器和类别感知语义特征融合，将异构检测数据集投影到指定目标标签空间。"
tags: ["CVPR 2026", "多数据集检测", "Label-Aligned Transfer", "PPG", "SFF"]
---

# Mind the Gap: Transferring Labels to Align Object Detection Datasets

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Kennerley_Mind_the_Gap_Transferring_Labels_to_Align_Object_Detection_Datasets_CVPR_2026_paper.html)  
**官方代码**: 未发现论文声明的官方代码  
**任务**: 异构目标检测数据集的标签空间迁移

## 一句话总结

Label-Aligned Transfer（LAT）先让各数据集专属检测器跨域生成伪标签，再用 Privileged Proposal Generator 把真值与多标签空间伪框共同送入 RoI，最后由 Semantic Feature Fusion 对重叠候选做类别感知注意力，把源数据标注直接转写到指定目标数据集的语义与框规范。

## 研究背景与问题

Cityscapes、Waymo、nuImages 即便都描述道路场景，也可能把 cyclist 与 bicycle 分开或合并，把 car、bus、truck 统称 vehicle，甚至采用不同框边界习惯。直接合并会把合法目标当背景；手工 taxonomy 只能处理已知类别名，难以解决一个大框拆成两个框等空间差异。LAT 的目标不是建立平均意义上的通用标签集，而是固定一个部署目标标签空间，把其他数据集转成该空间。

论文采用 many-to-one 投影：每个数据集先训练 upstream detector，在其他数据集图像上生成该数据集标签空间的 pseudo-label。由于同一图像没有两套人工真值，LAT 必须依靠目标域真值锚点、跨模型重叠候选和类别分数，在无成对映射监督下学习语义与定位对应关系。

## 方法总览

LAT 基于两阶段检测器，但用冻结 DINOv2 替代普通 backbone，并以 Privileged Proposal Generator（PPG）替换 RPN。PPG 不预测，只收集当前图像的 ground truth、来自其他 label space 的 pseudo boxes 及其类别，轻微抖动或随机移除真值后作为 RoI proposals。重叠候选进入 Semantic Feature Fusion（SFF），将区域视觉特征与类别分数共同聚合，再由分类/回归头输出目标标签空间预测。

## 方法详解

PPG 保留各数据集类别离散性，即使文本名相同也不直接合并。它把 region-label 对同时送入 RoI 和 SFF，使“Cityscapes car”和“Waymo vehicle”能够因空间重合建立关系，又避免预先假设二者完全等价。训练批次中仍保留当前数据集 ground truth，因此伪标签噪声不会像纯 EMA teacher 那样自我循环。

SFF 对 `M` 个 RoI 特征计算缩放点积注意力 `A=QKᵀ/√d`。分类分支的 value 来自分类分数投影 `Vc`，特征分支来自区域特征 `Vr`。真值 proposal 的置信权重设为 1，伪框权重为其最大类别分数；特征分支用该权重调节注意力。分类分支则把每行最大注意力限制到 `T=1/√N`，迫使模型汇聚多个数据集一致支持的重叠候选，抑制孤立伪框。训练计算 loss 前只开放当前批次数据集类别，推理时只开放指定目标 label space。

这种 class scaling 的含义不是平均所有来源。若三个数据集都在相近位置预测某个对象，单个候选不能垄断注意力，分类证据会在多个重叠 proposal 间汇集；若只有一个低置信伪框独立出现，它既缺少跨源支持，又在 feature branch 被置信权重削弱。与普通 attention 只根据视觉相似度聚合相比，SFF 把“谁以什么标签提出了这个框”也纳入区域表示。

LAT 的最终产物分两步使用。首先训练 LAT 本身，把其他数据集图像转换成目标 label space 的 refined pseudo-label；然后用这些新标签与目标真值训练 downstream detector。论文的 AP 增益来自后者，而不是要求部署时保留 PPG、SFF 或 DINOv2。这个区分对复现很重要：若直接把 LAT 当最终检测器测试，就会混淆标签迁移质量与下游模型能力。

Cityscapes、nuImages、Waymo 的等量子集实验隔离了“类别粒度差异”与“数据量差异”，而 ACDC、BDD100K、SHIFT 组合则专门检查小目标集能否借助大源集。前者证明语义和空间对齐有效，后者暴露大数据训练不足问题；两类 benchmark 共同说明 LAT 不是无条件把更多图片加入训练就一定获益。

SAM3 对照进一步说明语言或基础模型语义不能独自解决框拆分。它能帮助识别“这里有车辆”，却未必知道目标数据集要求把骑手和自行车拆成两个实例，也不知道 police vehicle 是否应归入 car。LAT+SAM3 略优于任一单独方案，合理的定位是让基础模型补充候选，而由目标真值与 SFF 学习数据集规范。

## 实验与证据

- Class Divergence Benchmark 使用 Cityscapes、nuImages、Waymo，类别数分别为 `8、24、3`，并各取约 3000 张以隔离 taxonomy 差异。Small-to-Large Benchmark 使用 Cityscapes、ACDC、BDD100K、SHIFT。
- FRCNN 上 Cityscapes/nuImages/Waymo baseline 为 `55.2/39.2/44.6 AP`，LAT 达 `60.1/41.7/48.5`；RT-DETR 上 LAT 为 `60.6/39.5/49.6`，优于 RT-DETR baseline 的 `56.8/37.0/45.3` 和 Plain-DET。
- 小到大迁移中，ACDC FRCNN baseline `45.0`，LAT `53.4`，提升 `8.4 AP`。大数据集因训练不足会轻微下降，延长训练后 BDD100K/SHIFT 恢复到 `57.8/71.4`。
- 下游训练策略消融：50/50 batch 为 `57.5/40.5/46.8`，随机 Mixed Batch 为 `58.9/40.0/47.1`，混合预训练后最后 10000 iter 目标域 fine-tune 达 `60.1/41.7/48.5`。
- SFF 消融中无注意力、标准注意力、类别感知 SFF 在三数据集分别为 `57.5/39.8/46.1`、`58.1/40.4/47.0`、`60.1/41.7/48.5`；class scaling 也优于 feature scaling。SAM3 单独迁移不如 LAT，LAT+SAM3 略有进一步提升。

## 对 YOLO-Agent 的启发

对多来源 YOLO 数据，Agent 不应只做类别名字符串映射。更可靠的流程是先保留每个来源的原标签与专属 teacher，让同图上的多空间候选形成“软对应图”，再在目标标签空间训练最终模型。若一辆带骑手的自行车在不同来源出现合框/拆框，Agent 应触发空间结构迁移，而不是强制一对一 class remap。

**Harness**：选 Cityscapes 为目标域，构造“仅目标训练、手工名称映射、普通伪标签、PPG 无 SFF、完整 LAT”五组；另对 cyclist/bicycle、vehicle 子类单独统计。观测目标 AP、TIDE 分类/定位错误、伪框 precision/recall、跨空间重叠一致率、每类混淆与训练时长。通过条件为完整 LAT 至少比普通伪标签高 `2 AP`，合并/拆分类别的定位错误下降 `10%`，目标域稀有类不下降超过 `1 AP`；若增益只出现在共享同名类、长训练仍不能恢复大数据集性能，或孤立伪框权重持续过高，则迁移判定失败。

## 优点

- 同时处理类别语义和框规范，不依赖手工统一 taxonomy。
- ground truth、跨源共识与置信度在 SFF 中有明确作用。
- 对 FRCNN、RT-DETR 和大小不均数据组合都给出验证。

## 局限

- PPG 依赖预先生成的多组伪标签，数据准备和存储成本较高。
- LAT 训练阶段使用特权 proposal，最终标签质量受 upstream detector 覆盖率限制。
- 大源数据可能造成欠拟合，需要额外训练和目标域微调，成本并非恒定。

## 评分

- **创新性**: ★★★★★
- **证据强度**: ★★★★☆
- **工程可用性**: ★★★☆☆
- **YOLO-Agent 参考价值**: ★★★★★
