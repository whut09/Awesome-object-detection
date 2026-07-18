---
title: "Modality Translation for Object Detection Adaptation without forgetting prior knowledge"
description: "详解 ModTr 如何冻结 RGB 检测器并训练任务驱动的红外输入翻译器。"
tags: ["ECCV 2024", "目标检测", "红外检测", "模态适配"]
---

# Modality Translation for Object Detection Adaptation without forgetting prior knowledge

**论文**：[官方论文页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/12401_ECCV_2024_paper.php)  
**代码**：[官方代码](https://github.com/heitorrapela/ModTr)

## 一句话总结

Modality Translator（ModTr）不微调 COCO 预训练的 FCOS、RetinaNet 或 Faster R-CNN，而训练 U-Net 将 IR 输入变为三通道 RGB-like 表示，并用 Hadamard product 把翻译结果作为像素门控作用于红外图像，从而适配新模态并完整保留原 RGB 检测能力。

## 研究背景与问题

RGB 检测器直接处理红外图像会遭遇巨大分布偏移；常规 fine-tuning 能恢复 IR 性能，却破坏 COCO 知识。为每种模态复制完整 detector 又使内存线性增长。CycleGAN、CUT、FastCUT 关注视觉重建，不保证输出适合检测；HalluciDet 虽用检测监督，仍需源 RGB 数据和预微调。

## 方法总览

每种模态 `d` 配置翻译器 `hϑ^d`。单通道 IR 经 U-Net 输出同分辨率三通道权重图，与输入经非参数融合 `Φ` 后送入冻结的 RGB 检测器 `fθ`。训练只把原检测分类和回归损失反传到 `ϑ`，检测器参数 `θ` 固定。部署时保留一个 detector，并按模态路由到对应 ModTr。

## 方法详解

目标为 `LModTr=Ldet(fθ(Φ(hϑ(x),x)),Y)`。因为监督来自框，翻译器无须生成好看的伪 RGB，只需激活原检测器知识。最佳 **ModTr⊙** 使用 Hadamard product：`hϑ(x)` 充当逐像素门控图，压制非目标区域并突出行人、车辆。论文可视化也显示中间表示会模糊无关物体，任务价值高于图像美学。

知识保存实验比较 N-Detectors、合并微调的 1-Detector、N-ModTr-1-Detector。最后一种保持单个原始检测器，并为 LLVIP、FLIR 各训一个小翻译器，以输入节点隔离模态变化，避免共享检测权重互相覆盖。

## 实验与证据

LLVIP 使用 12,025 对训练、3,463 对测试 RGB/IR 图像，只标注行人；对齐 FLIR 使用 4,129 对训练、1,013 对测试图像，评估 bicycle、car、person。ModTr⊙ 在 LLVIP 的 FCOS/RetinaNet/Faster R-CNN AP 为 57.63/54.83/57.97；HalluciDet 为 28.00/19.95/57.78。FLIR 上 ModTr 达 35.49/34.27/37.21，高于 FastCUT 的 24.02/22.00/26.68。

与全量微调相比，ModTr 在 FLIR 三个 detector 上分别提升 7.52、5.81、6.28 AP；LLVIP 上接近微调。遗忘实验中，1-Detector 在 COCO 的 AP 几乎归零，仅 0.33/0.29/0.40；N-ModTr-1-Detector 完整保持 38.41/35.48/39.78。翻译器由 ResNet34 24.4M 缩至 MobileNetV2 6.6M 后，Faster R-CNN 在 FLIR 仍有 36.77 AP，对比默认 37.21。

ModTr 与常规 image-to-image translation 的差异在监督闭环。CycleGAN、CUT、FastCUT 必须访问源 RGB 图像，却在 LLVIP 的三个 detector 上只有 14.30–26.54 AP；ModTr 只需要目标 IR 与框，直接通过冻结 detector 的任务损失训练。HalluciDet 同样使用检测目标，但要先在源域微调，其 FCOS、RetinaNet 结果分别只有 28.00、19.95，显示源知识被改变后再做幻觉映射并不稳定。

LLVIP 与 FLIR 的差异也揭示适配边界。LLVIP 是固定监控视角、单一 pedestrian 类，ModTr 与 fine-tuning 基本持平；FLIR 来自车载移动相机，背景变化更复杂，ModTr 在三个 detector 上均明显超过全量微调。说明冻结的 COCO 表征在多类移动场景中仍有价值，而翻译器把传感器差异隔离在输入端，避免小数据重写高层类别知识。

参数实验给出实际服务成本。以 Faster R-CNN 41.8M 为例，复制 RGB/IR 两个 detector 约需 83.6M 参数；一个 detector 加 MobileNetV3-small translator 约 44.9M。FCOS 从双模型 66.4M 降到 36.3M，RetinaNet 从 68M 降到 37.1M。MobileNetV2 在 LLVIP/FLIR 为 56.15/36.77 AP，与 ResNet34 的 56.35/37.21 很接近，证明默认 U-Net 编码器并非越大越好。

论文还联合微调 ModTr 与 detector，FLIR 性能可进一步提升，但这会失去“单一不可变服务器”的核心保证。因此应把该实验视为另一种低成本 fine-tuning 初始化，而不能与知识保存主设定混用。中间图像并不自然，作者尝试加入视觉质量损失也未得到确定检测收益，复现时不应以 PSNR、SSIM 代替 AP 判断成功。

N-ModTr-1-Detector 的平均 AP 在 FCOS、RetinaNet、Faster R-CNN 上分别为 43.84、41.52、44.98，高于 N-Detectors 的 41.25、39.24、43.44。优势并非每个单域都最大：Faster R-CNN 在 LLVIP 上 N-Detectors 为 59.62，ModTr 为 57.97；但 ModTr 在 FLIR 更强且完全保留 COCO，因此综合服务更好。评价多域系统应看每域下限、平均值和原域保持三者，而非只挑目标域最高点。

Hadamard 门控需要数值保护。翻译器最后使用 Sigmoid，使权重落在 0–1，理论上只能保留或衰减原输入，不能直接把像素放大到任意范围；三通道权重仍可形成颜色样式差异。这一限制可能帮助稳定训练，也可能妨碍极暗区域增强。复现可增加 `2·sigmoid` 或残差 `1+δ` 对照，但若改变输出范围，就必须检查 detector 预训练 normalization 是否仍匹配。

## 对 YOLO-Agent 的启发

面向热成像、深度、边缘图或新相机时，可优先训练小型输入适配器，并把原域回归测试设为硬约束。输出不必符合图像质量审美，应围绕检测损失和部署成本优化；多域服务让路由器选择 translator，而不是动态重载整套 YOLO 权重。

### 论文专属 Harness

- **对照组**：同一 COCO 预训练 YOLO 比较 IR 零样本、全量微调、head-only、LoRA、U-Net ModTr⊙、MobileNetV2 ModTr⊙。
- **观测指标**：IR AP/AP50/AP75、原 COCO AP、适配参数量、切换时间、端到端延迟及目标/背景门控响应比。
- **通过阈值**：IR AP 距全量微调不超过 1.5 点或在 FLIR 反超，原 COCO AP 下降不超过 0.1，适配参数低于复制 detector 成本的 25%。
- **失败判断**：原域 AP 下降超过 0.5、门控饱和，或 IR 增益低于 LoRA 且延迟高 15% 以上，即改用独立模型。

服务端路由还需处理未知域。若来自新相机的 IR 被错误送入 LLVIP translator，冻结 detector 虽不会遗忘，但输出可能失真。可在 translator 前增加轻量域识别或输入统计校验，并以 COCO 原图绕过翻译器。每次新增 translator 都应运行全部旧域回归；这种测试成本远低于重训 detector，却是维持“一个不可变模型服务”承诺的必要条件。

三个 detector 使用相同数据顺序与增强，并以三种随机种子报告均值和标准差。LLVIP 上 Faster R-CNN 的 ModTr 标准差 0.85，低于 fine-tuning 的 1.23；FLIR 三组也较稳定。Harness 应保留方差门槛，避免只接受平均 AP 略高但训练不稳定的 translator。

## 优点

- 冻结检测器从机制上阻断灾难性遗忘，COCO 回归直接证明。
- 不需要配对 RGB 目标图像，三种 detector 与两个 IR 数据集趋势一致。

## 局限

- 每个新域仍需单独 translator，且依赖新模态框标注。
- 像素级 U-Net 增加前处理延迟，极少信息模态仍不及微调。

## 评分

- **问题重要性**：★★★★★
- **方法清晰度**：★★★★★
- **实验证据**：★★★★★
- **工程可迁移性**：★★★★★
- **YOLO-Agent 参考价值**：★★★★★
