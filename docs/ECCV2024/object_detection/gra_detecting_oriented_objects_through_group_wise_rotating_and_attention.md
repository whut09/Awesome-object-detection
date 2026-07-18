---
title: "GRA: Detecting Oriented Objects through Group-wise Rotating and Attention"
description: "解析 GRA 以样本级分组旋转卷积核和旋转不变注意力同时建模方向敏感与方向稳定特征。"
tags: [ECCV2024, oriented-object-detection, rotation-equivariance, dynamic-convolution, attention]
---

# GRA: Detecting Oriented Objects through Group-wise Rotating and Attention

## 论文与代码

- 论文：[ECCV 2024 官方页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2600_ECCV_2024_paper.php)
- 代码：作者明确表示训练代码尚未公开；当前可复核范围只有补充材料中的 Group-wise Rotating PyTorch 伪代码，不包含完整 GRA 配置、权重或 RIA 训练流水线。

## 一句话总结

GRA 为每张图预测多组旋转角，把卷积输出通道分组后分别旋转 3×3 核以提取多方向等变特征，再用 Rotation-invariant Attention 把各方向响应压成稳定表示，兼顾旋转框需要的角度敏感定位与类别识别所需的方向不变性。

## 研究背景与问题

遥感旋转检测的矛盾在于：回归角度需要特征随目标方向变化，分类又希望同一类别旋转后语义不变。固定多方向卷积会让所有图像共享同一组角度，无法适配一张图里占主导的朝向；逐通道动态旋转虽然灵活，却显著增加参数与计算。GRA 选择“样本级角度、通道组级核旋转”：同一幅特征图只预测少量角度，每组输出通道共享一个角度，既保留动态性又控制代价。

## 方法总览

Group-wise Rotating（GR）先以 Angle Generator 从输入特征全局池化并预测 N 个角度；标准卷积核按输出通道切成 N 组，每组经对应二维旋转矩阵变换后再与该样本卷积。Rotation-invariant Attention（RIA）随后对多方向响应做注意力聚合，使输入旋转带来的方向维循环变化最终汇成稳定描述。完整 GRA Block 把 GR 的方向等变分支和 RIA 的方向不变分支组合，可替换 ResNet 普通卷积并接入 Oriented R-CNN 等检测头。

## 方法详解

GR 的实现路径在补充伪代码中非常明确：输入 [B,C,H,W] 经 angle generator 得到 [B,N] 角度；每个角度转成 [9,9] 的 3×3 核采样矩阵；权重 [Cout,Cin,3,3] 重排为 [N,9,Cout/N×Cin]，用一次 torch.bmm 完成全部组的旋转；再把 batch 与输出通道合并，以 groups=B 的卷积让每张图使用自己的旋转核。因此它不是旋转 feature map，也不是枚举旋转输入，而是动态改变卷积权重。

RIA 针对 GR 产生的方向响应建立注意力，学习哪些角度分量应保留用于类别语义。作者把二者称为 Dynamic Feature Alignment：GR 对齐目标主方向，RIA 消除剩余方向差。检测数据流是“图像→GRA backbone 的多组动态旋转核→方向响应→RIA 聚合→FPN→旋转 proposal 与分类/回归头”。关键是将方向信息提取和方向信息消除分开，而不是用同一个不变特征同时承担分类和角度回归。

## 实验与证据

论文在 DOTA-v1.0、DOTA-v1.5、DOTA-v2.0、HRSC2016、FAIR1M-v1.0 上验证。以 ResNet-50 与 Oriented R-CNN 为统一设置，DOTA-v1.0 单尺度训练/测试达到 79.54 mAP，多尺度达到 81.17；在 DOTA-v1.5 与 DOTA-v2.0 分别达到 72.30 和 57.15，HRSC2016 的 VOC07/VOC12 指标为 90.30/98.17，FAIR1M-v1.0 为 48.77。与 ARC、ReDet、S2A-Net 等旋转建模方法比较，GRA 在多个数据集同时保持较高精度，说明收益并非只针对单一角度分布。

消融从 Oriented R-CNN 基线 75.87 mAP 出发：加入 GR 达到 77.90，加入 RIA 达到 77.36，完整 GRA 达到 79.54；GR 与 RIA 具有互补性。组数实验显示过少无法覆盖方向，过多则每组通道不足且动态核成本增加，中等组数最佳。效率分析强调旋转过程主要是一批矩阵乘法，补充材料称相对标准卷积额外成本很小，但完整网络仍需同时考虑 angle generator 与注意力支路。

## 对 YOLO-Agent 的启发

在 YOLO 的中高层 CSP/C2f 中只替换部分 3×3 卷积为 GR，并将 RIA 放到 PAN/FPN 入口，可避免低层动态核吞噬延迟预算。DOTA-v1.0 为主场、HRSC2016 为细长船舶外场时，**对照组**需覆盖标准卷积、固定 8 方向旋转核、仅 GR、仅 RIA 与完整 GR+RIA，OBB head 和角度编码保持一致。把 mAP50、mAP75、角度 MAE、长宽比大于 5 的 AP、每 15° 朝向分桶 AP、FPS 定义为**指标**，另将同图旋转 30°/60°/90°，检查分类分数方差与预测角度的等变平移。**失败判断**分两级：RIA 若使分类更稳定却令角度 MAE 恶化超过 1°，视为不变特征污染回归；GR 若在朝向最不均衡切片不优于固定旋转核，或延迟增加超过 10% 而 mAP 增幅不足 1 点，则不纳入组件库。

## 优点

- 核旋转而非图像旋转，避免多次前向并保持空间分辨率。
- 用 GR/RIA 分别处理等变性与不变性，设计目标清晰。
- 可通过批量矩阵乘法实现，具备部署轻量化潜力。

## 局限

- 样本级少量角度难以完整覆盖同图中方向差异极大的密集目标。
- 动态权重对编译器、量化和 TensorRT 融合不如静态卷积友好。
- 官方训练代码尚未公开，RIA 的工程复现风险高于普通模块。

## 评分

- 创新性：9/10
- 实证充分性：8/10
- 工程可迁移性：7/10
- 对 YOLO-Agent 价值：8/10
