---
title: "[论文解读] An Empirical Study of Adder Neural Networks for Object Detection"
description: "AdderNet 检测研究：以负 L1 距离替代卷积内积，并用 BN 统计更新、符号梯度和 R-PAFPN 解决分类骨干迁移到检测时的精度与能耗问题。"
tags: ["NeurIPS 2021", "目标检测", "AdderNet", "FCOS", "R-PAFPN", "低功耗检测"]
---

# An Empirical Study of Adder Neural Networks for Object Detection

**论文**：[NeurIPS 官方页面](https://proceedings.neurips.cc/paper_files/paper/2021/hash/37693cfc748049e45d87b8c7d8b9aacd-Abstract.html)  
**PDF**：[NeurIPS 官方 PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/37693cfc748049e45d87b8c7d8b9aacd-Paper.pdf)  
**代码**：论文页面与正文未声明独立官方代码仓库。  
**核心对象**：AdderNet-50、FCOS、Faster R-CNN、RetinaNet、RepPoints、R-PAFPN

## 一句话总结

这篇论文不是简单地把 ResNet 卷积换成“无乘法算子”，而是系统定位 AdderNet 从 ImageNet 分类迁移到检测时的批归一化统计失配和输入特征梯度爆炸，并以更新 BN、符号梯度及低能耗 R-PAFPN 将 FCOS-AdderNet-50 在 COCO 上推进到 37.8 AP。

## 研究背景与问题

AdderNet 的 adder filter 不计算卷积内积，而以局部特征与滤波器之间的负 L1 距离产生响应。若输入窗口为 X、加法核为 W，其输出可概括为 y=-Σ|X-W|。该算子把高能耗乘法换成加减与绝对值运算；论文采用的 45 nm 能耗模型中，32 位浮点乘法约为 3.7 pJ，加法约为 0.9 pJ，因此理论上适合移动端和边缘检测。

困难在于分类可用不等于检测可用。作者把 ImageNet 预训练的 AdderNet-50 接入 FCOS 后发现两个直接故障：检测输入尺度与分类裁剪差异很大，冻结 BatchNorm 会保留错误的 running mean/variance；同时加法层对输入特征的原始梯度幅值过大，深层骨干在检测微调中容易不稳定。常规 FPN 还会在横向融合和 3×3 平滑卷积中重新引入大量乘法，使“无乘法骨干”的节能优势被检测颈部抵消。

## 方法总览

论文的数据流是：图像进入以 adder layer 替换残差块 3×3 卷积的 AdderNet-50；微调阶段允许骨干 BN 更新当前检测数据的统计量，并把加法层传回输入特征的梯度改成 sign 形式；C3、C4、C5 多尺度特征随后送入 R-PAFPN，先完成自顶向下语义传播，再沿新增的自底向上路径把定位细节重新聚合到高层；最终由 FCOS 的分类、中心度和框回归分支产生密集预测。

这种设计把问题拆成三层：adder filter 决定乘法替换，BN 与 sign gradient 保证骨干可训练，R-PAFPN 则处理检测器颈部的精度—能耗折中。论文还把 AdderNet-50 接到 Faster R-CNN、RetinaNet、RepPoints 上验证，说明改进不是只对 FCOS 头有效。

## 方法详解

### 1. Adder filter 替换

作者保留 ResNet-50 的阶段划分和残差连接，只把 bottleneck 中的 3×3 convolution 替换为 adder layer；1×1 卷积仍用于通道变换和捷径投影。加法响应衡量特征块与核的 L1 相似性，不再执行逐元素乘法。这样可以复用 ImageNet 预训练 AdderNet 权重，并与标准 ResNet-50 在相同检测框架下进行配对比较。

### 2. BN 统计更新与输入符号梯度

检测训练通常因 batch 小而冻结 BN，但 AdderNet 对输入分布更敏感。论文在 COCO 微调时解冻骨干 BN 的 running statistics；表 2 中，固定 BN 的 AdderNet-50 FCOS 只有 24.6 AP，允许统计量更新后达到 36.1 AP，增幅 11.5 AP。进一步把加法层对输入特征的梯度替换为 sign(X-W)，避免梯度幅值随 L1 差值扩大；该项在表 3 把 36.1 AP 提升到 37.3 AP，并明显缩小深层 stage 的梯度范数。

### 3. R-PAFPN

R-PAFPN 从 Path Aggregation FPN 的双向融合出发，但将路径中的 3×3 卷积替换为 adder filter，并采用论文验证的自底向上聚合。标准 FPN 的 COCO 结果为 36.6 AP；使用加法颈部但不做重新设计时为 36.1 AP，说明机械替换会损伤融合；完整 R-PAFPN 达到 37.8 AP。其关键不是“再堆一层金字塔”，而是在保持低乘法成本的同时，让低层边缘和位置线索沿 P3→P4→P5 方向回流。

需要注意，论文的训练仍从 ImageNet 分类权重起步，而不是从随机初始化学习全部加法核；因此检测收益同时取决于预训练表征质量和下游适配策略。复现时若更换预训练版本，必须同步报告其 ImageNet top-1，否则无法判断 AP 变化来自算子、预训练还是颈部。

## 实验与证据

- **数据集与协议**：MS COCO 2017 使用 train2017 训练、val2017 评估，主指标为 AP、AP50、AP75、APS、APM、APL；PASCAL VOC 使用 VOC2007 trainval 与 VOC2012 trainval 训练，在 VOC2007 test 上报告 mAP。
- **COCO 主结果**：FCOS-AdderNet-50 配合 R-PAFPN 为 37.8 AP、57.1 AP50、40.6 AP75，估算能耗 244.4 mJ；ResNet-50 FCOS 为 38.5 AP、57.3 AP50、41.6 AP75 和 407.7 mJ。精度差 0.7 AP，但模型能耗下降约 40%。
- **跨检测器验证**：AdderNet-50 在 Faster R-CNN、RetinaNet、RepPoints 中分别达到 37.3、36.5、37.5 AP；相应 ResNet-50 基线为 37.9、36.4、37.6 AP，说明方法在两阶段、单阶段和点表示检测器上都接近乘法骨干。
- **VOC 结果**：AdderNet-50 + R-PAFPN 的 FCOS 为 76.4 mAP，对应 ResNet-50 + FPN 为 77.5 mAP；能耗由 407.7 mJ 降至 244.4 mJ。
- **关键消融**：BN 更新解决最大精度崩塌；sign gradient 再贡献 1.2 AP；R-PAFPN 相比标准 FPN 增加 1.2 AP。分类预训练精度也很重要：ImageNet top-1 从 74.9% 提升到 76.8% 时，COCO AP 从 37.3 增至 37.8。

## 对 YOLO-Agent 的启发

建立 Adder-YOLO Harness 时固定输入分辨率、COCO 划分、增强、优化器、训练轮数、检测头和随机种子，设置五个递进对照：标准卷积骨干；直接替换 3×3 为 adder layer；加入 BN 统计更新；再加入 sign gradient；最后接入双向加法金字塔。每一级都保存 AP、AP50、AP75、APS/APM/APL、各 stage 梯度范数、BN running statistics 漂移、吞吐、设备实测延迟、峰值显存与板端功耗。

通过条件不能只看理论乘加数：三种子平均 AP 与卷积基线差距不超过 1.0，同时板端能耗至少下降 25%，且小目标 AP 不出现超过 1.5 的额外退化，才允许进入候选配方。出现以下任一情况即失败：冻结 BN 后训练看似收敛但 AP 大幅下跌；梯度范数随 stage 深度持续放大；加法颈部比标准 FPN 更慢；能耗估算下降但真实硬件延迟或功耗无改善；关闭 sign gradient 或 R-PAFPN 后结果没有可重复变化。

## 优点

- 用检测实验而非分类能耗推断检验 AdderNet，覆盖四种检测器和两个数据集。
- 把精度损失定位到 BN、梯度和特征金字塔三个可独立消融的环节。
- 在 COCO 上以 0.7 AP 的代价把论文估算能耗从 407.7 mJ 降到 244.4 mJ，权衡清晰。

## 局限

- 能耗依据 45 nm 算子模型估算，不等同于特定 GPU、NPU 或 FPGA 的端到端实测功耗。
- 1×1 卷积和检测头仍含乘法，完整系统不是严格的无乘法检测器。
- BN 更新依赖训练 batch 与分布，迁移到小批量或强域偏移数据时可能再次失配。
- 与 ResNet-50 的差距虽小，但论文没有覆盖更深骨干、现代无锚 YOLO 训练配方和端侧编译器优化。

## 评分

- **方法针对性：9/10**——三个迁移故障均有对应修改和独立消融。
- **实验证据：8/10**——COCO、VOC 与多检测器结果完整，但硬件能耗仍是模型估算。
- **工程迁移：7/10**——adder layer 接口明确，BN、梯度算子和部署内核需要协同实现。
- **YOLO-Agent 价值：8/10**——适合作为低功耗骨干与颈部候选，但必须以真实设备测量决定是否采用。

