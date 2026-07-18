---

# Towards Rotation Invariance in Object Detection
title: "Towards Rotation Invariance in Object Detection"
description: "以最大内接椭圆生成旋转后的轴对齐框，并用 Rotation Uncertainty Loss 抑制高不确定角度标签噪声的目标检测方法。"
tags:
  - 目标检测
  - 旋转不变性
  - 数据增强
  - 标签噪声
  - ICCV-2021
---

- **论文链接**：[ICCV 2021 官方论文页](https://openaccess.thecvf.com/content/ICCV2021/html/Kalra_Towards_Rotation_Invariance_in_Object_Detection_ICCV_2021_paper.html)
- **官方代码**：论文未提供作者 GitHub 官方代码仓库；实现与补充材料请见[官方论文页](https://openaccess.thecvf.com/content/ICCV2021/html/Kalra_Towards_Rotation_Invariance_in_Object_Detection_ICCV_2021_paper.html)。

## 一句话总结

给定原始轴对齐框 \(b^0\) 和旋转角 \(\theta\)，方法先把框内未知目标形状建模为最大内接椭圆 \(S_{\text{ellipse}}\)，执行 \(S_{\text{ellipse}}\rightarrow\mathcal R^\theta(S_{\text{ellipse}})\rightarrow\mathcal B(\cdot)\)，得到新的轴对齐训练框。随后，Rotation Uncertainty（RU）Loss 根据角度计算确定性阈值 \(C(\theta)\)，在标签不可靠而预测框已足够接近时关闭回归损失，使低旋转角学到的框先验能够迁移到接近 \(45^\circ\) 的高不确定区域。

实验同时覆盖一阶段 RetinaNet 与二阶段 Faster R-CNN，主要使用 Detectron2、PyTorch 和 COCO val2017，并扩展到 Pascal VOC、Transparent Object Bin Picking、Synthetic Fruit Dataset、Oxford Pets，共五个数据集。对照包括 No Rotation、沿用已久的 Largest Box、Random Boxes、Scaled Octagon、RotIoU、Ellipse、Ellipse+RU Loss，以及利用分割形状获得的 Perfect Labels；评价指标为 AP、AP50、AP75，并额外分析标签 Expected IoU 与 Label AP。

COCO 消融中，测试旋转 \(30^\circ\) 时 Largest Box、Ellipse、Ellipse+RU Loss（\(\delta=10^\circ\)）的 AP 分别为 18.47、29.95、32.72；同一 RU 配置在 \(0^\circ/10^\circ/20^\circ/30^\circ\) 获得 39.33/38.31/36.00/32.72，优于 \(\delta=15^\circ\) 的 39.14/38.19/35.78/32.50。标签质量实验中，\(20^\circ\) 旋转下 Ellipse 的 Label AP75 为 56.2，而 Largest Box 仅为 25.6，直接揭示宽松标签对高 IoU 指标的破坏。

## 研究背景与问题

核心困难不是图像如何旋转，而是只有轴对齐框、没有目标轮廓时，旋转后的正确框并不唯一。同一个 \(b^0\) 可以由大量不同形状产生；这些形状旋转后会对应不同的轴对齐包围框。Largest Box 把原框本身视作目标形状，旋转后再取外接框，因此保证包含目标，却系统性高估面积。检测 AP 依据 IoU 判断真阳性而非“是否完全包含”，所以这种保守策略反而制造了严重标签噪声。

作者把所有可能旋转框表示为集合 \(Q^\theta\)，并将目标改写为：寻找一个候选框，使其与可能真实框之间的期望 IoU 最大。由于无法直接获得各候选框成为真实框的概率，论文采样触碰原框四边的随机形状，用平均 IoU 近似 Expected IoU；再对形状、旋转角和长宽比联合进行可微优化。梯度上升的稳定解收敛到最大内接椭圆，而不是 Largest Box。

## 方法总览

椭圆由原框中心 \((x_c,y_c)\)、宽 \(b_W^0\) 和高 \(b_H^0\) 定义：

\[
\frac{(x-x_c)^2}{(b_W^0/2)^2}+
\frac{(y-y_c)^2}{(b_H^0/2)^2}=1.
\]

旋转该椭圆并取轴对齐外接框即可生成标签，代码复杂度与 Largest Box 基本相同。它在统计意义上更接近各种潜在真实轮廓旋转后的包围框，也不会依赖 COCO 的具体形状分布；论文验证了直接使用 COCO 形状采样与随机形状优化的表现近似。

但椭圆可能低估细长、不规则或接近矩形目标的尺寸，且这种误差在 \(45^\circ、135^\circ、225^\circ、315^\circ\) 附近最大。RU Loss 因而定义

\[
C(\theta)=1+\frac{1-\cos(4\theta)}{2\cos(4\delta)-2},
\]

并将阈值下限限制为 0.5；\(\delta\) 表示 \(C(\theta)=0.5\) 对应的角度。模型依据预测框与标签的接近程度决定是否继续施加回归损失，避免在最不确定角度强行拟合错误框。

## 方法详解

训练使用单张 P100，批量大小为 3；为匹配公开预训练基线，训练时长约为 Detectron2 默认配置的五倍。所有实验从均值 \(0^\circ\)、标准差 \(15^\circ\) 的正态分布采样旋转角。COCO 测试通过分割标注生成旋转后的真实框，并将 val2017 的旋转角按 \(10^\circ\) 分桶；其他四个数据集缺乏完整分割标注，因此仅在标准 \(0^\circ\) 测试集评估。

COCO 的 \(30^\circ\) 测试中，Ellipse+RU Loss 达到 AP/AP50/AP75 32.72/51.60/33.97；No Rotation 为 29.19/45.37/30.39，Largest Box 为 18.47/46.30/10.91，Perfect Labels 为 34.08/51.83/36.02。该方法的 AP50 已非常接近分割标签上界，同时在严格的 AP75 上远胜 Largest Box。

## 实验与证据

四个常规测试集上，Largest Box 多数情况下比不旋转更差，而完整方法均获得正增益。Transparent Object Bin Picking 最具代表性：No Rotation 的 AP75 为 54.3，Largest Box 降至 28.45，下降 47.6%；Ellipse+RU Loss 提升至 56.76，相对增益 4.53%。Pascal VOC 上 AP 从 51.94 提升至 52.89，AP75 从 56.54 提升至 57.97；Oxford Pets 上 AP75 从 88.76 提升至 91.09。

结果说明主要收益并非来自增加更多样本，而是来自修正增强后的监督信号。Expected IoU 与检测性能呈强相关；所有备选标签方法都明显优于 Largest Box，而 Ellipse 在除 \(0^\circ\) 外的测试角度取得最佳标签生成结果。RU Loss 则进一步弥补椭圆先验在大旋转角下残留的不确定性。

## 对 YOLO-Agent 的启发

Harness 应固定同一检测器、初始化、训练轮数和角度采样，设置 No Rotation、Largest Box、Ellipse-only、Ellipse+RU（\(\delta=45^\circ/30^\circ/15^\circ/10^\circ\)）五类核心控制组；COCO 另加入 Perfect Labels 作为分割监督上界。不得同时改变 Mosaic、尺度增强、框匹配规则或骨干网络，否则无法把差异归因于旋转标签策略。

指标必须逐角度报告 COCO 的 AP、AP50、AP75，并记录 Expected IoU 或 Label AP50/AP75；外部验证至少包含 Transparent Object Bin Picking，重点观察高 IoU 定位质量。具体失败判据为：在 COCO \(20^\circ\) 或 \(30^\circ\) 上，Ellipse+RU 的 AP 或 AP75 未同时超过 No Rotation 与 Largest Box；或在透明物体数据集上 AP75 未超过 No Rotation 的 54.3。任一条件出现，都意味着没有复现论文关于旋转泛化或高精度定位的核心结论。

## 优点

方法只处理轴对齐框旋转增强，并不解决旋转框检测，也没有搜索最佳增强概率或角度分布。椭圆是未知形状下的统计先验，对矩形、强凹结构和极端长宽比物体可能低估外接范围；RU Loss 虽能减少过拟合，但也可能停止对某些仍可改进预测的回归监督。

## 局限

最重要的启示是：几何增强不能只保证标签“覆盖目标”，而应直接优化与最终评价规则一致的标签质量。将标签生成写成 Expected IoU 优化，再用角度确定性调节回归监督，比单纯扩大框更符合 AP/AP75 的需求。这一设计无需更换检测架构，可作为 RetinaNet、Faster R-CNN 或 YOLO 类轴对齐检测器的数据管线与损失插件。

## 评分

最大内接椭圆修正了 Largest Box 的系统性框膨胀，RU Loss 又抑制了高旋转不确定区域的监督噪声。二者组合在五个数据集、一阶段与二阶段检测器上均显示稳定收益，并在 COCO AP50 上接近使用真实分割轮廓生成标签的上界，以极低实现成本提供了可靠的旋转增强方案。
