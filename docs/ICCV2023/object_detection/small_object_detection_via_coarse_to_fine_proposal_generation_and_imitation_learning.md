---

# Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning
title: "Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning"
description: "解析 CFINet 如何以粗到细候选框生成和在线特征模仿，缓解纯小目标检测中的低质量样本与弱表征问题。"
tags:
  - 小目标检测
  - CFINet
  - 候选框生成
  - 模仿学习
  - 对比学习
  - ICCV2023
---

- **论文链接**：https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_Small_Object_Detection_via_Coarse-to-fine_Proposal_Generation_and_Imitation_Learning_ICCV_2023_paper.html
- **官方代码**：https://github.com/shaunyuan22/CFINet

## 一句话总结

CFINet 是面向“纯小目标”场景的两阶段检测框架，核心数据流为：ResNet-50 与 FPN 产生 \(P_2\) 至 \(P_5\) 特征，Coarse-to-fine RPN（CRPN）先通过面积自适应的 Anchor Mining 从全部金字塔层选择候选锚框，执行粗回归；随后依据粗候选框的偏移，用 Adaptive Convolution 对齐候选框与特征，再完成细回归和前景—背景分类。它不是简单降低 RPN 正样本阈值，而是同时保证极小目标拥有足够锚框，并避免大量低质量锚框主导优化。

检测头旁接 Feature Imitation（FI）分支。模型根据当前预测的类别置信度与预测框 IoU 计算 Instance Quality（IQ），把高 IQ 实例的 RoI 特征在线写入按类别组织的 Exemplar Feature Set；困难实例则从同类 exemplar 中采样正样本，从其他类别及背景 RoI 中采样负样本。当前 RoI 与 exemplar 均经过 Feat2Embed：三个无填充 \(3\times3\) 卷积、两层感知机以及 128 维嵌入层；当前特征路径更新参数，exemplar 路径冻结参数，最终以监督式对比损失拉近同类、推远异类和背景。

实验覆盖驾驶场景 SODA-D 与航拍旋转框场景 SODA-A，基线分别是 Faster RCNN 和 Rotated Faster RCNN，主要指标采用 COCO 式 AP、AP50、AP75，以及 APeS、APrS、APgS。CFINet 在 SODA-D 达到 30.7 AP、60.8 AP50、14.7 APeS，相比 28.9 AP 的基线提升 1.8；在 SODA-A 达到 34.4 AP、73.1 AP50、13.5 APeS。组件消融中，单独 CRPN 为 30.3 AP，单独 FI 为 29.5 AP，二者结合为 30.7 AP，说明候选框质量与特征模仿具有互补性。

## 研究背景与问题

论文针对两个相互关联的问题。第一，小目标区域有限，常规锚框与真值框的最大 IoU 偏低，固定 0.7 正样本阈值导致训练样本不足；直接降至 0.5 虽略增召回，却会引入低质量样本，并使正常尺寸目标的 ARN 从 57.1 降至 54.1。第二，小目标 RoI 缺乏纹理与结构信息，分类和定位容易产生不确定预测，而依赖大目标、离线特征库或直接 L2 对齐的方法可能带来额外模型、维护成本和特征坍缩。

SODA 将小目标进一步划分为 eS、rS、gS，面积范围分别为 \((0,144]\)、\((144,400]\)、\((400,1024]\)。这使论文关注的不只是总体 AP，而是极小目标能否获得足够高质量 proposal，以及表征增强是否真正改善最困难尺度。

## 方法总览

CRPN 的动态正样本阈值为：

\[
T_a=\max\left(0.25,\ 0.20+\gamma\log\frac{\sqrt{wh}}{12}\right),
\]

其中 \(w,h\) 为目标宽高，\(\gamma=0.15\)，12 对应 SODA 的最小尺度定义。目标越小，阈值越宽松；目标增大时阈值平滑提高。所有 \(P_2\) 至 \(P_5\) 层中 IoU 大于 \(T_a\) 的锚框均参与粗回归，不再像 Cascade RPN 那样只保留单一金字塔层的正样本。

CRPN 损失包含粗、细两阶段 IoU 回归损失与仅在细阶段计算的交叉熵分类损失，权重分别设为 \(\alpha_1=9.0\)、\(\alpha_2=0.9\)。在每张图仅保留 300 个 proposal 时，CRPN 达到 42.6 AR、24.6 AReS 和 49.1 ARgS，高于普通 RPN 的 41.2、24.0 和 47.3，也优于 Cascade RPN 的 41.8 AR。

## 方法详解

IQ 将当前模型对某个真值实例的正确类别置信度与定位 IoU 相乘后求平均，因此只有“分类正确且定位准确”的实例才能成为 exemplar。阈值 \(T_{hq}\) 太低会频繁写入不可靠教师特征，太高又会使早期 exemplar 长时间滞留。实验中 \(T_{hq}=0.65\) 得到 30.7 AP；0.50、0.55、0.70 分别得到 30.3、30.6、30.5 AP。

FI 使用监督式对比目标，而不是强制两个 RoI 特征逐元素相同。损失样本集合由同类 exemplar 正样本及异类、背景负样本构成，温度 \(\tau=0.6\) 时达到最佳 30.7 AP；改为 0.10、0.50、0.80 时分别为 30.3、30.4、30.2 AP。FI 仅在训练阶段启用，因此不会增加推理阶段开销。

## 实验与证据

SODA-D 包含 24,828 张图像、278,433 个实例和九类道路目标；SODA-A 包含 2,513 张航拍图像、872,069 个任意方向实例，平均每图约 350 个目标。原始高分辨率图像被裁成 \(800\times800\) patch，步长为 650，再缩放至 \(1200\times1200\)。

训练使用单张 RTX 3090、批量大小 4、12 个 epoch；初始学习率 0.01，在第 8、11 个 epoch 衰减十倍；SGD 动量为 0.9，权重衰减为 0.0001，仅采用随机翻转。SODA-D 上 CFINet 还超过面向微小目标设计的 RFLA：AP 提升 1.0，APeS 提升 1.5，APrS 提升 0.9。

## 对 YOLO-Agent 的启发

若将论文思想交给 YOLO-Agent 验证，不应直接把 FI 损失接到 YOLO 后宣称有效，而应建立四组严格控制实验：原始检测器；仅加入面积动态标签分配；仅加入在线 exemplar 与 Feat2Embed；两者联合。所有组必须固定骨干、输入切片、训练轮数、增强、随机种子和 NMS 参数。

报告总体 AP、AP50、AP75，并强制分列 APeS、APrS、APgS；候选框侧额外报告固定 proposal 数下的 AR、AReS、ARrS、ARgS，以及 IoU≥0.5 的高质量 proposal 数。具体失败判据：联合模型若未超过基线至少 1.0 AP，或 APeS 没有提升，或提升仅来自增加输入分辨率与训练预算，则不能复现论文主张；若动态分配提高 AReS 却令 ARN 下降超过 2 点，也应判定样本质量失衡。

## 优点

关键贡献不是某个孤立模块，而是闭环关系：CRPN 为极小目标提供更多且更准的 proposal，使 IQ 更可信；可信 IQ 又为 FI 提供稳定 exemplar；FI 改善后的 RoI 表征进一步提高分类与定位质量。表 5 中 CRPN 与 FI 联合优于各自单独使用，正是这一闭环的证据。

## 局限

方法仍依赖多个阈值与损失权重，包括 \(\gamma\)、\(T_{hq}\)、\(\tau\) 和 \(\alpha_3\)。其中 FI 权重 \(\alpha_3=0.5\) 最佳，0.25 和 0.75 分别为 30.6 与 30.4 AP。在线 exemplar 也可能受到类别长尾、错误高置信预测和早期训练噪声影响；论文通过 IQ、更新数量上界及冻结 exemplar 嵌入路径缓解，但没有彻底解决教师库陈旧问题。

## 评分

这项工作的实用启示是：小目标检测不能只依靠更高分辨率或更密集锚框，而应分别治理“可训练候选不足”和“RoI 语义不足”。CRPN 将固定阈值改成连续的尺度相关阈值，FI 将静态大目标教师改成随模型状态更新的在线高质量教师。其结果表明，面向小目标设计 proposal 生成机制，再用质量感知的对比模仿增强困难实例，比单纯放宽正样本条件更稳定。
