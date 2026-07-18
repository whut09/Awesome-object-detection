---

# H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection
title: "H2RBox-v2：融入对称性以增强水平框监督的旋转目标检测"
description: "通过翻转—旋转一致性从目标对称性自监督学习角度，并以 CircumIoU 结合水平框弱监督学习位置、尺度与类别的 NeurIPS 2023 论文笔记。"
tags:
  - 旋转目标检测
  - 弱监督学习
  - 自监督学习
  - 遥感影像
  - 对称性
  - HBox监督
---

## 一句话总结

该方法把训练图像 \(I\) 进行垂直翻转与随机旋转，生成 \(I_{\mathrm{flp}}\) 和 \(I_{\mathrm{rot}}\)，三种视图进入参数共享的 ResNet-50 与 FPN。自监督分支先在各目标的全部采样点上平均角度特征，再由 assigner 匹配跨视图目标、剔除旋转后失去对应关系的实例，随后通过 PSC 角度编码器输出 \(\theta\)、\(\theta_{\mathrm{flp}}\)、\(\theta_{\mathrm{rot}}\)，分别构造翻转一致性损失 \(L_{\mathrm{flp}}\) 与旋转一致性损失 \(L_{\mathrm{rot}}\)。弱监督分支不再预测角度，只用 HBox 标注学习分类、中心度、位置和尺寸，并以 CircumIoU 直接约束预测 RBox 与其外接标注框。

实验覆盖 DOTA-v1.0、DOTA-v1.5、DOTA-v2.0、HRSC 与 FAIR1M，以 FCOS、ResNet-50、FPN 为统一基础，并同时比较 H2RBox、BoxInst-RBox、BoxLevelSet-RBox、SAM-RBox，以及采用真实 RBox 标注的 Rotated FCOS。主要指标为 AP、AP50、AP75 和 FPS。仅使用 HBox 时，该方法在五个数据集上的 AP50 分别为 72.31、64.76、50.33、89.66、42.27；对应全监督 FCOS 为 72.44、64.53、51.77、88.99、41.25。DOTA-v1.0 上推理速度为 29.1 FPS，与 H2RBox 相同，明显快于 SAM-RBox 的 1.7 FPS。

关键消融显示角度周期处理不是可选细节：在 DOTA 上，同时启用 PSC 与 snap loss 时达到 AP/AP50/AP75＝40.69/72.31/39.49；两者均不用时仅为 24.24/52.24/19.48，而只用 PSC 时几乎无法收敛，AP50 仅 0.77。HRSC 上完整组合达到 58.03/89.66/64.80；只用 snap loss 为 48.95/88.52/50.03，只用 PSC 则 AP50 仅 0.88，说明自监督角度学习中的边界不连续会直接破坏训练稳定性。

## 研究背景与问题

研究问题是：只有易获得的水平框标注时，能否直接训练可靠的旋转框检测器。此前 H2RBox 主要从外接 HBox 的几何约束中推断角度，因此依赖同类目标覆盖大量方向、标注框接近精确外接矩形；它在小规模 HRSC 上只能取得 7.03 AP50，也不能自然结合随机旋转增强，并对旋转产生的黑边敏感。

本文转而利用飞机、舰船、车辆、球场等目标的近似反射对称性。若角度网络同时满足“垂直翻转后输出取反”和“图像旋转 \(R\) 后输出增加 \(R\)”两种等变约束，那么对称目标的预测方向会收敛到其对称轴或垂直于该轴的方向。该结论把角度监督从 HBox 几何关系转移到图像自身结构，因而对标注噪声和训练样本不足更稳健。

## 方法总览

自监督分支的基础关系为：

\[
f(I)+f(\mathrm{flp}(I))=k\pi,\qquad
f(\mathrm{rot}(I,R))-f(I)=R+k\pi .
\]

训练时随机旋转范围采用 \(\pi/4\sim3\pi/4\)，边界区域使用 reflection padding。旋转角过于接近零会使一致性学习进入退化状态；HRSC 消融中，该范围取得 AP75＝64.99，优于更窄范围的 64.03 和 61.28。

为处理角度周期，snap loss 在所有相隔 \(\pi\) 的候选目标中选择与预测最接近者：

\[
\ell_s(\theta_p,\theta_t)=
\min_{k\in\mathbb Z}\operatorname{SmoothL1}(\theta_p,k\pi+\theta_t).
\]

由此 \(L_{\mathrm{flp}}=\ell_s(\theta_{\mathrm{flp}}+\theta,0)\)，\(L_{\mathrm{rot}}=\ell_s(\theta_{\mathrm{rot}}-\theta,R)\)，且 \(L_{\mathrm{ss}}=\lambda L_{\mathrm{flp}}+L_{\mathrm{rot}}\)。默认 \(\lambda=0.05\)：DOTA 上 AP50＝72.31，HRSC 上 AP50＝89.66；若完全去掉翻转项，即 \(\lambda=0\)，两者分别降至 66.37 和 0.32。

## 方法详解

CircumIoU 是弱监督分支支持随机旋转的核心。旧方法先把预测 RBox 转成 HBox，再与标注 HBox 计算 IoU；随机旋转后标注也成为 RBox，这一路径不再成立。本文把预测框投影到标注框方向，得到 \(B_{\mathrm{proj}}\)，再计算：

\[
\ell_p(B_{\mathrm{pred}},B_{\mathrm{gt}})
=-\ln\frac{|B_{\mathrm{proj}}\cap B_{\mathrm{gt}}|}
{|B_{\mathrm{proj}}\cup B_{\mathrm{gt}}|}.
\]

总损失为 \(L_{\mathrm{total}}=L_{\mathrm{cls}}+L_{\mathrm{cn}}+L_{\mathrm{box}}+L_{\mathrm{ss}}\)。推理时只保留一次 backbone、PSC 角度头及分类、回归、中心度头，不需要生成多视图。

## 实验与证据

DOTA-v1.0 上，该方法相对同环境复现的 H2RBox 从 70.05 提升至 72.31 AP50；使用多尺度后从 75.35 提升至 77.97，同时加入随机旋转达到 78.25，超过同增强条件下全监督 FCOS 的 77.68。使用 Swin-B 后达到 80.61。

优势在困难条件下更明显：HRSC 从 H2RBox 的 7.03 跃升至 89.66；FAIR1M 从 35.94 提升至 42.27。DOTA 标注框宽高加入 30% 均匀噪声时，AP50 仍有 71.11，而 H2RBox 为 67.39。仅使用 10% DOTA 训练数据时，两者 AP50 分别为 44.61 与 37.71，证明对称性提供了独立于标注几何的有效角度信号。

## 对 YOLO-Agent 的启发

复现 Harness 应固定 FCOS、ResNet-50、FPN、AdamW、初始学习率 \(5\times10^{-5}\)、batch size 2、相同训练日程与数据切片。控制组至少包含：①全监督 Rotated FCOS；②原始 H2RBox；③去掉 PSC；④以 Smooth L1 替换 snap loss；⑤以普通 IoU 替换 CircumIoU；⑥关闭随机旋转；⑦零填充替换反射填充；⑧DOTA 仅取 30% 和 10% 样本；⑨HBox 加入 10% 和 30% 尺寸噪声。

必须同时记录 AP、AP50、AP75、FPS、显存与是否收敛，并分别报告 DOTA-v1.0 和 HRSC，避免只用大数据集掩盖退化。具体失败判据：完整配置若未达到 DOTA AP50 72.31±0.5 或 HRSC AP50 89.66±0.5，应判复现失败；若关闭 PSC 或 snap loss 后仍与完整配置近似，则说明角度周期处理、视图匹配或损失接线实现错误；若 30% 标注噪声下 AP50 下降超过论文的 1.20 点，也不能支持其鲁棒性结论。

## 优点

局限在于反射对称只是近似先验。桥梁、港口、复杂建筑等目标的可见结构未必存在稳定对称轴，遮挡、阴影和背景纹理也可能造成伪对称；网络输出还可能落在真实对称轴的垂直方向。论文通过 HBox 的位置尺寸约束和数据统计规律解决方向选择，但没有从理论上消除语义朝向歧义。

## 局限

最有价值的思想不是增加一个角度头，而是把“方向”拆成图像内生的自监督变量，把位置、尺度、类别留给弱标注。这样既绕开昂贵的 RBox，又降低对外接水平框精度和方向覆盖率的依赖。PSC、snap loss、CircumIoU、旋转范围与反射填充共同构成稳定训练条件，缺少其中任一项都可能使理论一致性无法转化为检测性能。

## 评分

- 论文：[NeurIPS 2023 官方页面](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html)
- 官方代码：[OpenMMLab MMRotate](https://github.com/open-mmlab/mmrotate)
