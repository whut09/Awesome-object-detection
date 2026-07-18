---

# Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence
title: "Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence"
description: "将旋转框映射为二维高斯分布，以 Kullback-Leibler Divergence 构造参数耦合、尺度不变的回归损失，提升大长宽比目标与高 IoU 阈值下的旋转检测精度。"
tags:
  - 旋转目标检测
  - KLD
  - 高斯分布
  - 边界框回归
  - 航空影像
  - NeurIPS-2021
---

- **论文链接**：https://proceedings.neurips.cc/paper_files/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html
- **官方代码**：https://github.com/yangxue0827/RotationDetection

## 一句话总结

该方法不直接对旋转框参数 \((x,y,w,h,\theta)\) 分别施加 Smooth L1，而是执行明确的数据流：检测器预测偏移量 \((t_x^p,t_y^p,t_w^p,t_h^p,t_\theta^p)\) → 解码为旋转框 → 将预测框与真实框分别转换为二维高斯分布 \(\mathcal N_p(\mu_p,\Sigma_p)\) 和 \(\mathcal N_t(\mu_t,\Sigma_t)\) → 计算两分布的 Kullback-Leibler Divergence → 经 \(L_{\text{reg}}=1-\frac{1}{\tau+f(D)}\) 归一化后参与多任务训练。转换中，框中心构成均值 \(\mu=(x,y)^\top\)，宽、高及旋转矩阵共同构成协方差 \(\Sigma\)，因此位置、尺度与角度进入同一概率空间。

KLD 的关键不是简单更换距离函数，而是形成 self-modulated optimization。\(D_{\mathrm{KL}}(\mathcal N_t\Vert\mathcal N_p)\) 具有全部参数的链式耦合；论文主要采用的 \(D_{\mathrm{KL}}(\mathcal N_p\Vert\mathcal N_t)\) 虽为半耦合，但其中心梯度由真实框的 \(1/w_t^2、1/h_t^2\) 及 \(\theta_t\) 调制，宽高梯度受角度误差影响，角度梯度又受长宽比影响。目标越细长，模型对角度误差的惩罚越强；目标或某条边越小，对相应方向中心偏移的惩罚越重。这正对应船舶、长文本和密集小目标在高 IoU 阈值下的定位需求。

实验覆盖 DOTA-v1.0/v1.5/v2.0、UCAS-AOD、HRSC2016、ICDAR2015、MLT、MSRA-TD500，以 RetinaNet 和 R3Det 为主要检测器，并与 Smooth L1、Gaussian Wasserstein Distance（GWD）及多种旋转损失比较。HRSC2016 上，RetinaNet 的 AP75 从 Smooth L1 的 48.42 提升到 KLD 的 72.39，增幅 23.97；R3Det 的 AP75 从 43.42 提升到 77.38，增幅 33.96。损失形式消融中，直接使用原始 \(D_{\mathrm{KL}}\) 仅得 0.20，而 \(f(D)=\log(D+1)、\tau=1\) 达到 85.25，说明非线性压缩是稳定训练所必需的。

## 研究背景与问题

传统旋转检测通常继承水平检测的归纳式设计：在水平框回归参数后增加角度分量，并让五个参数独立优化。问题是评价指标使用旋转 IoU，而训练损失没有表达参数之间对 IoU 的联合影响。对大长宽比目标，即使中心和宽高接近正确，微小角度误差也会造成 IoU 急剧下降；对小目标，少量中心偏移同样可能决定正负样本质量。

论文反向采用“从一般旋转框推导特殊水平框”的演绎思路。旋转框先被视为二维高斯，水平框只是 \(\theta=0\) 的特例；相应 KLD 在水平情形可分解为中心的二范数项、宽高的一范数项及其倒数相关项，与常见 \(l_n\)-norm 回归具有一致结构，而不是临时拼接一个角度损失。

## 方法总览

KLD 包含均值差的马氏距离、协方差间的迹项以及行列式比值。均值项使中心误差沿真实框主轴分解，并按真实宽高归一；协方差项把 \(w、h、\theta\) 联合起来。相比之下，GWD 的中心距离仍是独立欧氏距离，只耦合宽、高、角度，因此论文称其为 semi-coupled loss，预测框仍可能发生轻微整体平移。

另一个重要性质是仿射不变性：对任意满秩矩阵 \(M\)，同时将预测与真实高斯变换为 \((M\mu,M\Sigma M^\top)\)，KLD 保持不变；当 \(M=kI\) 时即得到尺度不变性。Smooth L1 与 GWD 不具备这一性质，所以同样的相对几何误差不会因图像或目标整体放缩而改变损失尺度。

## 方法详解

最终设置采用 \(D_{\mathrm{KL}}(\mathcal N_p\Vert\mathcal N_t)\)、\(f(D)=\log(D+1)\)、\(\tau=1\)，回归与分类权重为 \(\lambda_1=2、\lambda_2=1\)，分类项使用 Focal Loss。网络默认以 ResNet-50 初始化，Momentum 为 0.9，权重衰减为 0.0001；8 张 V100 上总批量为 8，训练 20 个 epoch，初始学习率 \(5\times10^{-4}\)，在第 12、16 个 epoch 各衰减十倍。推理阶段仍只输出普通旋转框，高斯转换与 KLD 仅用于训练，因此不增加推理时间。

## 实验与证据

在 RetinaNet 上，KLD 相对 Smooth L1 在 MLT、UCAS-AOD、DOTA-v1.0、DOTA-v1.5、DOTA-v2.0 分别提升 9.17、1.58、5.55、3.63、3.53 个点。其中 DOTA-v1.5 与 v2.0 含大量不足 10 像素的小目标，结果支持按真实框尺度调制中心梯度的设计。MSRA-TD500 上 Hmean75 从 36.73 提升到 46.95；ICDAR2015 的 RetinaNet-F 设置中 AP75 从 36.97 提升到 44.46。

KLD 的优势主要集中在严格定位指标，而非只改善 AP50。HRSC2016-R3Det 的 AP50 仅增加 1.45，但 AP75、AP85、AP50:95 分别增加 33.96、20.54、15.22，准确体现其“高精度框”目标。

## 对 YOLO-Agent 的启发

复现实验应建立三组严格控制：同一旋转检测头分别使用 Smooth L1、GWD、KLD；固定 backbone、锚框定义、正负样本规则、训练轮数、学习率、增强与随机种子。主控制组使用 RetinaNet，能力上界验证使用 R3Det；不得把 KLD 与额外 neck、标签分配器或训练时长同时变更。

HRSC2016 固定 R+F+G 增强及 \(500\times500\) 输入，报告 AP50、AP60、AP75、AP85、AP50:95；MSRA-TD500 与 ICDAR2015 同时报 Hmean50/60/75/85/50:95。必须额外按长宽比分桶，并统计角度误差、中心偏移和旋转 IoU，以验证“长宽比调制角度梯度”而不只比较总 mAP。

具体失败判据：若 KLD 在 HRSC2016-RetinaNet 上不能同时超过 Smooth L1 的 AP75=48.42 与 AP50:95=47.76，或其 AP75 提升不足 15 个点，则核心高精度结论复现失败；若 AP50 上升但 AP75/AP85 不升，说明收益可能来自一般训练扰动而非参数耦合。还应复现原始 KLD、sqrt、log 三组；若未经非线性变换仍稳定获得接近 85.25 的结果，需要检查实现是否暗含裁剪或归一化。

## 优点

优势在于把几何关系直接写入损失，无需新增推理模块；同时解决参数独立优化、尺度敏感以及训练损失与旋转 IoU趋势不一致的问题。它可嵌入不同旋转检测器，并对细长目标、小目标和严格 IoU 阈值尤为有效。

## 局限

原始 KLD 对大误差极敏感，直接训练仅得 0.20，必须依赖 log 或 sqrt 压缩。收益也并非每项指标都恒定领先：ICDAR2015 的 RetinaNet R+F 设置中，KLD 的 AP75=43.27、AP85=11.09，低于 GWD 的 45.59、11.65。KLD 的非对称方向及其 min、max、JS、Jeffreys 变体整体差距不大，没有形成普遍最优的对称化方案。

## 评分

这项工作的本质贡献是把旋转框回归从“五个独立数值的拟合”改写为“两个几何分布的匹配”。二维高斯不是附加模块，而是统一中心、尺度和方向的中间表示；KLD 则通过目标尺度、方向和长宽比动态重加权梯度。最有说服力的证据不是普通 AP 的小幅增长，而是 HRSC2016 上 AP75、AP85 与 AP50:95 的显著跃升，以及在 DOTA 小目标版本和长文本数据上的一致改进。
