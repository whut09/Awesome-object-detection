---
title: "[论文解读] SAFE: Sensitivity-Aware Features for Out-of-Distribution Object Detection"
description: "解析 SAFE 如何从残差卷积与 BatchNorm 关键层提取目标级特征并进行后验 OOD 检测。"
tags: ["ICCV 2023", "目标检测", "OOD 检测", "可靠性"]
---

# SAFE: Sensitivity-Aware Features for Out-of-Distribution Object Detection

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/ICCV2023/html/Wilson_SAFE_Sensitivity-Aware_Features_for_Out-of-Distribution_Object_Detection_ICCV_2023_paper.html)  
**代码**：官方未提供  
**会议**：ICCV 2023

## 一句话总结

SAFE 冻结已有 Faster R-CNN，只监控骨干中“残差连接对应卷积且其后紧接 BatchNorm”的四个敏感层，通过预测框对这些层做 ROIAlign、拼接目标级向量，再训练一个三层 MLP 区分干净 ID 特征与 FGSM 扰动产生的代理 OOD 特征。

## 研究背景与问题

目标检测的 OOD 判断不同于整图分类：一张图中可以同时存在已知与未知物体，因此需要对每个检测框获得 OOD 分数。已有 feature-based 方法通常选最后层或大量层融合，却没有解释哪些层稳定携带域外信号；直接堆叠所有层还会把弱层噪声送入辅助分类器，并使 MLP 权重矩阵随输入维度近似二次增长。

SAFE 的层选择来自两个可验证机制。残差连接有助于保留输入变化到隐藏空间的敏感性；BatchNorm 使用 ID 数据统计量 $E_{in},V_{in}$，当 OOD 数据真实统计 $E_{out},V_{out}$ 不匹配时会产生异常激活。论文假设二者结合的层比单独残差或单独 BatchNorm 更适合 OOD 检测。

## 方法总览

对冻结检测器 $f$，图像产生 $D$ 个预测 $(c_d,b_d)$。从 $L$ 个 SAFE 层取得特征图 $M_1, …, M_L$；每个预测框 $b_d$ 在各层裁剪并池化为 $p_{l,d}$，随后按层拼接成 $q_d$。辅助 MLP $f_β$ 输出 $ŷ_d=f_β(q_d)∈[0,1]$ 作为该目标的 OOD 分数。训练 MLP 不使用真实 OOD 数据，而用同一 ID 图像的对抗扰动版本构造代理异常。

## 方法详解

### 1. 敏感层与理论动机

特征提取器满足双 Lipschitz 约束时，输入距离应被隐藏空间既保留又限制：

$$
L₁‖x-x′‖ᵢ ≤ ‖f(x)-f(x′)‖ꜰ ≤ L₂‖x-x′‖ᵢ.
$$

$L_1$ 表示 sensitivity，避免不同输入在特征空间坍缩；$L_2$ 表示 smoothness。SAFE 不重新训练骨干，不能保证谱归一化带来的上界，但残差网络仍能保留足够敏感性。BatchNorm 则执行

$$
BN(z; γ, β, ε) = ((z-E_in[z]) / sqrt(V_in[z]+ε))γ + β,
$$

当部署分布统计偏离 ID 时，异常归一化响应为 MLP 提供信号。ResNet-50 与 RegNetX4.0 中都识别出四个这类关键层。

### 2. 目标级 SAFE 向量

对层 $l$ 和检测 $d$，预测框在 $M_l$ 上通过 ROIAlign 得到目标区域 $O_{l,d}$，再沿空间维双线性池化为通道向量 $p_{l,d}$。拼接后

$$
q_d = [p_{1,d}; …; p_{L,d}],    |q_d| = Σ_l c_l,
$$

$c_l$ 是第 $l$ 层通道数。三层全连接 MLP 的隐藏维度逐层减半，末层前使用 dropout，Sigmoid 输出 OOD 概率。

### 3. 代理异常训练

对 ID 图像 $x$，FGSM 生成

$$
x_o = x + ε · sign(∇_x J(θ,x,y)),
$$

$J$ 是冻结检测器任务损失，$θ$ 是其参数，$ε$ 为扰动幅度。干净图预测框同时用于干净和扰动图的特征裁剪，避免框位置变化混入标签。干净 SAFE 向量标为 ID，扰动向量标为代理 OOD，仅训练 MLP；主检测器无需重训。

## 实验与证据

评测沿用 VOS 协议：ID 为 PASCAL-VOC 或 BDD100K，OOD 为移除 ID 类别后的 MS-COCO、OpenImages 子集。基模型是 ResNet-50 或 RegNetX4.0 Faster R-CNN，指标为 AUROC（高为优）和 FPR95（低为优），五个随机种子取均值和标准差。基线包括 MSP、ODIN、Mahalanobis、Energy、Gram、ViM、KNN、Generalized ODIN、CSI、GAN-Synthesis、VOS。

SAFE 在 8 组“ID 数据×OOD 数据×指标”组合中取得 7 组最佳。PASCAL-VOC→OpenImages 上，SAFE-ResNet50 为 **92.28 AUROC / 20.06 FPR95**，VOS-ResNet50 为 85.23/51.33；RegNetX4.0 上 SAFE 为 **94.38/17.69**。BDD100K→OpenImages 时 SAFE-RegNetX4.0 达 95.97/13.98，BDD100K→COCO 为 93.91/21.69。

层消融验证了选择原则。PASCAL-VOC 为 ID 时，四个 SAFE 层融合在 OpenImages 上为 92.28 AUROC、20.06 FPR95；12 个 residual layers 为 91.33/24.82，全部 60 个卷积层为 89.88/26.73。随机 1、4、8、16 层均明显更差。扰动幅度消融显示 ResNet-50 在 $ε∈[4,8]$ 对不同 ID/OOD 组合保持高性能；主比较取 $ε=8$，RegNetX4.0 取 1。

## 对 YOLO-Agent 的启发

接入点应在 YOLO backbone 每个 stage 的残差下采样/shortcut 汇合且紧随 BN 的卷积输出，而不是检测头最终特征。用 YOLO 已解码框在这些层做 ROIAlign，拼接后训练独立 OOD MLP；检测器参数冻结。对照组应为最后 neck 层、全部 backbone 层、仅 residual 层、SAFE 条件层，并比较 VOC/BDD 为 ID、COCO/OpenImages 为 OOD 的 AUROC 与 FPR95。论文最关键的是 FPR95，因此失败阈值可设为：相对最后层 FPR95 改善不足 5 个百分点，或相对全层融合没有更低 FPR95，或 ID 检测 AP 变化超过 0.1。若输入维度使 MLP 参数超过基线两倍，应先缩减每层通道投影，而不是纳入更多层。

## 优点

- 后验方法不改检测器训练目标，也不影响原始 AP。
- 层选择有残差敏感性与 BatchNorm 统计失配的理论动机，并被逐层消融支持。
- 四层融合优于全部 60 层，兼顾 OOD 质量和辅助网络规模。
- 同一原则在 ResNet-50 与 RegNetX4.0 上都成立。

## 局限

- 依赖残差卷积后接 BatchNorm 的结构，无 BN 或 Transformer 骨干不能直接套用。
- 代理 OOD 来自 FGSM，不保证覆盖真实未知物体的全部外观变化。
- 每个检测框都需多层 ROIAlign 与 MLP，部署延迟会随框数增长。

## 评分

- **创新性：8.5/10**——把层敏感性作为 OOD 检测的明确设计变量。
- **实验充分性：8.5/10**——覆盖两种 ID、两种 OOD、两种骨干及逐层消融。
- **可复现性：8/10**——实现细节充分，但缺少官方代码。
- **YOLO-Agent 适配度：8/10**——适合带残差和 BN 的 YOLO 骨干，需评估 ROIAlign 延迟。


