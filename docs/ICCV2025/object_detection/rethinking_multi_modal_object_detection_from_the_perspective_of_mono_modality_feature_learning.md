---
title: "[论文解读] Rethinking Multi-modal Object Detection from the Perspective of Mono-Modality Feature Learning"
description: "解析 M²D-LIF 如何用单模态教师蒸馏和局部光照感知权重缓解 RGB-IR 融合退化。"
tags: ["ICCV 2025", "多模态目标检测", "RGB-IR", "知识蒸馏"]
---

# Rethinking Multi-modal Object Detection from the Perspective of Mono-Modality Feature Learning

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Zhao_Rethinking_Multi-modal_Object_Detection_from_the_Perspective_of_Mono-Modality_Feature_ICCV_2025_paper.html)  
**代码**：论文条目未提供官方代码链接  
**发表**：ICCV 2025  
**主题**：Mono-Modality Distillation、Local Illumination-aware Fusion、RGB-IR 检测

## 一句话总结

M²D-LIF 先用同模态教师和跨模态显著位置掩码补强联合训练中被削弱的 RGB/IR 编码器，再由 Local Illumination-aware Fusion 预测局部亮度图，为不同区域生成互补的 RGB 与红外权重，从“先学好每个模态、再做轻量融合”的路径缓解融合退化。

## 研究背景与问题

论文首先指出 fusion degradation：一些目标能被单模态检测器发现，却被 RGB-IR 联合模型漏掉。作者在 FLIR 上对 Halfway Fusion、IWM 和 CMX 做线性探测：冻结单模态或多模态联合训练后的 CSPDarknet53 编码器，再接新检测头训练。联合训练得到的每个模态骨干均弱于对应单模态训练骨干，说明融合模块会限制单模态表征学习。

这一观察改变了常见的优化重点。以往多模态检测主要继续堆叠 cross-attention 或复杂交互模块，但若输入融合模块的 RGB、IR 特征已经不足，再强的融合也无法恢复丢失目标。线性探测还显示加权式 IWM 在较弱骨干上仍有竞争力，促使作者选择轻量、显式的加权融合而非更重的注意力结构。

完整框架名为 **M²D-LIF**，由 Mono-Modality Distillation（M²D）与 Local Illumination-aware Fusion（LIF）组成。训练期额外使用 RGB、IR 单模态教师；推理期移除教师，只保留双分支学生骨干、LIF 与检测头。

## 方法总览

对配对 RGB 图像 `I_V` 与红外图像 `I_I`，学生骨干 `B_V、B_I` 输出 `f_V、f_I`，同模态教师 `\tilde B_V、\tilde B_I` 输出 `\tilde f_V、\tilde f_I`。M²D 包含同模态蒸馏 `L_IM` 和跨模态蒸馏 `L_CM`；LIF 从 RGB 图像预测亮度图 `B`，生成区域级权重 `W_V、W_I`，并融合每层双模态特征。

训练总目标为

\[
L=L_{det}+\lambda_{M^2D}L_{M^2D}+\lambda_{LI}L_{LI},
\]

`L_det` 是检测损失，`L_M²D` 强化两条学生编码器，`L_LI` 监督亮度预测。推理只执行学生双骨干和 LIF。

## 方法详解

同模态蒸馏直接让每个学生分支学习对应教师：

\[
L_{IM}=D(f_V,\tilde f_V)+D(f_I,\tilde f_I).
\]

`D` 表示具体特征蒸馏方法，框架图和正文采用 PKD。为避免只匹配整体响应，作者用无参数 SimAM 从教师特征提取注意图 `f_M`。其中教师特征的空间均值、方差分别为 `\tilde\mu、\tilde\sigma^2`，`λ` 是数值稳定常数；注意图经 Sigmoid 得到目标显著位置先验。

跨模态损失为

\[
L_{CM}=D(f_{MV}\odot f_I,f_{MV}\odot\tilde f_V)+
D(f_{MI}\odot f_V,f_{MI}\odot\tilde f_I).
\]

`f_MV、f_MI` 分别来自 RGB 与 IR 教师。第一项用 RGB 显著区域掩码约束 IR 学生与 RGB 教师，第二项反向进行，使一个模态发现的目标位置能够指导另一模态提取对象相关特征。最终 `L_M²D=L_IM+L_CM`。

LIF 先用卷积块从 RGB 图像预测亮度图：`B=ConvBlock(I_V)`。作者把 RGB 转到 LAB 色彩空间，以 L 通道 `\tilde B` 作为监督：

\[
L_{LI}=\|B-\tilde B\|_2.
\]

随后由亮度生成互补权重：

\[
W_V=\beta\min\left(\frac{B-\alpha}{2\alpha},\frac12\right)+\frac12,
\qquad W_I=1-W_V.
\]

`α` 是决定 RGB 重要性的亮度阈值，`β` 控制权重振幅，二者保证权重位于以 0.5 为中心的有限区间。第 `i` 层融合为 `f_F^i=W_V^i⊙f_V^i+W_I^i⊙f_I^i`：明亮区域提高 RGB 比重，低照区域提高 IR 比重，而且权重按空间位置变化，而非整图一个标量。

## 实验与证据

实验覆盖三个数据集。DroneVehicle 有 28,439 对 RGB-IR 图像，训练/验证/测试为 17,990/1,469/8,980，并含 car、truck、bus、van、freight car 五类 OBB；FLIR-aligned 有 5,142 对，训练/测试 4,129/1,013，评估 person、car、bicycle；LLVIP 有 15,488 对低照图像，训练/测试 12,025/3,463。骨干为双分支 CSPDarknet53。

组件消融表中，无 M²D/LIF 的基线在 FLIR、DroneVehicle、LLVIP 为 43.5、66.9、68.2 mAP；仅 M²D 为 45.1、69.2、70.7，仅 LIF 为 45.0、68.5、69.4，联合为 46.1、70.6、70.8。完整模型仅把参数从 36.47M 增至 36.53M，FLOPs 从 116.7G 增至 123.2G。

LIF 的权重生成也有针对性消融：直接用 L 通道作权重在三数据集为 44.2、67.4、70.2；不使用 L 通道监督为 45.0、69.8、70.3；加入 L 通道监督达到 46.1、70.6、70.8。`β=0.4` 最优，超过 0.8 会造成模态失衡并显著降性能。

SOTA 比较中，M²D-LIF 在 DroneVehicle 测试集取得 81.4 mAP50、68.1 mAP，验证集 mAP50 为 84.5；FLIR 与 LLVIP 分别为 46.1、70.8 mAP，参数 36.5M。对比方法包括 Halfway Fusion、GAFF、CFT、CSAA、ICAFusion、RSDet、UniRGB-IR，以及 DroneVehicle 上的 TarDAL、UA-CMDet、GLFNet、TSFADet、CALNet、C2Former、CAGTDet、OAFA。

## 对 YOLO-Agent 的启发

接入点应分为训练与推理两处：训练器加载两个冻结的 RGB/IR 单模态 YOLO 编码器，向双模态学生 neck 各尺度输出提供 `L_IM+L_CM`；模型侧在双分支 neck 融合处加入 LIF 权重图。对照组为简单相加、仅 M²D、仅 LIF、M²D-LIF，并额外加入“重型 cross-attention + M²D”以验证单模态学习是否比融合复杂度更关键。

指标应按 FLIR/LLVIP 的昼夜或亮度区间报告 mAP，并记录单独冻结 RGB、IR 学生编码器后的线性探测 AP、融合权重均值及参数/FLOPs。论文参照是三数据集完整方案相对表格基线提升 2.6、3.7、2.6 mAP。若完整方案增益低于 1.0 mAP，或任一学生分支的线性探测 AP 比其单模态教师低超过 2.0，应判定蒸馏未解决根因；若 `W_V` 在 95% 像素上长期小于 0.1 或大于 0.9，或验证性能在 `β>0.8` 区域出现明显下降，则视为模态失衡并回退到 `β≤0.4`。

## 优点

- 通过线性探测把多模态漏检追溯到单模态编码器学习不足，而非仅凭融合模块猜测。
- M²D 同时包含同模态能力保持与跨模态目标位置互助，教师在推理时可删除。
- LIF 用局部亮度生成互补权重，参数增量极小，并在三类 RGB-IR 数据集上验证。

## 局限

- 训练前需要分别获得 RGB 与 IR 单模态教师，训练资源和数据管理成本增加。
- LIF 依赖 RGB 的 LAB L 通道监督，遇到曝光异常、彩色噪声或非 RGB 主模态时需重新设计质量信号。
- 论文聚焦已对齐 RGB-IR；跨传感器错位、时间不同步和缺失模态不在方法覆盖范围内。

## 评分

**9.1 / 10**：论文从诊断实验到 M²D 与 LIF 的结构闭环完整，且结果覆盖 OBB、常规检测与低照场景；其限制主要来自单模态教师依赖和对齐双模态假设。
