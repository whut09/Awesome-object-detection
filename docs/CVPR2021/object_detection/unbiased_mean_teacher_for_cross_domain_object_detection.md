---
title: "Unbiased Mean Teacher for Cross-Domain Object Detection"
description: "CVPR 2021 论文详解：用跨域蒸馏、target-like 训练与置信度分支修正 Mean Teacher 的源域偏置。"
tags: ["CVPR 2021", "目标检测", "无监督域适应", "Mean Teacher"]
---

# Unbiased Mean Teacher for Cross-Domain Object Detection

- **论文页面**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Deng_Unbiased_Mean_Teacher_for_Cross-Domain_Object_Detection_CVPR_2021_paper.html)
- **官方 PDF**：[Paper PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Unbiased_Mean_Teacher_for_Cross-Domain_Object_Detection_CVPR_2021_paper.pdf)
- **官方代码**：[kinredon/umt](https://github.com/kinredon/umt)（论文摘要明确给出）
- **发表会议**：CVPR 2021

## 一句话总结

UMT 不把 Mean Teacher 的伪标签误差简单归因于阈值，而是用 CycleGAN 构造 `source-like target → teacher` 与 `raw target → student` 的跨域蒸馏对，再用 `target-like source` 和 proposal 级分布内置信度共同消除教师、学生与教学过程的源域偏置。

## 研究背景与问题

任务设定是无监督跨域目标检测：源域图像有框和类别，目标域训练图像完全无标注，两个域共享类别空间。普通 Mean Teacher 让学生在源域真值上学习 Faster R-CNN，同时让 EMA 教师在目标图像上生成伪框；问题是教师本身由受源域监督的学生滑动平均而来，因此面对目标风格时，分类与定位都会偏向源域。

论文用一个直接实验确认偏置：先把目标图像通过 CycleGAN 翻译成 source-like 版本，再让同一教师分别检测原目标图与 source-like 图。三个迁移场景中，教师在 source-like 图上的 mAP 均明显更高，说明伪标签质量不足不是随机噪声，而是可测量的分布偏好。

## 方法总览

一个 mini-batch 同时含四路数据：有标注源图 `I_s`、其 target-like 图 `P_s`、无标注目标图 `I_t`、其 source-like 图 `P_t`。`I_s/P_s/I_t` 进入学生 Faster R-CNN，`P_t` 进入 EMA 教师；教师的 proposal 经 NMS、类别置信度与分布内置信度筛选后，作为 `I_t` 上的伪框监督学生，最终输出目标域的类别与回归结果。

其中三项修正各自对应一个偏置来源：Cross-domain Distillation 修正教师输入，Target-like Augmentation 修正学生训练分布，Confidence Branch 动态修正“哪些样本适合当前教师教学”。四项损失联合优化学生，教师参数仅由学生参数的 EMA（系数 0.99）更新。

## 方法详解

### 1. Simple Mean Teacher 基线

源图使用 Faster R-CNN 标准损失 `L_det=L_rpn+L_roi`。目标图分别做扰动后送入教师和学生，教师对每类 proposal 做 NMS，并保留最大类别概率高于 `T=0.8` 的框；这些框与类别被当作学生的伪真值，形成蒸馏损失。该基线记为 `UMT_S`。

### 2. Cross-domain Distillation

对每张目标图 `I_t` 离线生成 source-like 对应图 `P_t`。教师读取更符合自身偏好的 `P_t`，学生读取原始 `I_t`，于是伪框来自较可靠的源风格视图，但梯度仍迫使学生在真实目标风格上复现检测。该版本 `UMT_SC` 直接针对教师偏置，而不是让教师和学生处理同一种目标扰动。

### 3. Target-like Augmentation

源图 `I_s` 经 CycleGAN 变成 `P_s`，几何布局与框标注不变，因此 `P_s` 可复用源域真值计算 `L_rpn+L_roi`。学生同时从 `I_s` 学稳定监督、从 `P_s` 学目标外观，减少其源域倾向；加入此路后得到 `UMT_SCA`。

### 4. Confidence Branch 与总目标

检测头旁增加 proposal 级 Confidence Branch，输出样本属于当前模型分布的置信度 `σ_j`，并用 `-log(σ_j)` 约束有标注样本。训练分类头时，论文用 `σ_j p_j+(1-σ_j)y_j` 在模型预测与 one-hot 真值间插值；教学时则用 `σ_j × 最大类别概率` 与阈值比较，低分布内置信度的 source-like proposal 不进入蒸馏。总损失由源图检测、target-like 检测、跨域蒸馏和 confidence loss 组成，权重设置为 `λ=0.01、γ=0.1`。

## 实验与证据

- **Real→Artistic**：VOC 2007+2012 为源域，Clipart1k/Watercolor2k 为无标注目标域，ResNet-101 Faster R-CNN，以各类 AP 和 mAP 评价。Clipart1k 上 Source Only 为 27.7，`UMT_S/UMT_SC/UMT_SCA/UMT` 依次为 33.6/37.6/41.8/44.1；此前最强 DM 为 41.8。Watercolor2k 上 UMT 为 58.1，高于 SCL 55.2、SWDA 53.3。
- **Cityscapes→Foggy Cityscapes**：VGG16 Faster R-CNN，在 8 类 Foggy Cityscapes 验证集上 UMT 达 41.7 mAP，高于 HTCN 39.8；`UMT_S/UMT_SC/UMT_SCA` 为 28.5/39.2/40.4，逐步开组件提供了清晰消融证据。
- **SIM10K→Cityscapes**：只评 car AP，Source Only 34.3，HTCN 42.5，`UMT_S/UMT_SC/UMT_SCA/UMT` 为 40.8/42.0/42.6/43.1。
- **训练协议**：短边缩放至 600；前 50k iteration 学习率 0.001，后 30k 为 0.0001；每批四类图像各一张。主结果覆盖艺术风格、逆天气和合成到真实三种不同域偏移。

这些递增结果还揭示了三项机制的分工：仅靠 Mean Teacher 主要提供扰动一致性；跨域蒸馏首先提升教师伪框可靠性；target-like 源图继续改变学生所见外观；最后的 Confidence Branch 再按当前模型分布动态选择 proposal。若复现时四级曲线不呈这一顺序，应优先检查图像翻译配对、EMA 更新和置信度乘积筛选，而不是直接归因于随机种子。

## 对 YOLO-Agent 的启发

UMT 最值得迁移的不是 Faster R-CNN 外壳，而是“先测伪标签生产者偏向哪里，再改变 teacher/student 两端输入”的诊断方式。YOLO-Agent 可把 EMA teacher 的伪标签按域风格、天气或传感器条件分桶，比较原图与风格对齐图上的 precision、recall 和 box IoU，再决定是否启用跨视图伪标签，而非只调 confidence threshold。

### 专属 Harness：跨风格伪标签偏置测试

- **对照组**：A 为标准 YOLO EMA Teacher（teacher/student 均读目标图弱/强增强）；B 为 UMT 式配对（teacher 读 CycleGAN source-like 目标图，student 读原目标图）；C 在 B 上增加 target-like 源图监督；D 再增加 proposal 分布内置信度门控。
- **观测指标**：目标验证集 mAP50-95、伪标签 precision/recall、teacher 在原目标图与 source-like 图上的 mAP 差、伪框与真值的平均 IoU、每张图保留伪框数。
- **通过标准**：至少 3 个种子中，B 相对 A 的伪标签 precision 与 mAP50-95 同向上升；C、D 的增益可累加，且 D 将两种风格输入间的 teacher mAP 差缩小至少 20%，伪标签 recall 不下降超过 1 个百分点。
- **失败判断**：只有 source-like 图离线评估变好而目标学生不涨；收益来自伪框数量骤减；C/D 关闭后仍保留同等增益；或跨域翻译改变物体几何导致框错位，均说明 UMT 数据流未被可靠迁移。

## 优点

- 用 source-like/target-like 成对数据把“教师偏置、学生偏置、教学偏置”拆成可独立验证的三个问题。
- 组件递增结果在多个域偏移上方向一致，关键收益并非只出现在单一数据集。
- 训练结束后仍是普通检测器，CycleGAN 与教师分支不增加部署推理成本。

## 局限

- 依赖离线图像翻译；若 CycleGAN 改变边界、纹理语义或小目标结构，伪框监督会被系统性污染。
- Confidence Branch 学到的是相对训练分布的适配度，不等同于严格校准的检测置信度。
- 该分支若未经目标域可靠性校准，极端天气中的困难真阳性也可能被误删。
- “Oracle” 在部分数据集低于 UMT，说明不同训练配方与跨域辅助数据会影响其作为上界的解释性。

## 评分

- **创新性：8.5/10**：以可观测的 Mean Teacher 源域偏置为切入点，三项修正互相对应。
- **实验充分性：9/10**：三类域偏移、多个强基线和连续组件消融形成完整证据链。
- **工程可迁移性：7.5/10**：teacher/student 逻辑易迁移，但图像翻译质量和额外训练成本不可忽略。
- **综合评分：8.5/10**。
