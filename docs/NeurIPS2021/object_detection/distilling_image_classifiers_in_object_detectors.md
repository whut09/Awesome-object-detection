---
title: "[论文解读] Distilling Image Classifiers in Object Detectors"
description: "解析分类器到检测器的跨任务蒸馏：KD_cls 提升识别，KD_loc 通过可微裁剪提升定位。"
tags: ["NeurIPS 2021", "目标检测", "知识蒸馏", "跨任务迁移"]
---

# Distilling Image Classifiers in Object Detectors

**论文**：[官方论文页面](https://proceedings.neurips.cc/paper_files/paper/2021/hash/082a8bbf2c357c09f26675f9cf5bcba3-Abstract.html)  
**代码**：论文条目未提供官方代码链接  
**发表**：NeurIPS 2021  
**主题**：Classifier-to-Detector Distillation、KD_cls、KD_loc

## 一句话总结

该论文把检测数据中的真实框裁成分类数据训练教师，用 KD_cls 将分类器的软类别分布传给检测器正样本，再用 Spatial Transformer 把学生预测框和真实框裁成图像区域，比较二者经过同一分类教师后的中间特征，以跨任务方式同时改进识别与框定位。

## 研究背景与问题

此前检测蒸馏几乎都要求教师和学生解决同一目标检测任务，教师检测器的训练、结构匹配和部署前准备成本较高。论文提出更一般的问题：一个只做图像分类的教师，是否能把类别知识以及对对象形状、边界的表征传给一阶段或两阶段检测器。

为保证任务间语义对齐，作者从检测数据集 `D_det` 的真实框裁出对象，构造 `C` 类分类集 `D_cls`，并训练分类教师 `F_t`。检测学生 `F_s` 仍在原图和原检测标注上训练；教师只在学生的正 anchor/proposal 及其对应对象区域上提供监督。

方法明确拆成 **KD_cls** 与 **KD_loc**。前者解决类别识别，适配 Softmax 检测器与 Sigmoid/Focal Loss 检测器；后者不蒸馏教师框，而是比较学生预测裁剪与真实框裁剪在分类教师中的特征差异，直接把梯度回传到学生框坐标。

## 方法总览

设一个 batch 中有 `K` 个已匹配真实框的正 anchor 或 proposal。KD_cls 对每个正样本构造温度 `T` 下的教师、学生类别分布 `p_k^{t,T}` 与 `p_k^{s,T}`，最小化平均 KL 散度。KD_loc 对每个学生预测框生成可微裁剪 `O_k^p`，与对应真实区域 `O_k^{gt}` 同时输入分类教师，并比较指定层的池化特征。

整体损失为

\[
L=L_{det}+\lambda_{kc}L_{kd-cls}+\lambda_{kl}L_{kd-loc},
\]

其中 `L_det` 是学生原有分类与回归损失，`λ_kc、λ_kl` 控制两类跨任务监督。

## 方法详解

分类蒸馏为

\[
L_{kd-cls}=\frac1K\sum_{k=1}^{K}KL(p_k^{t,T}\|p_k^{s,T}).
\]

若学生用 Softmax，其输出含 `C` 个前景类和 1 个背景类；分类教师只有 `C` 类，因此教师分布末尾补 0 作为背景概率，再计算带 `T²` 缩放的 KL。若学生使用 RetinaNet 式 Sigmoid，单个 `C` 维输出并非和为 1 的分布；论文把每个类别概率改写成 `[1-p,p]` 的 False/True 二类分布，再对全部 `C` 类求平均 KL。

定位蒸馏对学生预测框 `B_k=(x1,y1,x2,y2)` 构造仿射矩阵 `A_k`，矩阵中的缩放项为框宽高相对原图 `w、h` 的比例，平移项由框中心坐标归一化得到。Spatial Transformer `f_ST(A_k,I,s)` 以网格尺寸 `s` 从原图 `I` 可微采样预测区域 `O_k^p`。

对分类教师第 `l` 个中间层 `F_t^l`，先用 Adaptive Pooling 把预测区域和真实区域的特征统一到 `M×W×H`，再计算

\[
L_{kd-loc}=\frac{1}{KLMWH}\sum_{k,l}\mathbf{1}_l
\|AP(F_t^l(O_k^p))-AP(F_t^l(O_k^{gt}))\|_1.
\]

`L` 是候选蒸馏层数，`1_l` 指示是否使用该层。由于裁剪与池化可微，误差能更新学生回归框。特殊项 `KD_loc^0` 直接比较池化后的图像像素，不依赖教师；它相当于比坐标回归更宽松的区域外观一致性约束。

## 实验与证据

实验在 COCO2017 上完成，训练集超过 118k 图像、验证集 5k 图像、80 类。紧凑学生包括 SSD300/512-VGG16、Faster R-CNN-Quartered ResNet50 和 Faster R-CNN-MobileNetV2；主要分类教师为输入 112×112 的 ResNet50。

SSD300 基线 25.6 mAP，KD_cls 为 26.3，KD_loc 为 27.2，联合达到 27.9；SSD512 从 29.4 提到 32.1。Faster R-CNN-QR50 从 23.3 经 KD_cls 提到 25.9，联合 KD_cls+KD_loc 达 27.2，增益 3.9。KD_cls 对 AP50 改善更明显，KD_loc 对 AP75 更有效：SSD300 的 KD_loc 将 AP75 从 26.3 提到 28.5，而联合后为 29.2。

与检测器到检测器蒸馏比较，Faster R-CNN-QR50 上本方法 27.2，FKD 为 26.1，联合两者为 28.0；SSD512 上本方法 32.1，FKD 31.2，联合为 32.6。对较大 R50 学生，单独收益缩小，但与 FKD 组合后 Faster R-CNN-R50 为 41.9、RetinaNet-R50 为 40.7。

消融中，SSD300 的 KD_cls 最优为 `λ_kc=0.4、T=2`，达到 26.3 mAP。ResNet50 教师优于 ResNet18，但 Top-1 更高的 ResNeXt101 反而降到 25.3，说明过强、过自信教师未必更适合。KD_loc 的采样尺寸在 112 后饱和；池化 8×8 得到 27.1 mAP；蒸馏对象像素与第一层特征联合为 27.2，继续加入更高层反而下降。

## 对 YOLO-Agent 的启发

可把 YOLO-Agent 的分类器资产复用于检测：按训练增强后的真实框裁出对象训练一个 `C` 类教师；检测训练时仅对 matcher 选出的正样本执行 KD_cls，并用预测框坐标调用可微 ROI sampler 计算 KD_loc。对照组应包含原 YOLO、仅标签平滑、KD_cls、像素级 KD_loc^0、教师特征 KD_loc、两者联合，以及现有 detector-teacher KD。

指标应分别观察 AP50、AP75 和分类/定位错误。论文参照是 SSD300 25.6→26.3（KD_cls）、25.6→27.2（KD_loc）、25.6→27.9（联合）。若 KD_cls 的 AP50 增益小于 0.3，或 KD_loc 的 AP75 增益小于 0.5，则对应分支失败；若预测框裁剪越界比例超过 5% 或 `KD_loc` 梯度范数超过原回归损失 2 倍，应裁剪仿射网格并降低 `λ_kl`。若更强教师使验证 AP 下降超过 0.3，应以教师校准误差而非分类 Top-1 选型。

## 优点

- 首次系统展示分类器到检测器的跨任务蒸馏，并同时覆盖识别与定位。
- KD_cls 兼容 Softmax 和 Sigmoid 检测头；KD_loc 对一阶段、两阶段均可用。
- 同一分类教师可服务多个检测学生，并能与 detector-to-detector KD 叠加。

## 局限

- 需要从检测标注裁剪并训练分类教师，数据增强协议不同时还需单独教师。
- KD_loc 对正样本逐框裁剪和教师前向，训练计算量随正样本数增加。
- 大学生网络的单独收益较小，过强教师还可能因过度自信降低效果。

## 评分

**8.8 / 10**：跨任务设定新颖，KD_cls/KD_loc 的功能与 AP50/AP75 证据对应清楚；工程成本主要来自对象裁剪教师和逐框可微特征提取。
