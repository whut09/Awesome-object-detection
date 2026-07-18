---
title: "[论文解读] CrossKD: Cross-Head Knowledge Distillation for Object Detection"
description: "解析 CrossKD 如何通过学生中间特征与冻结教师后半检测头生成跨头预测。"
tags: ["CVPR 2024", "目标检测", "知识蒸馏", "CrossKD"]
---

# CrossKD: Cross-Head Knowledge Distillation for Object Detection

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)  
**代码**：官方条目未提供代码链接  
**发表**：CVPR 2024

## 一句话总结

CrossKD 不让学生最终预测直接追教师，而是把学生检测头第 \(i\) 层中间特征送入冻结的教师后半检测头，生成 cross-head predictions，再对这些预测做分类与回归蒸馏，从梯度路径上隔离真值监督和教师监督的目标冲突。

## 研究背景与问题

密集检测的标签由 assigner 动态决定，教师预测不可能逐位置复现学生训练真值；当师生使用不同 assigner 时冲突更严重。传统 prediction mimicking 在学生最终输出上同时施加真值损失与教师蒸馏损失，一个位置可能被真值要求为背景、却被教师赋予高前景分数，学生头接收方向相反的梯度。论文在 COCO minival 统计到，即使 ATSS/GFL 教师与 GFL 学生共享分配策略，也存在大量目标差异超过 0.5 的区域；换成 RetinaNet 教师后冲突区域进一步增加。

简单屏蔽高冲突区域会丢掉最有信息的困难位置。CrossKD 的思路是保留这些位置，但改变教师信号进入学生的路径：学生自己的预测只接受检测真值，教师信号通过“学生前半头+教师后半头”的混合分支回到学生中间特征。

## 方法总览

设一个检测分支由 \(n\) 个卷积层 \(C_1,\ldots,C_n\) 组成，\(f_i\) 是第 \(i\) 层中间特征，\(f_0\) 是 FPN 输入。学生正常经过自己的完整 head 得到 \(p^s\)，教师得到 \(p^t\)。CrossKD 选取学生 \(f_i^s\)，将其送入教师的 \(C_{i+1}^t,\ldots,C_n^t\)，得到跨头预测 \(\hat p^s\)。教师 head 参数冻结，但蒸馏梯度可穿过这些层回到学生 \(f_i^s\)。

分类分支以教师分类分数为软标签，采用 QFL；若回归头直接预测框偏移，则用 GIoU；若像 GFL 一样输出位置分布，则用 KL/LD 蒸馏分布。全图位置等权参与，不设计复杂区域筛选。

## 方法详解

传统预测模仿写为

\[
L_{KD}=\frac1{|S|}\sum_{r\in R}S(r)D_{pred}(p^s(r),p^t(r)),
\]

\(r\) 是预测图位置，\(S(r)\) 是区域权重，\(D_{pred}\) 是分类 KL、回归 L1/LD 等距离。CrossKD 将学生最终预测替换为跨头预测：

\[
L_{CrossKD}=\frac1{|S|}\sum_{r\in R}S(r)D_{pred}(\hat p^s(r),p^t(r)),
\]

论文令 \(S(r)=1\)，即所有位置参与。\(\hat p^s\) 同时包含学生中间表示和教师任务头先验，比纯特征 L2 更接近最终检测目标，又避免直接扭曲 \(p^s\)。

总目标为

\[
L=L_{cls}(p_{cls}^s,p_{cls}^{gt})+L_{reg}(p_{reg}^s,p_{reg}^{gt})
+L_{CrossKD}^{cls}(\hat p_{cls}^s,p_{cls}^t)+L_{CrossKD}^{reg}(\hat p_{reg}^s,p_{reg}^t).
\]

前两项只沿学生完整 head 回传，后两项沿冻结教师后半 head 回到学生中间层。这样学生 head 后段专注真值，蒸馏只塑造较早的潜在特征，冲突被平滑而非删除。

## 实验与证据

实验在 MS COCO minival 上，主要使用 GFL：ResNet-101 教师 44.9 AP，ResNet-50 学生 40.2 AP。CrossKD 将学生提升到 43.7（+3.5），超过 LD 的 43.0 与 PKD 的 43.3；与 PKD 组合达到 43.9。CrossKD 还将 RetinaNet-R50、FCOS-R50、ATSS-R50 分别从 37.4、38.5、39.4 提升到 39.7、41.3、41.8，均超过对应 ResNet-101 教师。

接入位置消融使用 GFL R50→R18：基线 35.8 AP，选择 \(i=0,1,2,3,4\) 分别为 38.2、38.3、38.6、38.7、38.2，直接 LD 为 37.8；第 3 个中间特征最好。相同位置的 PKD neck/cls/reg/cls+reg 最好为 38.0，CrossKD 为 38.7。严重冲突实验更有辨识度：ATSS-R101 教师直接 KD 使 GFL-R50 降到 39.7，CrossKD 为 42.1；更弱且分配器不同的 RetinaNet-R101 教师使直接 KD 崩到 30.3，而 CrossKD 仍为 41.2，高于 40.2 基线。

异构 backbone 上，Swin-T 教师蒸馏 RetinaNet-R50，CrossKD 为 38.0，PKD 为 37.2；ResNet-50 教师到 MobileNetV2 学生，基线 30.9、PKD 33.2、CrossKD 34.1。论文还观察到同时加入传统 prediction mimicking 会降至 38.1，低于单独 CrossKD，说明直接输出冲突会被重新引入。

分类与回归分支的距离函数保持任务特定，而“交叉”只改变预测生成路径。GFL 的分类 cross-head 输出用 QFL 拟合教师软分数，离散回归分布用 KL；RetinaNet、ATSS、FCOS 这类直接框回归则用 GIoU。这样 CrossKD 并不发明新的分类或定位距离，而是让已有 prediction mimicking 损失作用在更合适的变量 \(\hat p^s\) 上，论文的提升因而主要来自监督解耦。

梯度分析支持这一解释。直接 prediction mimicking 最容易减小学生与教师输出距离，却会让学生到真值的距离更大；CrossKD 虽不直接约束学生最终输出，仍能逐步缩小师生差异，同时保持更小的真值距离并更快提高 AP。交叉位置太靠前时，教师后半 head 承担过多变换，学生特征难以适配；太靠后则接近直接输出模仿，隔离冲突的空间不足，因此中间位置取得最好结果。

教师后半 head 在蒸馏时保持冻结，所以跨头分支不会把教师参数重新适配到学生；真正被更新的是产生 \(f_i^s\) 的学生网络。这一点保证 \(\hat p^s\) 始终由固定教师映射解释，也让不同交叉位置的比较具有明确含义。

## 对 YOLO-Agent 的启发

接入点应选在 YOLO decoupled head 的中后部卷积：学生前若干层正常提取特征，将某一层特征复制到冻结教师 head 的剩余层，分别生成分类和回归 cross-head 输出。对照组应包含无 KD、直接 logits/DFL KD、同层 PKD、CrossKD 不同 \(i\)、CrossKD+直接 KD；若师生 assigner 不同，再单列冲突压力测试。

指标除 AP/AP75 外应记录学生输出到真值、学生输出到教师的 L1 距离，以及冲突区域比例。论文中第 3 层最好且 CrossKD 比 PKD 高 0.7 AP，因此失败阈值可设为：最佳 \(i\) 相对直接 KD 增益低于 0.5 AP，或 CrossKD 在异分配器教师下不能保持高于无 KD 基线。若加入直接 KD 后再次下降，则应保留纯 CrossKD。工程上还要检查通道与归一化兼容；异构 head 若无法直接接续，需要最小适配层，但应把它作为独立对照，避免把容量增益算给 CrossKD。

## 优点

- 通过梯度路径而非样本屏蔽解决 target conflict，保留困难区域知识。
- 蒸馏对象仍是任务预测，比普通特征模仿更贴近分类与定位目标。
- 对不同检测器、不同 assigner 和异构 backbone 均有实验证据。

## 局限

- 需要学生中间特征能输入教师后半 head，头结构或通道差异大时并非无缝。
- 训练阶段必须额外执行教师部分检测头，带来计算和显存开销。
- 最佳交叉位置依赖 head 深度与师生组合，论文没有给出无需搜索的选择准则。

## 评分

- **创新性：9/10**——用跨头预测和分离梯度路径重构 prediction mimicking。
- **实验充分性：9/10**——包含位置、分支、冲突、SOTA、检测器与异构实验。
- **可复现性：8.5/10**——目标清楚，但结构对接与最佳层需逐模型调整。
- **对 YOLO-Agent 价值：9/10**——特别适合有解耦多层 head、且师生分配策略可能不同的场景。
