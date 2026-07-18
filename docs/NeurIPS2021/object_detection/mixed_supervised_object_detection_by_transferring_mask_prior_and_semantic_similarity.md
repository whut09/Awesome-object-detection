---
title: "Mixed Supervised Object Detection by Transferring Mask Prior and Semantic Similarity"
description: "NeurIPS 2021 论文详解：TraMaS 从全框标注基类向仅图像级标注新类迁移 mask prior 与 SimNet 语义相似度。"
tags: ["NeurIPS 2021", "目标检测", "弱监督检测", "混合监督", "TraMaS"]
---

# Mixed Supervised Object Detection by Transferring Mask Prior and Semantic Similarity

- **论文页面**：[NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/20885c72ca35d75619d6a378edea9f76-Abstract.html)
- **官方 PDF**：[Paper PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/20885c72ca35d75619d6a378edea9f76-Paper.pdf)
- **官方代码**：[bcmi/TraMaS-Weak-Shot-Object-Detection](https://github.com/bcmi/TraMaS-Weak-Shot-Object-Detection)（论文摘要明确给出）
- **发表会议**：NeurIPS 2021

## 一句话总结

TraMaS 用全框标注基类训练“带粗语义 mask 的类无关 proposal 生成器”和成对框 SimNet，再把这两种先验迁移到只有图像级标签的新类：mask prior 改善候选框覆盖，语义相似度权重抑制 MIL 挖出的错类伪框。

## 研究背景与问题

弱监督目标检测只有图像级类别标签，WSDDN 式 MIL 容易把最具判别性的局部当成整物体，也容易利用共现背景。Weak-shot Object Detection 额外拥有一组类别不重叠、但有精确框标注的基类；既有方法主要迁移 class-agnostic objectness，仍未充分利用基类的像素布局和“两个框是否同类”的知识。

论文设置两个数据域：源集 `D_s` 对 base categories 有框与类别，目标集 `D_t` 对 novel categories 只有图像级标签，基类与新类不重叠。TraMaS 的目标不是直接把基类分类器迁过去，而是迁移两种跨类别稳定能力：粗 mask 如何帮助回归完整物体，以及同类 proposal 在区域特征上应当彼此相似。

## 方法总览

源图先经 ResNet-50 backbone；倒数第二层特征进入三层卷积加 normalized Global Weighted Pooling（nGWP）的 Mask Generator，生成类别粗语义 mask。该 mask 追加到 backbone 后层特征，形成 mask-enhanced feature map，再由 RPN、objectness predictor 与 box regressor 产生 candidate boxes。

目标图的 candidate boxes 进入 WSDDN MIL Classifier：classification branch 在类别维做 softmax，detection branch 在 proposal 维做 softmax，两者逐元素相乘并沿 proposal 求和得到图像级分类；分数高于 0.8 的框被赋予伪类别。随后 SimNet 对同一伪类别的一批框两两计算语义相似度，平均相似度成为每个伪框的训练权重，最终加权监督下一轮 objectness 与回归。

## 方法详解

### 1. Object Detection Network 与 MIL Classifier

检测网络基于 Faster R-CNN，但 detection head 只含 class-agnostic objectness predictor 和 box regressor。proposal 与真值 IoU 达阈值时 objectness 为 1；低于 0.05 的 proposal 被过滤。基类真值首先教会 RPN/回归器产生完整物体候选，这项能力再用于新类。MIL 只在目标集工作，以图像级 binary cross-entropy `L_mil` 学新类，并从候选中挖伪框。

### 2. Mask Prior Transfer

Mask Generator 对全部类别输出像素分数并加入常数背景通道，softmax 后得到粗 mask；nGWP 汇聚成图像级分数，以 multi-label soft-margin loss `L_mask` 训练。与直接按 CAM 选正框不同，TraMaS 把粗 mask 拼接到 backbone feature map，让 objectness 与 box regression 自行学习如何利用形状先验。论文比较拼在最后层与倒数第二层，默认选择最后层。

### 3. Semantic Similarity Transfer

SimNet 是两层全连接加 sigmoid。训练时从基类随机选 `K=8` 类、每类 `M=8` 个真值框，组成 64 个 region features，再构造所有成对特征；同类 pair 标签为 1、异类为 0，以 BCE `L_sim` 学习。应用到新类时，从同一伪类别采样 `M` 个框，框 `i` 与其他框双向相似度的平均值 `w_i` 作为其 objectness 与 Smooth-L1 回归损失权重；离群错标框因与多数框不相似而被降权。

### 4. 三步迭代训练

每轮先在源集用基类真值与上一轮加权新类伪框训练 ODN+Mask Generator，损失为 `L_obj^w + γL_reg^w + αL_mask`，其中 `α=0.1、γ=1.0`；第二步在目标集更新 backbone、Mask Generator 与 MIL，`L_mil+βL_mask`，`β=0.1`；第三步冻结 backbone/mask，用基类真值框训练 SimNet，再给源集和目标集的新类伪框赋权。论文共迭代 `T=4` 次。

## 实验与证据

- **数据**：目标集是 VOC 2007，trainval 5011 张仅保留图像级标签，test 4952 张评 mAP，trainval 评 CorLoc。源集一是去掉 VOC 20 类相关图像后的 COCO-60（21987 train+921 val），二是 ILSVRC-179（共 150134 张）。
- **COCO-60→VOC-20**：比较 WSDDN、WCCN、OICR、PCL、Ren et al.、CASD，以及 MSD、Chen et al.、Zhong et al.。ResNet-50 的 `Ours+distill` 达 62.9 mAP，强基线 Zhong et al.+distill 为 60.2；对应 TraMaS CorLoc 为 77.7。单尺度 TraMaS 为 60.9 mAP。
- **ILSVRC-179→VOC-20**：ResNet-50 TraMaS 达 58.3 mAP、74.8 CorLoc，高于 Zhong et al. 的 56.5 mAP；VGG16 版本为 57.8/74.1。
- **组件消融**：无 mask/相似度为 58.2 mAP；mask 拼最后层 59.7，拼倒数第二层 59.1；仅 cosine similarity 58.8，训练 SimNet 59.5；最后层 mask + SimNet 达 60.9。两项先验既各自有效，又能叠加。
- **源数据规模**：仅用 COCO-60 的 20%/50% 时，单尺度 mAP 为 59.4/59.8，完整源数据为 60.9，说明更多全监督基类数据持续提升迁移。
- **类别重叠扩展**：COCO-70 设置让 10 类同时拥有源集框监督与目标图像级监督，`Ours+distill` 在 mixed categories 上 69.9 mAP，而纯 novel categories 为 62.4。

消融中的 60.9 不是单纯增加参数得到：粗 mask 单独解决 proposal 是否覆盖整物体，SimNet 单独处理伪类别是否可信；二者联合比 59.7 和 59.5 都高，说明候选质量与标签噪声是两个不能互相替代的瓶颈。这也是 Harness 必须同时记录候选 recall、框 IoU 与离群权重的原因。

## 对 YOLO-Agent 的启发

TraMaS 提供了一种比“整模型伪标签”更细的知识迁移：YOLO-Agent 可以用框标注充足的老类别训练 class-agnostic box quality、mask-conditioned feature 和 pairwise similarity，再服务于只有图像标签的新类别。对于长尾增量检测，SimNet 权重尤其适合替代单纯按 MIL confidence 硬筛选，因为高分类分数并不保证伪框覆盖完整物体或类别正确。

### 专属 Harness：新类伪框去噪与完整性测试

- **对照组**：A 为 YOLO proposal/objectness + 图像级 MIL；B 在 A 上把 CAM/mask 拼入最后层特征；C 在 A 上仅用 region cosine 给伪框加权；D 在 A 上使用基类真值框训练的两层 SimNet；E 同时启用最后层 mask 与 SimNet。所有组固定 base/novel 划分、候选阈值 0.05、伪框阈值 0.8 和四轮 refinement。
- **观测指标**：novel 类 mAP50-95、CorLoc、伪框 precision、伪框与真值 mean IoU、完整物体覆盖率、错误类别离群框平均权重，以及最判别局部框占比。
- **通过标准**：B 必须提高候选框 recall/完整覆盖率，D 必须让错类伪框权重显著低于正确伪框且优于 C；E 在至少 3 个种子上同时超过 A 的 mAP 与 CorLoc，并保留 B、D 各自对应的中间指标改善。
- **失败判断**：mask 只提升图像分类却不改善候选 IoU；SimNet 权重与伪框正确性无相关；E 的收益在关闭任一模块后不变；或新类视觉分布与基类差异过大导致正确框被系统性降权，均判定迁移先验不成立。

## 优点

- 将 mask prior 与 semantic similarity 明确拆成“候选框质量”和“伪标签去噪”两条数据流，作用边界清楚。
- 消融同时比较 mask 注入位置、cosine 与学习式 SimNet，能够支持具体设计选择。
- COCO-60 与 ILSVRC-179 两种源集、mAP 与 CorLoc 双指标验证了跨源数据稳定性。

## 局限

- 训练流程包含 ODN、Mask Generator、MIL、SimNet 和四轮伪框迭代，工程复杂且误差可能逐轮累积。
- 假设同一伪类别批次中 inlier 占多数；早期 MIL 若被系统性背景模式支配，平均相似度可能反而强化错误簇。
- 基类与新类完全不重叠且共享相似视觉结构的设定，并不覆盖开放词汇或极端跨域类别。

## 评分

- **创新性：8.5/10**：在 objectness 之外迁移 mask 使用能力与成对语义相似度。
- **实验充分性：8.5/10**：主对比、位置/相似度消融、源规模与重叠类别分析较完整。
- **工程可迁移性：7/10**：模块均可实现，但多阶段迭代和伪框管理成本较高。
- **综合评分：8.2/10**。
