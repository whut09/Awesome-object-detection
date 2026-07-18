---
title: "[论文解读] Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction"
description: "解读两步共形预测如何把类别不确定性传播到自适应边界框区间，并提供逐类覆盖保证。"
tags: ["ECCV 2024", "目标检测", "共形预测", "不确定性量化"]
---

# Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction

**论文**：[ECCV 论文页](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/12292_ECCV_2024_paper.php)  
**代码**：官方代码链接未提供  
**会议**：ECCV 2024

## 一句话总结

该文先用 ClassThr 为每个检测对象生成带逐类覆盖保证的标签集合，再从集合内选择最保守的类别专属框分位数，配合 Box-Std、Box-Ens 或 Box-CQR 构造坐标区间，从而把误分类风险纳入边界框不确定性保证。

## 研究背景与问题

已有检测框共形预测通常先假定类别预测正确，再按该类别的校准残差给四个坐标扩张区间。这个条件在多类别检测中很脆弱：一旦类别错了，就会选择错误的分位数，原本的 class-conditional coverage 不再成立。论文要保证的是，对真实类别为 \(y\) 且确实被检测到的对象，四个真实坐标同时落入预测区间的概率至少为 \(1-\alpha_B\)。

共形预测依赖校准样本与测试样本可交换，能够给出有限样本、分布无关的边际或逐类覆盖保证，但它不覆盖漏检对象，也不解决数据分布变化。本文的关键扩展是把类别预测也变成共形集合，并把第一步的标签覆盖率传播到第二步的框覆盖率；同时针对不同大小对象需要不同区间宽度的问题，设计自适应框残差。

## 方法总览

两步流程均为后处理。第一步 ClassThr 在每个类别内部校准分类分数，输出可能包含多个类别的集合 \(\hat C_L(X)\)。第二步对集合中的每个类别查询其坐标分位数，并对每个坐标取最大值，再用选定的 Box 方法构造预测区间。论文用 max-rank 而非 Bonferroni 修正四坐标同时覆盖的多重检验问题。

```mermaid
flowchart LR
    A[黑盒检测器: 类别概率与框坐标] --> B[ClassThr 类别共形校准]
    B --> C[标签集合 C_L]
    C --> D[按坐标取集合内最大类别分位数]
    A --> E[Box-Std / Box-Ens / Box-CQR]
    D --> E
    E --> F[四坐标联合预测区间]
```

## 方法详解

### 第一步：ClassThr 标签集合

对类别 \(y\)，非一致性分数定义为 \(s(\hat f_L(X),y)=1-\hat\pi_y(X)\)，其中 \(\hat\pi_y\) 是检测器给出的类别概率。按类别分别在校准集上求分位数 \(\hat q_L^y\)，测试时纳入所有满足 \(\hat\pi_y(X)\ge 1-\hat q_L^y\) 的类别。由此得到

\[
P(l_{n+1}\in\hat C_L(X_{n+1})\mid l_{n+1}=y)\ge 1-\alpha_L.
\]

不同于只返回 argmax 的 Top，ClassThr 允许难例包含多个候选类别，并对每个类别分别控制漏覆盖。论文还比较 Naive 概率质量累积集合与包含全部类别的 Full；前者依赖分类器完美校准，后者有保证但集合和框区间会过宽。

### 第二步：自适应框区间

Box-Std 使用绝对坐标残差 \(|\hat c^k-c^k|\)，区间为 \([\hat c^k-\hat q_B^k,\hat c^k+\hat q_B^k]\)，简单但宽度固定。Box-Ens 将残差除以集成模型对该坐标预测的标准差 \(\hat\sigma(X)\)，测试区间变为 \([\hat c^k-\hat\sigma\hat q_B^k,\hat c^k+\hat\sigma\hat q_B^k]\)；\(\hat c^k\) 由置信度加权框融合得到，因此困难对象可获得更宽区间。

Box-CQR 额外训练上下分位回归头 \(\hat Q_{\alpha_B/2}\) 与 \(\hat Q_{1-\alpha_B/2}\)，非一致性分数取下界越界量与上界越界量的最大值，最终用共形分位数向两端扩张。它通过样本相关的上下分位预测实现自适应，而不是依赖集成方差。

标签集合得到后，第 \(k\) 个坐标使用

\[
\hat q_B^k=\max_{y\in\hat C_L(X)}\hat q_B^{k,y}.
\]

该最大化策略保守但有效。两步保证相乘：标签和框同时正确覆盖的概率至少为 \((1-\alpha_L)(1-\alpha_B)\)。实验设 \(\alpha_L=0.01\)、\(\alpha_B=0.1\)，目标联合框覆盖约为 90%。四个坐标分别校准会产生多重检验，max-rank 利用坐标相关性，在秩空间中比 Bonferroni 得到更紧的区间。

## 实验与证据

实验使用预训练 Faster R-CNN，并在 COCO validation、Cityscapes、BDD100k 上测试 person、bicycle、motorcycle、car、bus、truck 六类；真值框与预测框通过 IoU 阈值 0.5 的 Hungarian matching 配对，结果跨 100 次数据划分取平均。指标包括标签覆盖率、框联合覆盖率、标签集合平均大小、平均预测区间宽度 MPIW，以及 COCO AP。

在 COCO、Cityscapes、BDD100k 上，只有 ClassThr 与 Full 持续同时达到标签和框目标覆盖。Full 因始终包含全部类别而产生膨胀区间；ClassThr 的平均集合大小约为 4，并保持实际可用的框宽。Top 经常标签欠覆盖，样本增多后框也趋于欠覆盖；Naive 在原实验中框覆盖较好且区间紧，但附加实验显示它对分类器失校准敏感。

COCO 的 90% 目标覆盖比较中，Box-Std 在 Faster R-CNN、YOLOv3、DETR、Sparse R-CNN 上的双侧区间覆盖分别为 0.88、0.88、0.88、0.89，MPIW 分别为 55.47、61.73、45.34、41.92。DeepEns 与 GaussianYOLO 的双侧覆盖仅 0.21 和 0.08，表明启发式不确定性不能替代覆盖保证。Box-Ens 和 Box-CQR 对大物体的覆盖更均衡，其中 Box-Ens 在 BDD100k 上效率尤其好；max-rank 也比 Bonferroni 产生显著更紧的区间。

## 对 YOLO-Agent 的启发

最直接的接入点是 YOLO 推理后的校准层，而非训练损失：保存匹配成功的校准检测，按真实类别和四个坐标统计非一致性分数；分类侧实现 ClassThr，框侧先落地单模型 Box-Std，再增加多模型或多 checkpoint 的 Box-Ens。对照组应包含原始 YOLO 点框、Top+Box-Std、Naive+Box-Std、ClassThr+Box-Std，以及 ClassThr+Box-Ens。

验收同时看 class-conditional Cov 与 MPIW。按论文设置 \(\alpha_L=0.01,\alpha_B=0.1\)，若任一主要类别的四坐标联合覆盖低于约 0.9，或 ClassThr 不能同时修复误分类对象的标签与框覆盖，则判失败；在覆盖达标的方案中再选择 MPIW 最小者。若目标数据以小物体为主而 Box-Ens 反而显著扩大 MPIW，应回退 Box-Std；若对象尺度跨度大且大物体明显欠覆盖，则必须启用自适应归一化。还需明确报告漏检率，因为这些共形保证只对已匹配 TP 生效。

## 优点

- 把类别错误纳入框区间有效性，而不是把“分类正确”藏在前提中。
- 给出逐类、有限样本、分布无关的覆盖陈述，并处理四坐标联合覆盖。
- Box-Ens 与 Box-CQR 能随对象难度和尺度调整区间。
- 后处理与检测器结构解耦，覆盖 Faster R-CNN、YOLOv3、DETR、Sparse R-CNN。

## 局限

- 保证只覆盖被检测并成功匹配的真实对象，FN 完全不在保证范围内。
- 依赖校准集与测试分布可交换，明显域移位会破坏理论前提。
- 对标签集合取最大框分位数较保守，容易过覆盖并增大 MPIW。
- 逐类与多坐标分区需要足够校准样本，长尾类别可能难以稳定估计分位数。

## 评分

**9.0/10**。论文把多类别检测中的条件漏洞补得很完整，理论目标、算法和实验指标一致；代价是保证仍局限于 TP，且高安全等级下区间可能偏宽。
