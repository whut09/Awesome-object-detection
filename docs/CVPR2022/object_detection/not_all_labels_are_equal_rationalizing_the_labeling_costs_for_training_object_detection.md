---
title: "Not All Labels Are Equal: Rationalizing the Labeling Costs for Training Object Detection"
description: "梳理 AL-SSL 如何用增强不一致性发现弱类难样本，并以高置信伪标签抑制主动学习造成的数据分布漂移。"
tags: ["CVPR 2022", "主动学习", "半监督检测", "伪标签", "标注成本"]
---

# Not All Labels Are Equal: Rationalizing the Labeling Costs for Training Object Detection

**官方论文**：[CVF 论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Elezi_Not_All_Labels_Are_Equal_Rationalizing_the_Labeling_Costs_for_CVPR_2022_paper.html) · [官方 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Elezi_Not_All_Labels_Are_Equal_Rationalizing_the_Labeling_Costs_for_CVPR_2022_paper.pdf)  
**官方代码**：[NVlabs/AL-SSL](https://github.com/NVlabs/AL-SSL)

## 一句话总结

AL-SSL 用原图与水平翻转图之间的检测不一致性补足熵采样对低性能类别的盲区，再把高置信框作为伪标签补回“容易样本”，从而让有限人工预算同时覆盖难例、弱类和原始数据分布。

## 研究背景与问题

传统检测主动学习通常把熵、预测损失或 Bayesian uncertainty 最大的图像送去标注。问题在于 acquisition function 本身来自尚未成熟的检测器：对 bottle、pottedplant、chair 等弱类，模型可能错误却高度自信，熵并不高；持续只抽取困难图像又会让新训练集偏离测试分布。另一方面，普通 consistency-based SSL 遇到增强前后预测不一致的图像时无法有效学习，朴素伪标签还会把过度自信的错框当真，并把图中未伪标的真实物体误当背景。

## 方法总览

每个主动学习周期先以标注、未标注与伪标数据训练 SSD300。进入采样阶段后，同一图像及其水平翻转版本分别前向；框通过最大 IoU 配对，类别分布的双向 KL 散度构成 object inconsistency，图像分数取所有 NMS 后目标的最大不一致性 `I`。同时计算最大类别熵 `H`，最终 acquisition score 为 `A=H×I`，按分数挑选预算内图像人工标注。未被选中的池中，置信度超过阈值 `τ` 的框由上一周期模型自动标注，其余区域仍作为未标注数据参与一致性训练。

## 方法详解

### 鲁棒性采样而非只看置信度

水平翻转后的框先映回原坐标系，再以 IoU 最大原则一一寻找对应预测。分类不一致损失是两个类别概率分布之间对称 KL；定位不一致把中心偏移、宽、高做平方差，其中横向中心偏移因翻转而取反。图像级聚合选择最大值而非平均值，因为一张图只要包含一个困难目标就有标注价值。熵善于处理已有较强判别能力的类别，不一致性则不依赖预测类别是否正确，两者相乘避免任一信号单独主导。

### 伪标签与局部未标注区域

模型仅对 `max(c_i)≥τ` 的预测生成 one-hot 伪标签。训练损失不是把整张伪标图当完全标注图，而是在标准 MultiBox 分类项中额外加入 pseudo-positive 项；未得到伪标签的候选区域不会被强制归入背景。总损失由标注/伪标样本的分类损失、Smooth L1 框回归，以及所有样本上原图与翻转图的分类和定位一致性损失共同组成。这样，难且不稳定的样本去人工标注，容易且可信的目标免费扩充监督，中间区域继续接受 SSL 约束。

## 实验与证据

- VOC07+12 使用 16,551 张训练图与 4,952 张 VOC07 test；COCO train2014 为 83k，val2017 为 5k。VOC 初始随机标 2,000 张，COCO 初始 5,000 张，随后均做 5 个周期、每周期新增 1,000 张人工图像。
- 所有方法统一为 VGG-SSD300，SGD 训练 120k iterations，batch 32，初始学习率 0.001，在 80k 与 100k 衰减；VOC 的 `τ=0.99`，COCO 为 `0.75`，匹配 IoU 阈值 0.5，并报告三次独立训练均值。
- VOC 第一周期相对 random 提升 10.5%，比 MI-ALD 高 8.2%；最后周期相对 random 高 9.1%，比 PM 高 2.8%。COCO 第一周期比 random 高 5.8%、比 PM 高 2.7%，后续周期保持领先。
- acquisition 消融中，VOC 第一周期 random/entropy/inconsistency/unified 分别为 67.19/67.24/67.39/68.40 mAP；在 unified score 上加入伪标签又带来 3.7% 相对提升。COCO 第一周期伪标签贡献 3.1% 增益。
- 不做 SSL 时，不一致性采样在五周期得到 63.26/65.79/67.16/68.65/70.33，明显低于带 consistency loss 的 67.39/70.42/72.43/72.80/74.90，说明 acquisition 必须建立在模型确实尝试过稳定这些未标注图像的前提上。
- `τ` 消融表明 VOC 使用 0.99 的伪标签正确率约 96%，降至 0.9 或 0.5 会持续伤害性能；统一方法最多可在达到同等检测性能时节省 82% 人工标注成本。

## 对 YOLO-Agent 的启发

可把该论文变成“数据采购 agent”，而不是网络结构插件。**Harness** 以同一 YOLO checkpoint 和固定初始标注集启动五轮预算实验。**对照组**平行运行 random、entropy、flip-inconsistency、`entropy×inconsistency`、统一分数+伪标签五条轨迹；每轮必须保存被选图像、类别频次、伪标签框和模型权重。**指标**除 mAP50-95 外，还包括每千个人工框带来的 AP 增量、最差 20% 类别的 AP、采样集与全池类别分布的 Jensen–Shannon 距离、伪标签 precision/recall、漏标区域被当背景的比例。**失败判断**：统一方案仅提升头部类、弱类均值不优于 entropy、分布距离连续扩大，或伪标签 precision 低于 95% 且总体 AP 低于无伪标签组时，应停止自动扩标；这比只看最终 mAP 更忠实于论文“标签价值不等且需控制漂移”的核心命题。

## 优点

- 主动学习、半监督一致性与伪标签被组织成同一闭环，而非三个互不通信的技巧。
- 对弱类逐类分析，解释了不一致性为何在复杂 COCO 类别上优于单纯熵。
- 固定 SSD300 与采样预算，和多种单模型、多模型 AL 基线比较，实验控制较严格。

## 局限

- 框匹配和增强仅采用水平翻转，面对尺度、颜色、遮挡变化时鲁棒性信号是否可靠没有验证。
- 伪标签阈值依赖数据集手工设定，且 VOC 仍有约 3.7% 错误伪标。
- 论文按整图付费并标注图中所有框，未覆盖按框、按类别或不同人工时长计价的现实标注系统。

## 评分

- **创新性：8/10**——关键价值在于把鲁棒性采样与分布修复放入同一标注决策。
- **实验说服力：8.5/10**——跨 VOC/COCO、多类基线、逐类与阈值消融均较充分。
- **YOLO-Agent 适配度：9/10**——无需改变检测头，适合直接作为数据循环策略。
- **综合：8.5/10**——对预算有限的检测项目很实用，但部署前必须重新校准伪标签可靠性和真实标注成本。
