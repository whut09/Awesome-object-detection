---
title: "[论文解读] Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection"
description: "解析 GFLV2 如何由边界框离散分布统计预测可靠定位质量。"
tags: ["CVPR 2021", "目标检测", "定位质量估计", "GFL"]
---

# Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.html)  
**代码**：官方条目未提供代码链接  
**发表**：CVPR 2021

## 一句话总结

GFLV2 从 GFLV1 的四条边离散回归分布中提取 Top-k 概率及均值，用极轻量的 Distribution-Guided Quality Predictor（DGQP）预测 IoU 质量，再与类别概率相乘形成 NMS 排序分数。

## 研究背景与问题

密集检测器需要 Localization Quality Estimation（LQE）给候选框排序，否则分类高分但定位差的框会在 NMS 中压掉更准的候选。YOLO Objectness、IoU-aware、centerness、ATSS 等都从普通卷积特征预测质量，但这些特征与最终框质量之间是间接关系，质量头必须额外学习“特征长什么样意味着框更准”。

GFLV1 已把每条边的连续偏移表示为离散概率分布。论文发现分布形状本身就是不确定性：尖锐分布通常对应明确、准确的边界，平坦或多峰分布对应模糊定位。GFLV2 因而不再只让分类特征猜 IoU，而是读取回归分布的统计量；输入与目标之间的相关性更直接。

## 方法总览

GFLV2 保留 GFLV1 的 General Distribution 和 Quality Focal Loss，但把 Classification-IoU Joint Representation 显式拆成类别表示 \(C\) 与质量标量 \(I\)。分类分支输出 \(m\) 类概率，回归分支输出左、右、上、下四个离散分布；DGQP 从四个分布抽取统计特征，经两层全连接网络预测 \(I\)，最终分数 \(J=C\times I\) 同时用于 QFL 监督和推理 NMS。

## 方法详解

GFLV1 对一条边在离散位置 \(y_0,\ldots,y_n\) 上预测概率 \(P(y_i)\)，回归值为期望

\[
\hat y=\sum_{i=0}^{n}P(y_i)y_i,
\]

其中 \(\sum_iP(y_i)=1\)。GFLV2 将联合分数分解为

\[
J=C\times I,
\]

\(C=[C_1,\ldots,C_m]\) 是分类分支的类别表示，\(I\in[0,1]\) 是 DGQP 给出的 IoU 表示，\(J\) 是最终逐类质量感知分数。训练与推理都使用 \(J\)，避免质量分数只在测试时相乘造成不一致。

对每个方向 \(w\in\{l,r,t,b\}\)，分布为 \(P_w=[P_w(y_0),\ldots,P_w(y_n)]\)。DGQP 选择每个分布最大的 \(k\) 个概率及其均值并拼接：

\[
F=Concat\left(\{Topk_m(P_w)\mid w\in\{l,r,t,b\}\}\right)\in\mathbb R^{4(k+1)}.
\]

Top-k 越大、均值越大，通常表示概率更集中；只取概率值而忽略其离散位置，使统计量对目标尺度和偏移位置更稳健。质量标量为

\[
I=\mathcal F(F)=\sigma(W_2\delta(W_1F)),
\]

\(\delta\) 为 ReLU，\(\sigma\) 为 Sigmoid，\(W_1\in\mathbb R^{p\times4(k+1)}\)、\(W_2\in\mathbb R^{1\times p}\)。典型设置 \(k=4,p=64\)，额外参数在 ResNet-50-FPN 模型中约占 0.003%。

## 实验与证据

实验在 COCO 上进行。COCO test-dev 中，GFLV2 ResNet-101 达到 46.2 AP、14.6 FPS，而同速 ATSS 为 43.6 AP；ResNet-50 为 44.3 AP、19.4 FPS，ResNeXt-101-DCN 为 49.0 AP。Res2Net-101-DCN 多尺度测试达到 53.3 AP。把 GFLV2 机制接入 RetinaNet、FoveaNet、FCOS、ATSS，分别带来 2.1、2.1、2.1、1.9 AP，且表中 FPS 基本不变。

关键消融显示，Top-4 加均值的 20 维统计达到 41.1 AP、44.9 AP75；只用 Top-4 为 40.8，只用均值为 40.2，加入方差反而为 40.9。显式分解形式 \(J=C\times I\) 为 41.1 AP，而把统计特征直接拼入分类头的 composed form 最好只有 40.7，且更慢。DGQP 让预测 IoU 与真实 IoU 的 Pearson 相关系数相对 GFLV1 提高 0.26，并带来 0.9 AP。效率对比中，GFLV2 与 GFLV1 都是 8.2 小时训练、19.4 FPS，而 PAA、RepPointsV2、BorderDet 有明显训练或推理开销。

统计特征消融给出了 DGQP 的具体设计边界。每条边只取 Top-k 时可以描述峰值大小，却缺少分布整体水平；只取均值又因为概率和固定，区分能力有限。二者组合后既保留主峰，又概括前几个高概率 bin 的集中程度。方差没有继续提升，说明带位置的二阶统计容易受偏移尺度影响，而只排序概率值能保持论文强调的尺度鲁棒性。默认隐藏维度 64 足以完成从 20 维统计到单一质量标量的映射。

显式分解优于 composed form 也具有训练含义。若直接把分布统计拼进分类特征并输出联合分数，网络仍需在一个向量中纠缠“是什么类别”和“框有多准”；\(J=C\times I\) 则把类别与定位质量分别建模，再在训练和测试统一相乘。QFL 对最终 \(J\) 监督，因此 DGQP 不是独立 IoU 回归器，而是与分类概率共同对真实类别的 IoU 目标负责。

论文的可视化显示，在 NMS 前多个重叠候选中，FCOS、ATSS、GFLV1 有时给最佳框较低质量分，导致其在 NMS 后消失；GFLV2 更常把最高分赋给最高 IoU 候选。训练曲线中 GFLV2 的 LQE 损失更快下降并收敛到更低水平，与“分布统计比普通卷积特征更容易学习质量”这一假设相符。速度表则证明收益不是由更重的边界采样模块换来。

兼容性实验还要求原检测器先支持四边分布表示。作者对 RetinaNet、FoveaNet、FCOS 和 ATSS 做最小修改，使每条边输出离散概率，再统一接入 DGQP；因此表中的约 2 AP 增益同时包含分布式框表示、联合分数与质量预测器的完整 GFLV2配置。复现时若模型本来已有 DFL，应把“仅分布回归、不加 DGQP”单独作为基线，才能隔离质量估计器本身的贡献。

从误差类型看，DGQP主要改变候选排序而非框坐标生成：边界期望仍由同一分布积分得到，新增标量只参与联合分类质量分数。它最可能改善 AP75 和 NMS 保留结果，而不会自动修正系统性回归偏差。这也解释了为什么论文把相关系数和 NMS 可视化作为核心证据，而不是声称 DGQP产生了更精确的原始回归值。

## 对 YOLO-Agent 的启发

若 YOLO-Agent 已采用 DFL/离散边界分布，DGQP 可直接接在四方向分布 logits 的 Softmax 之后：每方向取 Top-4 与均值，拼接后用两层 MLP 预测质量 \(I\)，再与类别概率相乘用于训练目标和 NMS。对照组应包括原 DFL YOLO、普通卷积 IoU head、仅 Top-k、Top-k+mean、以及 composed 拼接形式。

必须同时记录 AP、AP75、预测质量与真实 IoU 的 Pearson 相关系数、FPS 和训练时长。论文中相关系数提升 0.26、AP 提升 0.9，且几乎零速度损失；因此可将“相关系数增幅低于 0.1 或 AP75 不提升”设为质量估计失败，“FPS 下降超过 2%”设为轻量性失败。若小目标 AP 下降，应检查 DFL bin 数和 Top-k：分布过短时固定 \(k=4\) 可能覆盖过多概率，无法区分尖锐与平坦形状。

## 优点

- 直接使用回归不确定性统计预测定位质量，输入与目标关系清楚。
- DGQP 参数和计算量极小，推理分数仍是单一分类—质量联合表示。
- 可迁移到多种密集检测器，尤其适合已有离散框分布的模型。

## 局限

- 前提是检测器输出边界分布；连续回归头需先改为分布式表示。
- Top-k 与均值是手工统计，未显式利用多峰位置结构或四边之间的相关性。
- 分布尖锐并非总意味着框准确，系统性偏移也可能产生高置信错误。

## 评分

- **创新性：9/10**——首次把边界分布统计直接用于 LQE。
- **实验充分性：9/10**——包含输入统计、结构、相关性、兼容性与效率分析。
- **可复现性：9/10**——模块仅两层 FC，默认 \(k,p\) 与数据流明确。
- **对 YOLO-Agent 价值：9.5/10**——与采用 DFL 的现代 YOLO 回归头高度契合。
