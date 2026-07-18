---

# Variational Pedestrian Detection
title: "Variational Pedestrian Detection"
description: "将密集候选框视为潜变量，以定制 AEVB 优化拥挤场景行人检测匹配过程的论文笔记。"
tags:
  - 行人检测
  - 变分推断
  - AEVB
  - 密集遮挡
  - CrowdHuman
  - CVPR2021
---

- **论文链接**：[CVPR 2021 官方论文页面](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Variational_Pedestrian_Detection_CVPR_2021_paper.html)

**???????**: [?Variational Pedestrian Detection?????????????????????????](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_Variational_Pedestrian_Detection_CVPR_2021_paper.html)
- **官方代码**：论文正文未给出作者 GitHub 地址，未发现可确认的官方代码仓库；实现基于 MMDetection。可通过上述官方论文页面获取论文及补充材料。

## 一句话总结

这项工作不再把密集候选框 \(z\) 当作需要由 IoU 规则确定的直接优化目标，而是把它建模为连接输入图像 \(f\) 与最终检测框 \(x\) 的辅助潜变量。作者为此提出检测定制的 Auto-Encoding Variational Bayes（AEVB）算法：网络学习候选框分布而非单一坐标，通过训练期随机采样主动探索候选框与拥挤行人真值之间的匹配空间，最终直接优化稀疏检测结果。

完整数据流由两个概率模块组成：卷积网络参数化的变分密集候选生成模块 \(q_\phi(z|f)\) 先把图像编码为候选框分布，再由最终检测提取模块 \(\tilde p(x|z)\) 从采样候选框得到检测结果。每个位置预测九个变量，即一个分类分数、四个定位均值 \(\mu\) 与四个标准差 \(\sigma\)；训练时利用重参数化 \(z=g_\phi(\epsilon)\) 传播伪检测似然的梯度，推理时不采样，只取均值作为最大似然定位结果。

实验在 CrowdHuman 与 CityPersons 上进行，以 log-average Miss Rate（MR，越低越好）为主指标，比较 RetinaNet、RFB-Net、FreeAnchor、FCOS、Faster R-CNN、Adaptive-NMS、OP-MP 等方法。在 CrowdHuman 上，FreeAnchor 从 52.83 降至 50.67，FCOS 从 48.31 降至 47.57；关键消融显示，仅把 FreeAnchor 的 L1 定位损失换成 IoU 损失得到 51.61，再加入 AEVB 达到 50.67，说明 1.22 点来自 IoU 建模，额外 0.94 点来自变分优化。

## 研究背景与问题

拥挤场景中的困难并不只是“框回归不准”。一个 anchor 往往同时与多个完整人体框高度重叠，离线分配先按固定 IoU 规则指定真值，再进行回归，容易形成歧义；FreeAnchor、HAMBox 等在线方法虽然先调整候选框或联合匹配，却仍依赖人工设计的匹配规则，探索空间有限。

本文将关注对象从候选框 \(z\) 转移到最终检测 \(x\)。其证据下界由两部分组成：一是 \(q_\phi(z|f)\) 与先验 \(p(z)\) 的 KL 正则，二是候选分布下伪检测对数似然的期望。前者约束候选框不要任意漂移，后者通过随机抖动搜索更好的真值匹配，从而把“学习定位”与“学习匹配”统一起来。

## 方法总览

作者采用单变量正态分布描述四个定位量。训练时从独立噪声分布采样 \(\epsilon\)，经均值和标准差重参数化为候选框 \(z\)，计算可微的伪检测似然，再将其梯度与 KL 梯度共同更新检测器。随着模型收敛，前景候选框的方差逐渐减小，匹配由广泛探索转向稳定回归；低置信度候选通常保留较大方差，以搜索可能漏配的真值。

先验选为标准正态分布，其作用是把密集候选限制在原 anchor 附近，并抑制方差退化为无意义的点估计。可视化中，AEVB 的分类响应比最大似然训练更集中、背景更干净；前景区域方差明显低于背景，且一个候选与越多真值框重叠，其方差中位数及大方差离群点越少，体现出针对遮挡程度的自适应性。

## 方法详解

真实的稀疏检测似然难以直接求密度，因此作者基于 FreeAnchor 构造可微的伪检测似然。正样本匹配质量定义为分类分数与 IoU 的乘积：
\[
M_{ij}=z^{cls}_j\operatorname{IoU}(x_i,z_j).
\]
对每个真值框选取 IoU 最高的一组候选，并使用 Mean-max 代替硬最大值，近似该真值被召回的概率。

负样本部分把 Focal Loss 重新解释为概率化在线难例挖掘：为候选框引入伯努利变量，只有被抽中的高价值候选才参与负似然。这样既避免海量低分背景主导训练，也更符合 AP、MR 等排序指标对高分误检更敏感的特点。IoU 还替代原有参数化定位似然，既免去额外超参数，又能由变分采样平滑定位梯度的尖锐区域。

## 实验与证据

CrowdHuman 使用 15,000 张训练图像、339,565 个人体实例和 4,370 张验证图像；CityPersons 使用 2,112 张图像训练，在 500 张验证图像上评估，训练包含 Reasonable 与 Highly Occluded 子集。主干为 ResNet-50-FPN，基础架构为 RetinaNet，每像素默认设置一个 anchor，并分别验证 anchor-based FreeAnchor 与 anchor-free FCOS。

CityPersons 上，FreeAnchor 的 Reasonable MR 从 14.8 降至 13.6，Highly Occluded MR 从 42.8 降至 41.5；两阶段版本仅用 AEVB 优化 Faster R-CNN 的 RPN，第二阶段保持不变，在 CrowdHuman 上从作者实现的 42.4 降至 40.7，优于 OP-MP 的 41.4。按遮挡划分时，FreeAnchor+AEVB 在 Bare、Partial、Heavy 子集分别取得 46.91、52.19、64.32，均优于普通 FreeAnchor 的 49.32、57.15、66.48。

## 对 YOLO-Agent 的启发

可把该思想作为 YOLO-Agent 的训练策略候选，而不是新增推理期结构：保留 YOLO 的分类与框回归头，将四个确定性框参数扩展为四个均值和四个标准差，训练阶段重参数化采样候选框，并以分类分数乘 IoU 构造正伪似然。KL 项约束采样框围绕 YOLO 网格点或先验框，推理阶段删除采样分支、仅输出均值，因此不应增加 NMS 前的候选数量。

若 YOLO 版本采用 anchor-free 解码，应对距离四边的四个回归量分别建模分布；若采用 anchor-based 解码，则对中心偏移、宽和高建模。不能直接复制 FreeAnchor 的候选袋规则而不做适配，必须保持 YOLO 原有尺度分配、正样本范围和解码方式一致，否则无法判断收益来自 AEVB 还是标签分配变化。

## 优点

核心优势是插件化：它既适用于 FreeAnchor、FCOS，也能只改 Faster R-CNN 的 RPN；推理仍取均值，不增加测试时随机采样。多 anchor 消融中，FreeAnchor 从单 anchor 的 50.67 提升到双 anchor 的 50.31、四 anchor 的 49.93，但训练时间由 8.63 小时增至 11.91 小时，说明更多采样可继续扩展探索，却不是免费收益。

## 局限

局限首先在于收益依赖可微伪似然，且正负样本构造仍继承 FreeAnchor、Focal Loss 等人工设计。其次，论文只对行人完整框进行验证，没有证明该概率模型在多类别、极小目标或显著尺度变化下仍稳定。训练需要额外预测标准差并进行随机采样，约增加训练成本；方差在推理时被丢弃，也未被进一步用于置信度校准或 NMS。

## 评分

复现实验应设置三组严格控制：① 原始 FreeAnchor，使用原定位损失与最大似然训练；② 仅将定位项替换为 IoU 的 FA-modified；③ 保持相同主干、输入尺度、单 anchor、训练轮次和 NMS，仅加入九变量输出、重参数化采样及 KL 正则的 FA+AEVB。另以 FCOS 基线与 FCOS+AEVB验证是否独立于 anchor，并以 Faster R-CNN 和仅改 RPN 的 Faster R-CNN+AEVB验证两阶段迁移。

主指标必须报告 CrowdHuman 全集及 Bare、Partial、Heavy 三种遮挡级别的 MR，同时报告 CityPersons 的 Reasonable、Highly Occluded MR；辅助记录训练时间、推理时间和每像素 anchor 数。若 FA+AEVB 未低于 FA-modified 的 51.61，或 FCOS+AEVB 未优于 48.31，或 Heavy 子集没有改善，则应判定“变分匹配有效”未被复现；若测试时间明显增加，则说明实现错误地保留了推理采样，违背论文只取均值的设定。
