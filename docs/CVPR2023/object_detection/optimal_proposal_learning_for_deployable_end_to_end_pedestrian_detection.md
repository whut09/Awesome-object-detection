---

# Optimal Proposal Learning for Deployable End-to-End Pedestrian Detection
title: "Optimal Proposal Learning for Deployable End-to-End Pedestrian Detection"
description: "基于 FCOS、C2F 渐进标签分配与 CPN 困难样本补偿的无 NMS 端到端行人检测方法笔记"
tags:
  - CVPR2023
  - 行人检测
  - 端到端检测
  - NMS-Free
  - FCOS
  - 拥挤场景
---

## 一句话总结

该方法以去除中心度分支和 NMS 的 FCOS 为基础，将 ResNet-50＋FPN 输出的多尺度特征送入共享检测头；检测头由回归分支、分类分支和 Completed Proposal Network（CPN）组成。分类分支末端不再使用 FCOS 原有的单层预测，而是接入 Coarse-to-Fine（C2F）堆叠分类块；回归分支输出边界框 \(B\) 与特征 \(f_{reg}\)，分类分支输出 \(S_{cls}\) 与 \(f_{cls}\)，CPN 接收 \(f_{cls},f_{reg}\) 产生补偿分数 \(S_{cpn}\)，最终置信度严格按 Hadamard 积 \(S=S_{cls}\cdot S_{cpn}\) 得到。

C2F 针对 CNN 在同一行人邻域内产生相似响应、直接 one-to-one 监督仍会留下重复高分框的问题。它串联若干由两个 \(3\times3\) 卷积构成的 classification block，每一步采用 one-to-\(M_i\) 分配，并保持 \(M_{i-1}>M_i\)：较大的 \(M_i\) 先提供丰富正样本学习粗粒度边界，较小的 \(M_i\) 再收紧监督、逐步修正分类决策边界；每步使用 Focal Loss，但推理时只保留最后一个分类块的分数。CrowdHuman 消融中，两步配置 \(M=\{16,4\}\) 得到 44.9 mMR，优于一步 \(\{4\}\) 的 46.1、一步 \(\{9\}\) 的 45.8、三步 \(\{16,9,4\}\) 的 45.3。

CPN 专门处理遮挡和小尺度困难行人，由 F1、F2、F3 三条流组成。F1 直接拼接 \(f_{cls}\) 与 \(f_{reg}\)，保留所有 proposal 的优化通路；F2 先拼接并卷积，再通过 Multi-scale Feature Enhancement（MSFE）收集相邻 FPN 层、双线性插值到当前尺度，并执行跨层邻域 3D max pooling，以高响应补偿弱 proposal；F3 对偏向大目标的回归特征执行 Neg，与分类特征拼接后经 Sigmoid 形成门控，再与原始拼接特征逐元素相乘并送入 MSFE，从而强调小目标和遮挡目标。三流相加后依次通过 GN、ReLU、卷积和 Sigmoid 输出 \(S_{cpn}\)。

## 研究背景与问题

传统行人检测依赖 NMS 删除同一实例的重复框，但拥挤区域中固定 IoU 阈值存在两难：阈值低会误删高度重叠的真实行人，阈值高又会留下重复假阳性。PED 等 query-based 无 NMS 方法虽然提高性能，却有训练时间长、计算量大和部署链路复杂的问题。本文选择工业系统更容易支持的单阶段、无锚 CNN 检测器，核心目标不是额外学习一个抑制后处理，而是让网络自身为每个 GT 只生成一个可靠高分 proposal。

困难之处有两层：一是 one-to-one 标签与 CNN 的局部平移相似性冲突，同一身体显著区域附近的候选点容易得到相似特征和高分；二是严格 one-to-one 会减少正样本，使被遮挡身体和低分辨率小行人的表示进一步恶化。C2F解决“重复候选如何逐渐收敛为一个”，CPN解决“唯一候选不能以遗漏困难实例为代价”，二者分别约束精度与召回。

## 方法总览

总损失为 \(L=L_{reg}+L_{cls}+L_{c2f}\)。其中 \(L_{reg}\) 使用 IoU Loss；最终 one-to-one 分类分数使用 Focal Loss；\(L_{c2f}\) 是各渐进分类块损失之和。标签分配采用 DeFCN 中的 one-to-one 与 one-to-many 规则，论文强调贡献并非设计新的匹配代价，而是把正样本数量变化组织成有方向的顺序优化。

CPN 的关键不是简单扩大感受野。F2 的局部最大值传播可能把早期假阳性噪声扩散给邻近 proposal，并使被替换的困难 proposal 丢失梯度；因此必须加入 F3 独立补偿路径。完整 CPN 中，F1、F2、F3 在 CrowdHuman 上依次将 mMR 从 47.2 降至 47.0、46.0、44.9，说明单纯特征拼接作用有限，跨尺度响应迁移和困难样本专用门控才是主要收益来源。

## 方法详解

实验覆盖 CrowdHuman、TJU-Ped-campus、TJU-Ped-traffic 和 Caltech。CrowdHuman 含 15,000 张训练图、4,370 张验证图，平均每图约 23 人；TJU 两部分别包含 55,088 张图/329,623 个实例和 20,338 张图/43,618 个实例；Caltech 训练、测试集分别为 42,782 和 4,024 张图。主指标是 FPPI 位于 \([10^{-2},10^0]\) 时的 log-average miss rate，即 mMR，越低越好，同时报告 AP、Recall，以及 R、RS、HO、R+HO、A 子集。

CrowdHuman 上，OPL 达到 44.9 mMR、91.0 AP、97.7 Recall；对应 PED 为 45.6、89.5、94.0，DeFCN 为 48.9、89.1、96.5。Caltech 上，OPL 在 R、HO、R+HO 分别取得 5.2、30.1、11.7 mMR，优于 FCOS 的 6.9、34.1、14.2。TJU-Ped-traffic 的五个子集结果为 23.4、28.8、62.7、28.0、38.7，也全部优于 FCOS 与 DeFCN。

## 实验与证据

CrowdHuman 基线是删除中心度分支和 NMS、直接采用 one-to-one 分配的 FCOS，结果为 49.3 mMR、90.3 AP、97.8 Recall。仅加入 C2F 后为 47.2/90.3/97.7，仅加入 CPN 后为 47.0/90.6/97.7，二者结合为 44.9/91.0/97.7：相对基线，完整系统降低 4.4 个百分点 mMR、提高 0.7 个百分点 AP。换用 ResNet-101 仅把 mMR 从 44.9 改善到 44.7，说明收益主要来自 proposal 学习机制，而非扩大骨干容量。

## 对 YOLO-Agent 的启发

对 YOLO-Agent，可把该论文视为一个“无 NMS 密集检测”候选策略，而不是可直接移植的插件。适合让 Agent 在 FCOS/YOLO 式共享头上搜索递减的 one-to-\(M\) 序列，并检查最终层是否真正形成单实例单峰响应；CPN 则可作为困难样本召回分支，但其输入必须同时包含分类语义和边界回归特征，不能只在最终 logits 后增加普通注意力。

Harness 必须设置四个论文特定控制组：无中心度、无 NMS 的 one-to-one FCOS；基线＋C2F；基线＋CPN；完整 C2F＋CPN。统一在 CrowdHuman train/val 上比较 mMR、AP、Recall，并额外比较 C2F 的 \(\{4\}\)、\(\{9\}\)、\(\{16\}\)、\(\{16,9\}\)、\(\{9,4\}\)、\(\{16,4\}\)、\(\{16,9,4\}\)。若完整模型不能同时达到 44.9 mMR、91.0 AP、97.7 Recall，或其 mMR 不优于两个单模块控制组，则判定复现失败。

## 优点

优势在于训练与推理均无需 NMS，保持卷积式密集检测器的部署友好性；C2F只在训练阶段提供多级监督，推理仅使用末级输出。方法对重遮挡、小目标及拥挤场景均有明确收益，并且增大骨干带来的提升很小，支持其“轻量机制优先于堆算力”的结论。

## 局限

局限在于 F2 的最大响应传播天然可能扩散噪声，完整效果依赖 F3 修正；实验没有给出端侧延迟、显存、参数量或具体硬件吞吐，因此“可部署”主要由 CNN 架构和无后处理推断，而非真实设备基准。方法也依赖逐阶段 \(M_i\) 配置，不同数据密度下是否仍以 \(\{16,4\}\) 最优尚未验证。

## 评分

- 论文链接：https://openaccess.thecvf.com/content/CVPR2023/html/Song_Optimal_Proposal_Learning_for_Deployable_End-to-End_Pedestrian_Detection_CVPR_2023_paper.html

**???????**: [?Optimal Proposal Learning for Deployable End-to-End Pedestrian Detection?????????????????????????](https://openaccess.thecvf.com/content/CVPR2023/html/Song_Optimal_Proposal_Learning_for_Deployable_End-to-End_Pedestrian_Detection_CVPR_2023_paper.html)
- 官方代码：论文正文未提供作者 GitHub URL，未发现官方代码仓库；可使用上述 CVF 官方论文页面获取论文及相关材料。
