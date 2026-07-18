---

# Robust Small-Scale Pedestrian Detection With Cued Recall via Memory Learning
title: "Robust Small-Scale Pedestrian Detection With Cued Recall via Memory Learning"
description: "ICCV 2021 论文笔记：以大尺度行人回忆记忆和大尺度嵌入学习，通过线索回忆增强小尺度行人检测。"
tags:
  - ICCV2021
  - 小尺度行人检测
  - 记忆学习
  - 多光谱检测
  - 目标检测
---

- **论文链接**：[CVF 官方论文页面](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Robust_Small-Scale_Pedestrian_Detection_With_Cued_Recall_via_Memory_Learning_ICCV_2021_paper.html)

**???????**: [?Robust Small-Scale Pedestrian Detection With Cued Recall via Memory Learning?????????????????????????](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_Robust_Small-Scale_Pedestrian_Detection_With_Cued_Recall_via_Memory_Learning_ICCV_2021_paper.html)
- **官方代码**：论文正文未提供作者 GitHub URL，未发现官方代码仓库；可查阅上述官方论文页面及其补充材料。

## 一句话总结

该方法不是对小目标特征做超分辨率，而是引入 **Large-scale Pedestrian Recalling Memory（LPR Memory）**，用小尺度 RoI 中残存的头部、躯干等线索查询已经记忆的大尺度行人外观。输入图像先经 VGG16 或 ResNet 骨干网络、RPN 与 RoI Align 得到 RoI 特征 \(F\)，再按预测框高度分成小尺度特征 \(F_S\) 和大尺度特征 \(F_L\)。仅 \(F_S\) 进入 LPR Memory，得到回忆特征 \(F_S^M\)；随后将 \(F_S^M\) 与原始 \(F_S\) 拼接，经 \(1\times1\) 卷积形成精炼特征 \(F_S^R\)，最后与 \(F_L\) 一起送入两阶段检测头完成分类和定位。

LPR Memory 是含 \(L\) 对槽位的键值记忆 \(M=\{M_K,M_V\}\)。每个小尺度 RoI 被展平后，与全部键槽 \(M_K\) 计算余弦相似度，再经 Softmax 得到寻址向量；该向量对值槽 \(M_V\) 加权求和，恢复与当前线索最相关的大尺度行人特征。训练时另建 **Large-scale Pedestrian Exemplar Set**：从训练图像裁剪边界框高度超过阈值 \(H_L\) 的大尺度行人，每轮随机抽取 \(N_E=32\) 个样本，通过共享骨干生成 exemplar 特征，用于把清晰、对齐的行人外观写入值记忆。

实验覆盖 KAIST Multispectral Pedestrian Detection Dataset 与 CVC-14，并分别测试热红外、可见光输入。基础检测器为 Faster R-CNN＋VGG16，指标是 FPPI 区间 \([10^{-2},10^0]\) 上的平均漏检率 MR，数值越低越好。KAIST 热红外上，Faster R-CNN 基线的 All MR 为 26.70，完整方法降至 19.16；Far MR 从 65.87 降至 51.40。关键消融显示，仅使用记忆损失或回忆损失时 All MR 分别为 22.88、23.16，而二者联合达到 19.16，证明性能并非来自单纯增加一个记忆模块。

## 研究背景与问题

小尺度行人与背景难以分离的根因是像素不足导致外观信息缺失。作者借鉴认知心理学中的“线索回忆”：人看到模糊远处目标时，会用局部线索关联记忆中的清晰对象，而非凭当前图像重建所有细节。因此，目标不是生成高分辨率图像，而是在特征空间中把小尺度行人拉向可靠的大尺度行人表征，同时保持其与背景的距离。

这一设计还避开了直接使用大尺度 RoI 作为教师的问题。普通 RoI 可能包含偏移、背景和不完整人体，而 exemplar set 由真实标注框裁剪，具有更稳定的空间对齐，并覆盖不同姿态与外观，因而更适合承担长期记忆的监督信号。

## 方法总览

**大尺度行人记忆损失 \(L_{\mathrm{Mem}}^M\)** 负责“写入”。Exemplar 特征与值槽计算余弦相似度和 Softmax 权重，再从 \(M_V\) 重构特征；重构结果与原 exemplar 特征之间使用均方误差，使值槽逐渐编码多样的大尺度行人外观。

**大尺度行人回忆损失 \(L_{\mathrm{Rec}}^M\)** 负责“对齐”。检测头分别生成 exemplar、小尺度精炼行人、大尺度行人及所有 RoI 的潜在特征，并计算它们相对于全部 RoI 的相似度分布。作者以 exemplar 的平均相似度分布为目标，通过 KL 散度约束小尺度和大尺度行人分布，使二者靠近 exemplar、远离背景。

总损失为  
\[
L_{\mathrm{Total}}=L_{\mathrm{RPN}}+L_{\mathrm{CLS}}+L_{\mathrm{LOC}}
+\lambda_1L_{\mathrm{Mem}}^M+\lambda_2L_{\mathrm{Rec}}^M
\]
实验中 \(\lambda_1=\lambda_2=1\)，默认 \(L=100\)、RoI 数量 \(N=256\)，KAIST 的大尺度分界 \(H_L=115\)。

## 方法详解

KAIST 含 95,328 对可见光—热红外图像、103,128 个标注和 1,182 名不同的行人，测试采用 2,252 张图像。尺度按高度划分为 Far：小于 45，Medium：45 至 115，Near：至少 115。CVC-14 使用 7,085 张训练图像和 1,433 张测试图像；因其中目标整体较大，Far 阈值改为 60。

KAIST 热红外 All MR 为 19.16，优于 TC Det 的 27.11、SML 的 22.51及基线的 26.70；白天和夜间分别取得 24.70 与 8.26。可见光条件下，All MR 从基线 30.66 降至 25.16，Far MR 从 77.82 降至 68.92。CVC-14 热红外上，All MR 从 27.42 降至 23.00，Medium 与 Far 分别由 28.57、72.83 降至 23.07、64.98。

## 实验与证据

记忆槽数量并非越多越好：\(L=25,50,100,200\) 时 All MR 分别为 23.68、20.23、19.16、20.46。200 个槽虽然令 Near MR 达到 2.79，却使 Medium 和 Far 退化，说明容量过大可能导致寻址分散或槽位冗余。

方法也能迁移到 FPN＋ResNet-50：基线 All MR 为 16.56，加入 LPR Memory 后为 13.91，Far MR 从 53.46 降至 46.43。若在 KAIST 上训练却改用 CVC-14 exemplar，All MR 仍为 22.68，优于 26.70 的基线，但弱于同域 exemplar 的 19.16，显示记忆具有跨数据集价值，同时仍受域差异影响。

## 对 YOLO-Agent 的启发

可把 LPR 思路移植到 YOLO-Agent 的多尺度检测分支：在检测头前选取负责小目标的高分辨率特征或正样本特征作为查询，以清晰大目标实例建立键值原型库，再把读取结果与原小目标特征融合。关键是保持“只增强小尺度分支”，避免清晰大目标被不必要地改写。

训练控制组必须至少包含：原始 YOLO-Agent、仅增加等参数卷积、仅有记忆模块但无 \(L_{\mathrm{Mem}}\)、仅有 \(L_{\mathrm{Mem}}\)、仅有 \(L_{\mathrm{Rec}}\)、完整双损失，以及不同槽数和同域／跨域 exemplar。除总体 mAP 外，应单独报告 AP\(_S\)、小目标召回率、漏检率，以及相同 FPPI 下的 MR，防止总体指标掩盖远距离行人退化。

具体失败判据：若完整模型在三次独立训练中的 AP\(_S\) 或 Far MR 未稳定优于等参数控制组，或小尺度召回提高却令误检率、Medium/Near 指标明显恶化，则不能声称线索回忆有效；若随机 exemplar 与真实大尺度 exemplar 表现相同，也说明收益可能仅来自额外容量。

## 优点

优点是模块逻辑清楚、可插入不同两阶段检测器，并同时适用于热红外与可见光。它不依赖生成清晰图像，而是直接修正判别特征；参数由 141.27M 增至 146.42M，其中 LPR Memory 为 5.02M，额外开销约 3.6%。

## 局限

局限在于 exemplar set 依赖大尺度标注实例，训练流程需要额外骨干前向与相似度计算；尺度划分和 \(H_L\) 也具有数据集依赖性。论文主要研究两阶段检测器与行人类别，尚未证明在密集多类别、小目标极少或跨域差异很大的场景中同样可靠。

## 评分

核心贡献是把小尺度检测重新表述为“由模糊线索读取清晰类别记忆”。LPR Memory 通过键寻址和值读取补充缺失外观，\(L_{\mathrm{Mem}}^M\) 保证记忆内容来自对齐的大尺度实例，\(L_{\mathrm{Rec}}^M\) 则让不同尺度行人在关系空间中共享 exemplar 结构。其最有价值的结论不是简单增加记忆容量，而是以明确的大尺度实例监督记忆内容，并用尺度级控制实验验证小尺度漏检确实下降。
