---
title: Neptune-X - Active X-to-Maritime Generation for Universal Maritime Object Detection
publication: NeurIPS
year: 2025
description: "聚焦 Neptune-X 以双向目标—水面交互生成海事样本，并依据类别、视角、地点和环境难度主动挑选高价值合成数据。"
paper: https://proceedings.neurips.cc/paper_files/paper/2025/hash/d7b0baefb84b8ddf6fbf6ec0f5d4fda3-Abstract-Conference.html
code: https://github.com/gy65896/Neptune-X
tags: [海事目标检测, 生成式增强, 主动采样, 条件扩散, 数据中心AI]
---

# Neptune-X: Active X-to-Maritime Generation for Universal Maritime Object Detection

## 一句话总结

Neptune-X 先用 BiOW-Attn 显式建模目标与水面的双向边界作用，生成带框、掩码和语义属性的海事图像，再以 Attribute-correlated Active Sampling 从生成池挑出最能弥补检测器困难属性的样本。

## 研究背景与问题

海事检测数据来自岸基、船载和无人机平台，采集与框标注昂贵，且船、浮标、人员等类别在视角、地点和天气上严重不均衡。通用 Stable Diffusion 或布局生成器容易出现船体悬浮、目标缺失、水面纹理穿透目标等错误；即使生成质量尚可，随机把大量合成图加入训练也会造成算力浪费，不能保证改善困难切片。

论文把问题拆为“生成什么”和“选择什么”。前者需要文本、目标框/类别、水面描述/掩码共同控制；后者需要依据检测器在类别、视角、地点、成像环境四个属性维度上的真实困难度，优先选择任务价值高而非仅视觉逼真的样本。

## 方法总览

**X-to-Maritime** 以冻结的 SD1.5 为基础，只训练布局条件嵌入器与 **Bidirectional Object-Water Attention（BiOW-Attn）**。原始标签经随机框缩放、翻转和文本属性采样形成条件，生成结果先通过 CLIP 语义一致性与 ResNet 框内类别准确性过滤。随后在真实小规模训练集上预训练检测器，计算各属性的 **ATDF**，再由 **Attribute-dependent Active Sampling（AAS）** 将属性困难先验与每张合成图的检测错误结合，排序选出 top-k 样本，与真实集共同微调检测器。

## 方法详解

**BiOW-Attn 的数据流**是：每个目标的类别文本与 Fourier 框坐标经 MLP 得到 object token；水面描述与水面最小外接框得到 water token。Object Cross-Attention 和 Water Cross-Attention 分别增强扩散特征，输出再由目标并集掩码、水面掩码与各自可学习 null token 限定空间。随后 Wat2ObjCA 用水面上下文更新目标特征，Obj2WatCA 反向用目标状态修正水面，最终经 FFN 回写 U-Net，使船体—水线边界更符合物理关系。

**ATDF 的数据流**从真实训练/验证集的预测框开始。每个框以分类是否正确和 IoU 共同形成准确度，再按类别、shore/ship/aerial 视角、sea/river/harbor/lake 地点、六类天气/时段聚合为属性困难度；EMA 持续更新，属性缺席时衰减动量以加快稀有属性刷新，最后在同一维度 softmax 为困难概率。

**AAS 的样本价值**不是单纯不确定性。对于一张带 N 个目标的合成图，算法把视角、地点、环境 ATDF 的乘积，与各目标类别 ATDF 及 `1-accuracy` 的平均结合成难度分数 `d`；生成池按 `d` 排序取 top-k。这样“检测器当前不会、训练集又稀缺”的雾天/夜间/特殊视角样本会排在容易的常规海面图之前。

## 实验与证据

作者构建 **MGD**：11,900 张图，来源包括 MaSTr1325、USVInland、SMD、SeaShips、Seagull、LaRS 等，按 7,140/2,380/2,380 划分；覆盖 5 类目标、3 视角、4 地点、6 成像环境。生成对比包括 SD1.5、LayoutDiff、GLIGEN、InstDiff、RC-L2I。Neptune-X 达 **FID 18.05、CAS 79.34、YOLO Score 17.08/39.14/13.52**，均优于主要条件生成基线。

加入 10,000 张生成数据后，YOLOv10 mAP 从 39.99 到 **43.62**，YOLOv11 从 41.29 到 **44.43**，YOLOv12 从 39.06 到 **42.91**；Grounding DINO 微调后再加生成数据由 65.03 到 68.04。BiOW 消融从 ObjCA 的 FID 21.44、YOLO mAP 10.69，逐步加入 WatCA、Wat2ObjCA、Obj2WatCA，完整模块达到 18.05 与 17.08。AAS 选 5,000 张即得 43.11 mAP，接近随机 10,000 张的 43.31；AAS 10,000 张为 43.62，并在 10k–20k 后出现收益饱和。

## 对 YOLO-Agent 的启发

1. 数据生成 agent 应同时优化视觉质量、布局正确性和下游增益，不能只看 FID。
2. 将错误按类别、视角、地点、天气维护为动态困难先验，可直接驱动生成提示词与采样预算。
3. 对水面、地面、天空等强关系场景，可设计“对象—载体介质”双向交互模块，而非只给目标框。

**YOLO-Agent Harness（Neptune-X）**：将生成器和样本选择器拆成两条审计链。生成侧**对照组**为 ObjCA、ObjCA+WatCA、单向 Wat2ObjCA、单向 Obj2WatCA 与完整 BiOW；在固定候选池上，采样侧比较随机、预测不确定性、ATDF-only 和完整 Attribute-correlated Active Sampling（AAS），预算统一检查 5k/10k/20k。验收**指标**同时包含 FID、CAS、YOLO Score、真实 MGD 测试集 mAP/mAP50、五类别×三视角×四地点×六环境的最差切片、船体—水面边界错误率与每千张合成图带来的 AP 增益。若视觉生成分数变好而真实检测不升、BiOW 没有降低水面穿透或悬浮布局错误、困难属性收益靠容易属性退化换取、AAS 5k 达不到随机 10k，或晴天眩光切片仍被平均值掩盖，则触发**失败判断**。

## 优点

- 生成与选择形成下游任务闭环，而非无差别扩充数据。
- BiOW-Attn 贴合海事场景独有的目标—水面关系。
- MGD 属性覆盖细，生成质量与检测收益均有量化证据。

## 局限

- ATDF 依赖预定义离散属性，难以描述连续能见度、浪高或传感器高度。
- 生成器训练约 100 小时，过滤和检测器重训仍有较高成本。
- 合成数据饱和明显，真实域外海况和罕见船型的泛化仍需验证。

## 评分

- 创新性：9/10
- 技术完整性：9/10
- 实验说服力：8.5/10
- 工程可迁移性：8/10
- 对 YOLO-Agent 价值：9/10
