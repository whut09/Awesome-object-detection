---
title: "Label, Verify, Correct: A Simple Few Shot Object Detection Method"
description: "用候选挖掘、DINO-kNN 验证与级联框校正，把少样本检测转化为高精度伪标注再训练。"
tags:
  - CVPR 2022
  - few-shot object detection
  - pseudo-labeling
  - DINO
  - box correction
---

# Label, Verify, Correct: A Simple Few Shot Object Detection Method

- 论文：[CVPR 2022 官方页面](https://openaccess.thecvf.com/content/CVPR2022/html/Kaul_Label_Verify_Correct_A_Simple_Few_Shot_Object_Detection_Method_CVPR_2022_paper.html)
- 代码与模型：[LVC 官方项目页](https://www.robots.ox.ac.uk/~vgg/research/lvc/)

## 一句话总结

LVC 的要点不是发明新的检测骨干，而是把少样本检测器产生的高分候选依次“扩充、验类、修框”，再用清洗后的新类伪标注与基类真值端到端重训 Faster R-CNN。

## 研究背景与问题

少样本目标检测给基类完整标注，却只为每个新类保留 K 个实例。基类训练阶段，未标注的新类物体被当成背景，作者称之为 `supervision collapse`：一方面 RPN 不愿为新类提案，另一方面编码器与 RoI 特征被训练成忽略新类。实验证明这不只是分类头的问题。COCO 30-shot 中，基类 RPN 用 100 个 proposal 时新类平均召回仅 `49.7`，微调 RPN 后为 `71.0`；即便给分类头使用全部类别标注，nAP 也只有 `18.3`，远低于完整训练 Faster R-CNN 的 `43.5`。

## 方法总览

流程按三个动词展开。`Candidate Sourcing` 用完成 Novel Training 的检测器扫描训练集，保留新类置信度 `q>0.8` 的候选；`Label Verification` 用 K-shot 真值裁剪建立 DINO ViT-S/8 CLS 特征的余弦 kNN 分类器，仅接受“检测器类别与 kNN 类别一致”的框；`Box Correction` 把通过验证的 RoI 特征依次送入三个类无关回归器，逐级改善坐标。最终，校正候选成为新类伪标注，与基类完整标注合并，所有检测模块重新端到端训练。

## 方法详解

作者先加强 Novel Training，而非沿用 TFA 只更新最终分类层。Faster R-CNN/FPN 的 `Φ_RPN`、`Φ_ROI` 和 `Φ_CLS` 都在基类采样与 K-shot 新类上微调，编码器保持固定；ColorJitter、RandomCrop、COCO 上的 Mosaic 以及 RoI Dropout 用来降低小样本过拟合。四卡 batch size 为 16，卷积骨干使用 SGD、momentum `0.9`，Transformer 骨干改用 AdamW。

验证器与检测器故意保持独立：候选框先裁剪、缩放，再输入自监督 DINO；kNN 邻居数由 `k=min(floor(K/3)+1,10)` 决定，因此 K=`1/2/3/5/10/30` 时 k=`1/1/2/2/4/10`。这种“双模型同意”策略牺牲召回换取伪标签类别精度。

框校正器借鉴 Cascade R-CNN，但只承担类无关回归。训练阶段将 RPN proposal 按 IoU>`0.3`、`0.5`、`0.7` 分给三只回归头；推理时同一个候选连续经过三头。对未通过验证的新类检测，最终重训并不直接当背景，而被标为 ignore region，避免残留真物体再次触发 supervision collapse。

## 实验与证据

COCO 使用 60 个基类、20 个 VOC 新类，评估 10/30-shot 的 nAP、nAP50、nAP75；VOC 使用三套 15+5 划分和 1/2/3/5/10-shot，报告 nAP50。30-shot、ResNet-50 的逐步消融非常清楚：增强后的基线为 `16.6/30.9/15.8`；仅加入 Candidate Sourcing 变为 `18.4/34.0/18.2`；再做 Label Verification 达 `21.8/40.3/21.1`；最后 Box Correction 达 `25.5/42.0/27.3`。校框阶段 nAP75 单独增加 6.2 点，而类别数量与分布不变，因而收益可归因于坐标质量。

完整比较中，ResNet-50 的 10-shot/30-shot nAP 从基线 `11.4/16.6` 提到 `17.6/25.5`，30-shot nAP50=`42.0`、nAP75=`27.3`。Swin-S 配置进一步达到 30-shot nAP=`26.8`。VOC Split 3 的 ResNet-101 在 1/2/3/5/10-shot 分别得到 `48.4/52.7/55.0/59.6/59.6` nAP50；但论文也指出 DeFRCN 某些设置更高，且它转移后不再保持基类检测能力，目标不同。

基线构建本身也有实证价值。TFA 式只调分类层时，单独加入 ColorJitter、RandomCrop 或 Mosaic 几乎没有改变 nAP，三者合用也只从 `13.0` 到 `13.8`；当 RPN 与 RoI 模块一并开放后，同样增强把结果推到 `16.2`，再加 RoI Dropout 为 `16.6`。小目标 nAPs 从 TFA 的 `5.5` 到完整基线的 `7.0`，说明增强只有在能更新空间提案和区域表征时才真正生效。

伪标注闭环没有以新类为代价遗忘旧类。初始基类模型的 bAP 为 `36.1`，少样本基线因联合微调降到 `29.5`；候选挖掘、验类、校框后逐步恢复至 `31.7/31.6/33.3`。理想完整标注模型为 `36.1`，所以 LVC 缩小了遗忘但没有完全消除。框校正示例中，输入 IoU 约 `0.27–0.33` 的候选能被修到 `0.94–0.97`；同时，基线到最终方法的 nAP50 增幅大于 nAP，表明新增实例扩大召回，而三级回归进一步负责严格定位。复现时应分别记录两种收益，不能只看一个总 AP 就宣称所有模块有效。

数据协议必须遵循固定的 TFA 样本列表与三套 VOC 新类划分，不能为某次运行重新抽取有利的 K-shot 图像。候选扫描范围应是原训练图像集合，且基类真值继续保留；若把外部无标签库混入，只能作为额外实验单列。DINO 只负责裁剪区域的类别复核，不得读取测试标签或用 Hungarian 结果反向调整候选，否则会把评估对齐误当成训练监督。

每轮伪标注必须保存版本、来源和过滤原因，确保错误能够追溯。

## 对 YOLO-Agent 的启发

该论文更适合作为 Agent 的离线数据治理动作：让当前 YOLO 在未穷举标注的训练图像上提出新类框，再调用冻结的自监督视觉编码器做类别复核，最后用专门的 box refiner 改坐标。它解决的是“数据缺口”，不应被包装成推理期插件。

**对照组**设四列且共用同一 YOLO 权重初始化、K-shot 列表、基类/新类采样、epoch 和增强：①强少样本微调；②①+置信度大于 0.8 的直接伪标注；③②+DINO-kNN 一致性过滤；④③+IoU 0.3/0.5/0.7 三级类无关校框及 ignore regions。每组保存完全相同的评估快照，避免只挑有利轮次。**指标**包括新类 AP50:95、AP50、AP75、基类 AP、每类新增伪框数、伪框类别 precision、IoU≥0.5/0.75 的框精度和重训 GPU-hours。**失败判断**是：③相对②不能同时提高伪标签 precision 与新类 AP；④相对③的 AP75 增幅小于 2 点；基类 AP 比初始基类模型下降超过 2 点；或某新类伪框被单一视觉共现模式垄断超过 70%。触发任一条，Agent 必须回退到人工复核队列，而不是自动扩库。

## 优点

- 三阶段每一步都有可测的伪标签质量目标，错误来源容易定位。
- DINO 验类不依赖检测器自身特征，降低确认偏差。
- 基类真值与新类伪标注联合重训，兼顾扩类和旧类保持。
- 普通 Faster R-CNN 即可得到强结果，方法不绑定特殊元学习结构。

## 局限

- 第一步候选召回构成上限；论文定性分析也承认基线漏检会限制伪标注多样性。
- 固定 `q>0.8` 和“两个分类器必须一致”可能丢掉长尾、遮挡及困难新类。
- DINO 特征与 kNN 在视觉近邻类别上仍会混淆，且增加离线计算与存储。
- 三级回归器依赖足够好的 RPN proposal；极差候选和完全漏框无法被修复。

## 评分

- 创新性：**8.0/10**
- 实验完整度：**9.0/10**
- YOLO 数据闭环价值：**8.5/10**
- 综合：**8.5/10**
