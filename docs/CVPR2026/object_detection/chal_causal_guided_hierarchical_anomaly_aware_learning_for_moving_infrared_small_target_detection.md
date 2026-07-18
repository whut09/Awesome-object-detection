---
title: "[论文解读] CHAL: Causal-guided Hierarchical Anomaly-aware Learning for Moving Infrared Small Target Detection"
description: "将移动红外小目标改写为背景正常性建模后的异常发现，并以运动验证和因果门控抑制背景混杂。"
tags: ["CVPR 2026", "红外小目标", "视频检测", "异常检测", "因果学习"]
---

# CHAL: Causal-guided Hierarchical Anomaly-aware Learning for Moving Infrared Small Target Detection

**论文**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.html)  
**代码**：[UESTC-nnLab/CHAL](https://github.com/UESTC-nnLab/CHAL)

## 一句话总结

CHAL 反转“直接学微弱目标”的常规范式：先用 Spatio-Temporal Neural Fields（SNF）重建连续背景，把与背景不一致处当作外观异常，再由 Hierarchical Anomaly-Aware Learning（HAL）用五帧运动一致性验证，最后通过 Causal Relation Guiding（CRG）抑制云边、亮点和噪声造成的伪相关。

## 研究背景与问题

移动红外小目标通常小而暗，缺乏纹理和稳定形状。目标中心式网络直接从复杂背景中寻找目标特征，容易把更显著的云边、传感器噪声或高亮背景当成目标。论文用结构因果模型描述这一问题：真实目标 `T` 应通过特征 `F` 影响检测 `Y`，但背景混杂变量 `Z` 同时污染特征并造成误报，形成 `F←Z→Y` 的后门路径。作者因此把任务改写为异常发现：背景是可学习的正常模式，真实移动目标只是背景演化中的异常；难点从“增强弱目标”变为“区分真实异常与背景伪异常”。

## 方法总览

五帧红外视频逐帧经 CSPDarknet 与 FPN 得到多尺度特征，早期融合后形成时空体。SNF 将其拆成静态场景编码与动态运动编码，并把三维坐标映射到 Fourier 位置特征，神经场根据场景编码和位置查询生成每帧预测背景 `B`。HAL 计算真实特征 `F` 与反事实背景 `B` 的余弦差异得到外观异常 `Aa`，再用时空 Swin Transformer 验证局部三维窗口内的运动一致性，解码并按学习帧权重汇总为最终异常 `Af`。CRG 把 `Af` 作为混杂代理，进行阈值分层、异常增强、因果门控和特征细化，输出 `Ff` 给检测头。

## 方法详解

### Spatio-Temporal Neural Fields

多尺度特征先上采样、求和并经过空间/通道注意，堆叠为时空体 `V`。场景编码器用 3D 卷积平滑后，经池化、展平和 MLP 提取不变外观 `Zs`；并行运动编码器直接从 `V` 提取动态表示 `Zd`，两者拼成背景先验 `Z`。每个关键帧像素对应三维坐标 `(x,y,αt)`，再用多频率正余弦映射产生查询 `Q`。背景神经场以 `Z` 和 `Q` 预测连续背景，而不是简单帧平均，因此可以拟合云层边缘等复杂正常变化。

### Hierarchical Anomaly-Aware Learning

第一层异常由 `1-cos(Fk,Bk)` 经三层残差卷积放大得到，负责发现所有外观偏离，包括真实目标和伪异常。随后各帧异常投影到高维子空间，加入位置编码，交给多层时空 Swin Transformer 在移位三维窗口内建模；只有在时间上呈现一致运动的候选才应被保留。解码后的异常序列通过 softmax 学习每帧权重，汇总为 `Af`。这一“外观发现—运动验证”层级流程比单帧显著性更能排除静态亮点。

### Causal Relation Guiding

CRG 借鉴后门调整，但用可微门控近似不可枚举的连续背景混杂。归一化 `Af` 后，以阈值分为高、低异常两层，高异常乘可学习放大因子，低异常乘抑制因子，形成分层分数 `H`。观测特征先经外观注意保留候选，再由 `H` 门控和卷积细化，使特征流更接近 `T→F→Y`，同时阻断背景 `Z` 对检测的直接影响。

SNF 的背景预测必须避免两种相反退化：若容量太弱，只生成平滑平均图，云边仍会作为异常；若容量太强，把微弱移动目标也重建进背景，后续差分就失去目标。论文用三维可视化说明目标峰值被从预测背景中排除，但没有给出独立背景重建数据集，因此复现时应额外统计目标区域与非目标区域的重建误差比例。

HAL 的运动验证默认五帧内目标具有可辨别的一致轨迹。对悬停目标、突然加速、镜头抖动或帧间配准失败，时空一致性可能反而抑制真目标。CRG 中的阈值和高低异常缩放也可能在数据域变化时失准，因此跨数据集测试应冻结这些超参数，避免在每套测试集上重新调节后再宣称因果泛化。

还应单独报告首帧冷启动表现，因为在线系统无法始终预先获得完整五帧窗口。

## 实验与证据

实验在 DAUB-H、NUDT-MIRSDT、IRDST-R 三个红外数据集上进行，所有多帧方法统一使用五帧；比较对象包括 ISNet、DNANet、PConv、ST-Trans、SSTNet、Tridos、DTUM 和 ADSUNet。CHAL 在 DAUB-H 达到 54.28 mAP50、74.15 F1，在 NUDT-MIRSDT 达到 75.25 mAP50、87.41 F1，在 IRDST-R 达到 71.37 mAP50、84.76 F1；模型为 15.69M 参数、137.04 GFLOPs，在 RTX 4090 上 12.96 FPS。可见光 RsCarData 上也报告 57.37 mAP50、76.14 F1，但这只是一项跨模态适应性证据。

关键消融链条非常清楚。DAUB-H 基线仅 23.48 mAP50/46.36 F1；完整 SNF 后为 43.76/60.90；加入 HAL 为 48.63/66.68；再加入 CRG 达到 54.28/74.15。NUDT-MIRSDT 对应从 50.30/70.56 提升到 63.50/76.58、68.49/82.73、75.25/87.41。CRG 还优于 Self-Attention 与 CBAM：在 DAUB-H 比 CBAM 高 3.83 mAP50，在 NUDT-MIRSDT 高 5.07，同时参数少于自注意版本。

## 对 YOLO-Agent 的启发

- 对视频 YOLO，可先实现“预测背景—残差候选—运动验证”的外挂分支，把候选异常作为检测头门控，而非直接替换全部主干。
- **Harness 对照组**：单帧 YOLO、五帧堆叠 YOLO、仅背景重建、背景重建+外观异常、再加运动验证、完整因果门控；CRG 需与 CBAM、自注意和普通 sigmoid 门控对比。
- **Harness 指标**：mAP50、F1、每分钟虚警、弱对比目标召回、云边/静态亮点误检率、遮挡目标召回、五帧延迟、GFLOPs 和背景重建误差。
- **失败判断**：若 SNF 同时重建掉真实目标导致弱目标召回下降，或 HAL 对静态亮点误检无改善，异常范式失败；若 CRG 不优于同参数普通注意力，不能把增益归因于因果设计；若帧率低于任务要求，即使 AP 最高也不适合部署。

## 优点

- 从背景正常性出发，绕开红外目标特征本身极弱的矛盾。
- SNF、HAL、CRG 对应背景建模、异常验证和混杂抑制，逻辑闭环且消融充分。
- 同时报告精度、F1、参数、GFLOPs、FPS及可见光迁移结果。

## 局限

- 五帧神经场和时空 Transformer 带来 137.04 GFLOPs，论文也承认参数与计算偏高。
- “因果”通过结构假设与门控近似实现，并非随机干预或可识别性证明。
- 对镜头剧烈运动、背景整体突变和目标长期静止的适用性尚未被系统验证。

## 评分

- **创新性**：★★★★★
- **实验可信度**：★★★★★
- **YOLO-Agent 参考价值**：★★★★☆
