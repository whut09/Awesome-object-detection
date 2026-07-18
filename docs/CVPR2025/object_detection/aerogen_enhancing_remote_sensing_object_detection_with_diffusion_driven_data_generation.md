---
title: "[论文解读] AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation"
description: "面向水平框与旋转框布局的遥感扩散生成器，以及从布局采样、双重过滤到检测增强的完整流水线。"
tags: ["CVPR 2025", "遥感检测", "扩散模型", "合成数据", "布局控制"]
---

# AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_AeroGen_Enhancing_Remote_Sensing_Object_Detection_with_Diffusion-Driven_Data_Generation_CVPR_2025_paper.html)  
**任务**: 遥感布局生成 / 检测数据增强

## 一句话总结

AeroGen 用统一八坐标布局编码、Dual Cross-Attention 与 Layout Mask Attention 精确生成水平框或旋转框遥感目标，再用布局 DDPM、统计过滤和语义/位置双过滤组成可扩展的检测数据增强流水线。

## 研究背景与问题

遥感目标尺寸小、方向任意且分布密集，通用文生图模型容易生成“像遥感图”却不服从框位置，随机拼接布局又会产生重叠冲突和不符合真实尺度分布的标签。仅比较 FID 也无法证明合成图能训练检测器。论文把问题拆成三个环节：框条件能否准确进入扩散 UNet，布局是否符合真实标注统计，生成图的类别语义和框内目标是否一致。

## 方法总览

每个 HBB 或 OBB 统一表示为四顶点八坐标，经 Fourier 编码后与冻结 CLIP 类别向量拼接成 layout token。UNet 同时接收全局文本和局部布局：Dual Cross-Attention（DCA）分别处理全局与局部条件，Layout Mask Attention（LMA）把类别嵌入限制注入对应框掩码。数据侧另训练 DDPM 生成布局矩阵，经高斯统计筛选和几何增强形成布局池；AeroGen 据此合成图像，再用 CLIP 与 ResNet101 分类器筛除语义或位置不一致样本。

## 方法详解

布局嵌入把水平框也展开为四顶点，因此同一网络无需为 DIOR 与 DIOR-R 切换编码格式。局部控制不是只把 token 输入交叉注意力：每个框先栅格化为 0/1 mask，类别条件在每一步去噪时只向该区域注入，强化小目标位置约束；DCA 同时保留文本描述的全局场景，避免局部对象正确但背景语义破裂。

生成流水线把真实标注构造成 `H×W×N` 布局矩阵，目标区域为 1、其余为 -1，使用 DDPM 拟合多类别空间分布。采样框按面积等属性的标准分数阈值过滤，再执行缩放、平移、旋转和翻转。图像生成后，CLIP 负责整体语义一致性，分类器负责框内类别与布局一致性。最终合成图与真实图共同训练 YOLOv8 或 Oriented R-CNN，生成器不进入检测推理图。

## 实验与证据

- 数据覆盖 DIOR/DIOR-R（23,463 图、192,518 目标、20 类）和 HRSC（1,061 图、2,976 目标、19 类）；AeroGen 训练 100 epoch，仅更新 UNet 注意力层与 LMA。
- DIOR 上相同 SD1.5 初始化与训练轮数下，AeroGen 达 `FID 38.57、CAS 76.84、YOLO Score 29.8/54.2/31.6`，优于 GLIGEN 的 `41.31、63.50、25.8/44.4/27.8`；DIOR-R 与 HRSC 的 OBB 条件也优于改造版 GLIGEN。
- 合成量带来稳定增益：DIOR 从 `54.22 mAP` 升至 50k 的 `57.92`；DIOR-R 从 `37.39` 升至 `41.69`；HRSC 从 `63.49` 升至 10k 的 `65.92`。
- LMA 与 DCA 全开时最好；完整流水线为 `41.69 mAP/64.12 mAP50`，去掉布局一致性过滤降至 `37.05/60.03`。HRSC-100 加 5k 合成图也显著优于仅 100 张真实图。

## 对 YOLO-Agent 的启发

- Harness 设置真实数据、随机真实布局生成、DDPM 布局但不过滤、完整 AeroGen、等量 CopyPaste 五组；检测器、总迭代数和真实样本采样次数必须一致。
- 除全局 mAP，按目标面积、旋转角、每图实例数和类别频次报告 AP；增益若只来自大目标或稀疏场景，判定不适合小目标遥感增强。
- 同时记录 FID/CAS/YOLO Score、框内分类正确率和真实测试 AP。生成指标改善但真实 AP 不升，说明过滤阈值或域差失败。
- 做合成量曲线并监控真实/合成比例；高配比退化时抽查重复纹理、框外目标和旋转框顶点错位。

## 优点

- 将布局生成、条件成像和质量过滤闭环连接到真实检测指标。
- 同一坐标接口支持 HBB 与 OBB，小目标区域有显式 mask 注入。
- 消融能定位 DCA、LMA 与两级过滤的独立贡献。

## 局限

- 生成和过滤成本高，筛选器可能继承 CLIP/分类器偏差。
- 主要验证光学遥感图像，跨传感器与跨地区真实性未知。
- 合成规模扩大不等同于覆盖长尾，个别类别仍会下降。

## 评分

- **创新性**: ★★★★☆
- **证据强度**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★☆
