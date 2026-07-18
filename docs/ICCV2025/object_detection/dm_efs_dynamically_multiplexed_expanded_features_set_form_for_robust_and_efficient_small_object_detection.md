---
title: "[论文解读] DM-EFS: Dynamically Multiplexed Expanded Features Set Form for Robust and Efficient Small Object Detection"
description: "原创中文解读 DM-EFS：扩展高分辨率特征集合，并依据图像目标尺寸动态选择必要特征层。"
tags: ["ICCV 2025", "小目标检测", "动态推理", "特征金字塔"]
---

# DM-EFS: Dynamically Multiplexed Expanded Features Set Form for Robust and Efficient Small Object Detection

**论文**：[CVF 官方论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.html)  
**代码**：可核对的正文与补充材料未声明官方实现地址，避免以第三方复现冒充作者仓库  
**发表**：ICCV 2025  
**类别**：Small Object Detection, Dynamic Feature Selection

## 一句话总结

DM-EFS 先把骨干浅层中未使用的高分辨率特征纳入 neck/head，以补足小目标表示，再由控制模块预测当前图像的最小/最大目标尺寸，查询训练期学到的尺寸—特征码本，只激活覆盖该尺寸范围所需的少数特征层。

## 研究背景与问题

在 640×640 等有限输入下，小目标缺少足够像素。提高输入分辨率成本高，超分辨率可能产生伪影，特征模仿又依赖手工设计。一个直接方案是启用骨干更浅、更高分辨率的特征，但 Expanded Features Set（EFS）会让 neck 和 head 持续处理额外尺度，AP 上升同时 FPS 明显下降。论文的核心问题不是“高分辨率特征是否有用”，而是“能否按图像内实际目标尺寸，只计算有用尺度，并让选择规则由检测结果学习而非人工划分”。

## 方法总览

DM-EFS 在基础检测器的 backbone、neck、head 外加入控制模块 `ΦC`，并在 neck/head 放置特征多路器 `ΨN/ΨH`。EFS 模式下所有门信号为 1，扩展后的全部特征参与推理。训练时，每个 head 尺度对不同目标尺寸的正确预测进行投票：预测框与真值 IoU≥0.5 才计为可靠票，跨图像、跨 epoch 累积成尺寸—特征 vote matrix，按每个尺寸得票最高的特征生成 one-hot Size-Features Codebook Ω。控制模块同时从 backbone 特征预测图像中最小与最大目标尺寸。推理时查 Ω 得到两端尺寸对应的 head 层，激活这两层及产生它们所需的 neck 子集，构成 Dynamically Multiplexed EFS。

## 方法详解

码本学习把“哪个尺度检测哪个尺寸”交给真实预测质量。训练增强不断改变目标尺寸，扩大码本覆盖；从未出现的尺寸用最近邻插值补齐。码本只编码目标尺寸与特征分辨率关系，不编码类别或场景。控制模块把 `s_min/s_max` 作为两个单标签多分类问题，以 focal loss 训练，默认权重 λ_szp=0.04，并与分类、框回归、objectness 损失联合优化。

推理阶段先预测 `s_min/s_max`，分别查询最可靠 head 特征索引 `k_min/k_max`。Head 只开启对应端点层；neck 从 `k_max` 向高分辨率方向保留生成所需 head 特征的最小连续子集。若图像同时含极小与大目标，两端层共同激活；若尺寸范围窄，则大量特征被跳过。最大特征数设为 4，额外控制开销换来接近基础模型的速度。

## 实验与证据

实验统一 640×640，覆盖 SODA-D、VisDrone 和低光 DarkFace。SODA-D 上 YOLOv7 为 30.52 AP、62.20 AP50，DM-EFS 为 34.91 AP、64.47 AP50；极微小 APeS 从 14.51 提到 17.30。VisDrone 上 AP 27.60→29.71，AP50 47.81→51.80。DarkFace 上 AP 28.91→29.30，AP50 64.31→69.30。速度方面，SODA-D 为 38.89 FPS，接近 YOLOv7 的 40.91，并快于 ESOD 30.23 与 CFINet 28.41。关键消融显示 EFS 将 VisDrone AP50 从 47.81 提至 52.01，但 FPS 从 39.71 降到 32.14；加入 DFM 后 AP50 保持 51.80，FPS 回升到 38.24。VisDrone 与 DarkFace 互换码本仅小幅下降，支持码本主要学习尺寸关系。λ_szp 过小会预测尺寸不准，过大则挤压检测损失，0.04 最平衡。

## 对 YOLO-Agent 的启发

DM-EFS 可作为动态 neck 搜索候选，尤其适合输入尺寸受限、目标尺度分布随图像变化的任务。Agent 需要同时修改训练统计、控制头和部署图，不能只复制一个门控层。

**机制特定 Harness**：**对照组**在 640×640、同一硬件与同一训练预算下比较 Base、始终展开全部路径的 EFS、随机码本门控、固定启用最高分辨率层以及完整 DM-EFS。**指标**按最小目标、最大目标、尺度跨度、拥挤度、旋转角和低光条件分桶，汇报 AP、AP50、AP75、APeS、APrS、FPS、实际激活层数、逐层耗时，并关联 DFM 的 `s_min/s_max` 预测误差与漏检；另以交换码本和“极小+大目标混合”图检验选择逻辑。**失败判断**以动态执行的准确率—速度闭环为准：DM-EFS 相对全量 EFS 的实测 FPS 提升不足 12%，或 AP 下降超过 0.4，或交换码本后 AP 波动小于 0.2 而激活路径明显改变，任何一项都表明动态多路没有产生有效选择。

## 优点

- 将小目标精度与推理效率的矛盾拆成 EFS 和 DFM 两个可验证部件。
- 码本由检测正确性投票得到，避免手工指定尺度边界。
- 在驾驶、无人机和低光人脸三类数据上均有收益，并可更换基础检测器。

## 局限

- 控制模块依赖准确预测图像尺寸范围，极端漏估会直接关闭必要特征层。
- 码本使用最小/最大尺寸概括整幅图，无法完整表达多峰尺度分布。
- 稀疏分支在不同推理引擎上的真实加速可能与 FLOPs 估计不一致。

## 评分

**8.8/10**。设计简洁且速度消融有说服力，适合工程化；主要风险是动态分支的部署支持和尺寸范围预测的失效传播。
