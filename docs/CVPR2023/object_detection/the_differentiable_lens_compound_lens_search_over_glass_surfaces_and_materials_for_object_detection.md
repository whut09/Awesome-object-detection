---
title: "The Differentiable Lens: Compound Lens Search Over Glass Surfaces and Materials for Object Detection"
description: "面向汽车目标检测的可微复合镜头联合设计方法：精确光线追迹、玻璃材料量化搜索、制造约束与检测任务协同优化。"
tags:
  - CVPR 2023
  - 可微光学
  - 复合镜头设计
  - 目标检测
  - 计算成像
---

# The Differentiable Lens: Compound Lens Search Over Glass Surfaces and Materials for Object Detection

## 一句话总结

该工作以精确可微光线追迹生成空间变化的 PSF、相对照度和畸变场，并用 **Quantized Continuous Glass Variables** 同时搜索球面曲率、镜片间距及 Ohara 目录玻璃，使仅含 2—4 个镜片的光学系统直接围绕 RetinaNet 检测 AP 而非传统成像质量完成设计。

## 研究背景与问题

传统镜头通常以 RMS spot size 等光学指标独立设计，随后才连接传感器和检测器；本文则在 BDD100K 汽车场景上，把镜头参数与 ResNet-50 RetinaNet 的参数共同训练。其核心问题不是“恢复清晰图像”，而是允许镜头像差按目标分布重新分配：例如双片镜头的平均光斑由 80.1 μm 恶化到 135.5 μm，BDD100K AP 却从 30.3 提升到 32.0。

此前可微复合光学方法往往只优化有限范围内的表面或间距，或者用针对固定镜头结构训练的代理网络预测 PSF。本文进一步处理三个困难：镜头目标高度非凸；表面相交、全反射和厚度等制造约束会使追迹失败；玻璃材料属于离散目录变量，不能直接由普通 SGD 连续优化。

## 方法总览

完整数据流为：自然场景图像 \(I_S\) → 由镜头变量计算 21 个视场位置上的几何 PSF → 通过空间变化 overlap-add 卷积加入模糊 → 乘以相对照度图 → 按畸变场进行双三次坐标变换 → 缩放为 \(1024\times1024\) → RetinaNet 输出检测框。训练阶段对镜头和检测器反向传播；评估阶段固定参数，并把畸变校正施加到预测框，而不是利用校正后的真值框计算 AP。

## 方法详解

### 可微成像模型

模型在球面界面间交替更新光线坐标，并依据 Snell 定律更新方向余弦。为处理强烈的 pupil aberration，作者增加 **ray-aiming correction**：将入口瞳孔变形成随视场变化的椭圆，使目标光线能够真正通过孔径边缘。

每个视场使用 2048 条瞳孔光线，分布于 32 个带抖动的同心圆；共追迹 15 个波长，即 RGB 每通道 5 个基于 Sony IMX172 量子效率选出的波长。光线落点在 \(260\,\mu m\)、\(65\times65\) 的虚拟网格上，通过高斯核 KDE 而非硬计数形成可微 PSF。

空间变化卷积采用 \(9\times9\) 图像分块、25% 重叠和二维 Hann 窗。相对照度由两条子午光线与一条弧矢光线在 587.6 nm 下近似计算；畸变则比较实际平均像高与近轴参考像高，插值得到径向位移场。

### 镜头与玻璃搜索

镜头变量包括归一化球面曲率 \(c'\)、玻璃及空气间距 \(s'\)，以及每个镜片的玻璃变量 \(g\)。最后一个曲率在每轮训练中解析求解，以维持单位焦距；**paraxial image solve** 计算后焦距 BFL，使像面在优化过程中基本保持合焦。

玻璃候选来自 65 种 Ohara 推荐材料，以 587.6 nm 折射率和 Abbe 数表征。前向传播时，每组连续变量被量化为最近的目录玻璃 \(g_m^*\)，反向传播则使用 gradient straight-through estimator；\(\ell_{\mathrm{GV}}\) 约束连续变量靠近当前材料，减少目录切换造成的性能跳变。

### 目标函数

镜头损失为：

\[
\ell_{\mathrm{lens}}=\ell_{\mathrm{S}}
+100\ell_{\mathrm{RP}}
+100\ell_{\mathrm{RA}}
+0.01\ell_{\mathrm{GV}}.
\]

其中 \(\ell_S\) 是各视场 RMS 光斑均值；\(\ell_{\mathrm{RP}}\) 约束空气间隔至少 0.01 mm、玻璃厚度位于 1—8 mm、像面净空至少 12 mm；\(\ell_{\mathrm{RA}}\) 将入射和折射角限制在 60° 内。联合训练使用

\[
\ell_{\mathrm{joint}}=\ell_{\mathrm{OD}}+\lambda_{\mathrm{lens}}\ell_{\mathrm{lens}}.
\]

训练采用 Adam、batch size 8，先运行 50k 步，再用半余弦调度衰减 100k 步。由于畸变会移动 anchor 对应内容，计算检测损失时先变换真值框；无偏评估时改为反向校正预测框。

## 实验与证据

实验使用 BDD100K 的 70k/10k 训练与评估图像，合并为车辆、重型车辆、自行车、行人、交通灯和交通标志六类；跨数据集测试使用 14k 张 Udacity 图像。统一光学规格为 f/2、±25° FOV、17.2 mm 焦距和 16 mm 传感器对角线，比较 Doublet、Cooke triplet 与 Tessar。

“完美光学”上限为 BDD100K AP 34.1。直接把其检测器用于基线 Doublet 像差图像，AP 仅 14.6；针对固定镜头微调后达到 30.3；联合优化镜头和检测器后达到 32.0，Udacity AP 也由 23.5 提升至 25.6。2×像差尺度下，Doublet、Cooke、Tessar 的联合优化增益分别为 +3.1、+0.2、+0.9 AP。

与 Tseng 等人的 proxy model 比较时，2× Tessar 在精确光线追迹评估下，代理方案只有 AP 18.4，并有 7.4% 光线渐晕；本文达到 32.2 AP、0% 渐晕。消融同样明确：完整方法为 32.2 AP；连续非目录玻璃降至 25.9；移除近轴像面求解为 26.5；移除 ray-path loss 为 15.4；移除 ray-angle loss 为 24.2；移除 spot-size loss 仅 11.1，且光斑扩大到 142 μm。

## 对 YOLO-Agent 的启发

### 专属 Harness

可将该方法作为 YOLO-Agent 的“光学策略层”：Agent 不只选择数据增强和检测超参数，还搜索镜头曲率、间距及目录玻璃。严格控制组应包括：①完美光学 YOLO；②固定传统 spot-size 镜头并直接推理；③固定镜头、仅重训 YOLO；④镜头与 YOLO 联合优化；⑤联合优化但移除玻璃量化；⑥以代理 PSF 替代精确追迹。

核心指标必须同时报告 COCO 风格 AP@[.50:.95]、跨数据集 AP、平均 spot size、PSNR/SSIM、渐晕光线比例、追迹失败率和制造约束违反率，并在相同 f-number、FOV、焦距、镜片数及训练预算下比较。可证伪标准是：若联合组相对“固定镜头、仅重训 YOLO”在至少三个随机种子上的平均 AP 提升不足 0.5，或任何提升伴随非零渐晕、追迹失败或不可采购玻璃，则不能宣称 Agent 学到了有效的任务驱动镜头设计。

## 优点

- 首次在统一可微框架内自由优化球面、间距和离散目录玻璃。
- 精确追迹、畸变及相对照度建模比固定边界代理网络更可靠。
- 制造约束直接复用追迹中间量，计算结构紧凑。
- 证明检测最优镜头不等于 PSNR、SSIM 或光斑最优镜头。
- 在 BDD100K 与 Udacity 上均观察到任务收益。

## 局限

方法把数据集图像视为无限远、亮度近似线性的场景，且要求原始图像分辨率与质量明显高于被模拟镜头。实验集中于强几何像差的 2—4 片球面镜头，没有完整覆盖衍射、景深、真实传感器噪声、ISP、动态模糊和多深度目标。优化仍依赖合适的人工起始镜头，制造可行性约束也不能替代公差、装调和实物验证。

官方论文页面：https://openaccess.thecvf.com/content/CVPR2023/html/Cote_The_Differentiable_Lens_Compound_Lens_Search_Over_Glass_Surfaces_and_CVPR_2023_paper.html

官方作者代码：https://github.com/princeton-computational-imaging/joint-lens-design

## 评分

**9/10。** 贡献边界清晰，光学模型、离散材料搜索、约束优化和检测证据形成了闭环；主要扣分来自场景假设较强、真实硬件验证不足，以及对高质量起始设计仍存在依赖。
