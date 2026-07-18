---
title: "Toward RAW Object Detection: A New Benchmark and a New Model"
description: "解析 ROD 高动态范围 RAW 检测基准、双层自适应动态范围调整模型及其对检测智能体设计的启示。"
tags:
  - RAW object detection
  - HDR
  - ROD dataset
  - YOLOX
  - autonomous driving
---

# Toward RAW Object Detection: A New Benchmark and a New Model

## 一句话总结

本文以 Sony IMX490 Bayer 传感器采集并融合四路 12-bit 数据，建立含 25,207 幅图像、237,379 个框和昼夜驾驶场景的 24-bit ROD 基准，证明高位深 RAW 不能直接交给常规检测器，并以图像级幂变换、像素级分段变换和融合卷积构成轻量前端，仅依靠检测损失与 YOLOX 或 Sparse R-CNN 端到端训练。

## 研究背景与问题

研究首先比较 10-bit 手机 RAW、12-bit PASCALRAW 和 ROD 的 24-bit RAW：动态范围越高，直接检测的退化越严重。以 0.90M 参数 YOLOX 为例，24-bit 数据的 RAW AP 仅 34.6，而 GEO GW5300 ISP 输出的 SDR AP 为 52.1；相反，10-bit RAW 与 SDR 仅相差 0.5 AP。这表明问题不是“RAW 天生不适合检测”，而是极宽动态范围造成像素集中于低值区，纹理难以进入 DNN 的有效特征空间。

作者进一步拆解软件 ISP：去马赛克 DM、灰度世界自动白平衡 AWB、Gamma 动态范围调整 DRA、双边滤波去噪和 JPEG 压缩。在 24-bit ROD 上，RAW 经 DM+AWB 只有 34.6 AP，加入 DRA 后跃升至 52.1；完整 ISP 为 53.3，而单独 DRA 已达 51.7，移除 DRA 的完整流水线则跌至 35.2。由此，论文把研究重点精确收敛到“为检测任务学习动态范围压缩”，而不是重建视觉悦目的 sRGB 图像。

ROD 面向真实驾驶：分辨率为 2880×1856，包含约 10k 白天图像和约 14k 夜间图像，类别为汽车、行人、骑行者、有轨电车、三轮车和卡车。相比 4,259 幅、12-bit 的 PASCALRAW，以及 2,230 幅、14-bit 的 LOD，它同时扩大了数据规模、动态范围和昼夜覆盖。

## 方法总览

输入 RAW 数据 \(X\) 先降采样为 \(X_{lr}\)，再并行进入 **Image-Level Adjustment** 与 **Pixel-Level Adjustment**。两支分别生成全局变换结果 \(Y_I\) 和局部变换结果 \(Y_P\)，取平均后由融合卷积 \(F_c\) 输出：

\[
Y=F_c\left(\frac{Y_I+Y_P}{2}\right).
\]

处理结果直接送入下游检测器完成分类与定位。训练阶段，调整网络和检测器从零开始联合优化，不使用配对 SDR、图像重建损失或 ISP 监督；推理阶段则固定为“RAW—自适应调整—检测器”的单次前向数据流。

## 方法详解

### 图像级调整

低分辨率输入经过 \(n_g=3\) 层步幅卷积提取全局特征，再由全连接层预测每幅图像的指数 \(\gamma\)，实施自适应幂变换：

\[
Y_I=g(X,F_I(X_{lr};\theta_I))=X^\gamma.
\]

该分支控制整幅图像的像素分布，承担类似 Gamma 校正但由场景决定参数的强动态范围压缩。

### 像素级调整

局部分支以 \(n_l=2\) 层步幅卷积生成 \(K-1\) 个像素级掩码 \(m_k\)，并将强度轴划分成分段基函数 \(\delta_k(X)\)：

\[
Y_P=f(X,F_P(X_{lr};\theta_P))
=\sum_{k=0}^{K-1}m_k\delta_k(X).
\]

与单一全局曲线不同，掩码允许不同空间位置采用不同调整幅度，因此可同时处理强眩光、暗背景和局部低对比纹理。所有参数预测都在 256×256 的 \(X_{lr}\) 上完成，而变换施加于原输入，降低额外计算量。

### 检测监督

总目标为：

\[
L_{\text{total}}=L_{\text{cls}}+\lambda L_{\text{reg}},
\]

其中分类使用 BCE Loss，回归使用 IoU Loss。调整模块没有被要求产生“好看”的图像，其输出只需最大化检测性能，这也是它区别于通用 learned ISP 的核心训练边界。

## 实验与证据

ROD 按场景独立划分，白天取 9k、夜间取 13k 训练，其余作为相应测试集，以避免昼夜域偏移混入评估。输入统一缩放到 1280×1280，训练 300 epoch、前 5 epoch warmup，采用 SGD、余弦学习率、动量 0.9、权重衰减 \(5\times10^{-4}\)、batch size 32，并使用翻转、尺度抖动和 Mosaic。指标包括跨 IoU 阈值的 AP、AR，以及 AP50、AP75。

在 0.90M 参数 YOLOX 上，直接 RAW 的白天/夜间 AP 为 34.6/1.7，GEO ISP 的 SDR 为 52.1/50.3；本文方法达到 **58.7/54.2 AP**，分别超过 SDR 6.6 和 3.9 AP，AP50 达 85.3/83.0。调整前端仅增加 0.08M 参数和 0.64G FLOPs，并优于 Gamma、Mu-Log、GTM、IA-Gamma、IA-ISPNet、MW-ISPNet 与 Lite-ISPNet。

该收益并非只适用于单阶段检测器。使用 104.54M 参数 Sparse R-CNN 时，本文方法取得 77.4 AP，超过 SDR 的 73.5 和 IA-ISPNet 的 75.6；YOLOX 扩大至 8.92M 参数后，本文方法为 75.5 AP，仍高于 SDR 的 69.3。纹理分析也提供了机制证据：RAW 的 GLCM 熵为 11.1691、AP 为 34.6；本文方法将熵提高到 24.5954，并取得 58.7 AP，高于 IA-Gamma 的 24.1645/53.5。

## 对 YOLO-Agent 的启发

### 专属 Harness

YOLO-Agent 可将该方法实现为由场景统计驱动的可选择前端，而不是默认把 RAW 转成标准 RGB。严格实验应固定 ROD 昼夜划分、1280×1280 输入、同一 YOLOX 初始化和 300-epoch 训练，设置四个控制组：① RAW 直接检测；② GEO ISP 的 SDR；③ IA-Gamma；④图像级调整、像素级调整和融合卷积组成的完整方案。统一报告 AP、AR、AP50、AP75、附加参数量和 FLOPs。

Agent 的决策变量应包括是否启用调整前端、预测的 \(\gamma\)、像素分段掩码及昼夜场景置信度；但最终策略仍须由检测损失学习，不能用视觉亮度或主观图像质量替代任务指标。一个可证伪的失败标准是：在相同协议下，完整方案若在白天或夜间任一子集的 AP 不高于 SDR，或无法分别复现相对 SDR 的 6.6/3.9 AP 增益趋势，则“任务驱动 RAW 调整优于现成 ISP”的核心假设不成立；若其额外成本明显超过论文的 0.08M 参数、0.64G FLOPs，也不能声称保持了原方法的轻量性。

## 优点

- 先通过跨位深实验和 ISP 消融定位根因，再设计模型，论证链条完整。
- ROD 同时覆盖 24-bit HDR、昼夜驾驶和大规模框标注，补足 RAW 检测基准空白。
- 调整网络只接受检测监督，绕开 RAW–sRGB 精确配对和主观图像质量目标。
- 在 YOLOX、Sparse R-CNN及不同模型规模上均有效，具有较好的检测器可迁移性。
- 计算集中于低分辨率参数预测，性能增益与额外成本比例突出。

## 局限

ROD 主要来自单一 Sony IMX490 传感器与驾驶领域，跨传感器响应、Bayer 排列和其他任务的泛化尚未充分验证。24-bit 数据由多曝光子像素线性融合得到，结论不必然适用于单曝光高位深 RAW。论文也缺少两条调整分支各自贡献、分段数量 \(K\)、降采样尺寸及融合方式的系统消融；此外，端到端训练仍依赖完整检测标注，未解决新相机的低成本适配问题。

## 评分

**8.7/10。** 其主要价值不只是提出轻量前端，而是用新基准证明：HDR 信息保存在 RAW 中并不意味着检测器能直接利用它，关键是以任务监督重新塑造像素分布与纹理。实验覆盖充分、核心结果明确，但传感器泛化和模块级消融仍有改进空间。

- 官方论文页面：https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Toward_RAW_Object_Detection_A_New_Benchmark_and_a_New_CVPR_2023_paper.html
- 官方作者代码：https://gitee.com/mindspore/models/tree/master/research/cv/RAOD
