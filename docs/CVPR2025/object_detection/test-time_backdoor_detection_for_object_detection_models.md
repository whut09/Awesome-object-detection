---
title: "Test-Time Backdoor Detection for Object Detection Models"
description: "解析 TRACE 如何以黑盒上下文一致性和焦点一致性，在测试时识别目标检测后门样本。"
tags: ["CVPR 2025", "目标检测", "后门检测", "TRACE", "黑盒防御"]
---

# Test-Time Backdoor Detection for Object Detection Models

**会议**：CVPR 2025  
**论文**：[arXiv](https://arxiv.org/abs/2503.15293)  
代码：未发现论文声明的官方代码。

## 一句话总结

TRACE 不读取模型参数和训练集，而是通过背景混合测量 Contextual Transformation Consistency（CTC），再用 Natural Backdoor Objects 探针测量 Focal Transformation Consistency（FTC），联合发现诱导假阳性和漏检的后门。

## 研究背景与问题

目标检测后门可能让触发区域凭空出现目标、令真实目标消失，或组合多种攻击目标；分类领域的测试时样本检测通常只处理单标签输出，难以覆盖框回归和多目标上下文。本文防御者只有被攻击模型的黑盒最终输出，没有训练数据、攻击类型和触发器先验，仅能访问公开背景图和前景参考物，因此必须从输出稳定性寻找异常。

## 方法总览

TRACE 由 Contextual Information Transformation、Focal Information Transformation 和 test-time evaluation 组成。对可见的假阳性触发物，系统把输入与 `b` 张背景低透明度混合，统计目标置信度方差形成 CTC，再用 SSIM 和同类公开参考图过滤 Natural Backdoor Objects（NBOs）。对会诱导漏检、因而没有可见框的触发器，系统把位置不敏感的 NBO patch 采样到图像各处，利用 Island Effect 形成 FTC。最后将高 FTC 与低 CTC 归一化相减，以阈值 `γ` 判定样本。

## 方法详解

假阳性触发器通过 shortcut learning 建立触发图案到目标类别的强映射，跨背景置信度异常稳定；普通物体则依赖道路、水下等上下文。TRACE 用 `(1-αbg)x+αbgδ` 在图像层混入背景，并以不同背景下预测分数方差 `ΔVar_B` 衡量稳定性。停止标志等 NBO 也可能稳定，因此将可疑框与 universal visual benchmarks 计算 SSIM，仅保留低于阈值 `τ` 的异常候选，再取图像内最小 CTC。

漏检触发器会压低邻域回归与分类响应。TRACE 选择位置不变性强的 NBO（示例为 stop sign）作为探针，通过 Monte Carlo 在 `k` 个不重叠位置贴入 patch，并重复 `f` 次近似扫描显著图。探针接近触发区域时置信度下降、覆盖触发器时新检测恢复，形成两侧下降、中间回升的 Island Effect；FTC 用二阶空间变化与新增检测偏移共同量化。最终 `Trace(x)=sigmoid(ΔVar_F)-sigmoid(ΔVar_B)`，大于 `γ` 即判为后门样本。

## 实验与证据

评估覆盖 MS-COCO、PASCAL VOC、Synthesized Traffic Signs，检测器包括 YOLOv5、Faster R-CNN、DETR；攻击包括 OGA、ODA、RMA、GMA、CIB、UTA、DC，并与 TeCo、STRIP、FreqDetector、SCALE-UP、Detector Cleanse 比较。主实验在 42 个后门模型组合上得到平均 F1 `0.880±0.040`、AUROC `0.897±0.041`，论文整体共验证 63 个后门模型；TRACE 相比 Detector Cleanse 的 F1 提升约 30%。

移除 CTC、SSIM Filter 或 FTC 都会伤害 RMA/GMA 的 precision、recall 与 AUROC。查询量在 `b=30` 背景、`f=50` 前景时取得性能与时间折中，SSIM 阈值取 `τ=0.1`。面对知道 TRACE 的自适应攻击，攻击者加入 `J_adap`；OGA 在 `λ=5e-4` 时 F1 0.581、ASR 0.733，而 `λ=1e-5` 时 F1 0.937、ASR 0.975，体现隐蔽性与攻击成功率冲突。

## 对 YOLO-Agent 的启发

TRACE 可作为部署侧安全门。**Harness** 准备干净、FP-trigger、FN-trigger、混合攻击和自然标准化物体五组，对比无防御、仅 CTC、仅 FTC、完整 TRACE；观测样本级 F1/AUROC、干净误报率、攻击 ASR、每图查询次数与 p95 延迟。通过条件为 AUROC ≥0.85、干净误报率 ≤5%、ASR 至少下降 50%，且 NBO 组误报不高于普通干净组 3 个百分点；若开销超过基础检测 5 倍、任一攻击原语召回低于 0.7，或正常 stop-sign 被系统性拦截，则失败。

## 优点

- 明确覆盖假阳性、漏检与混合攻击。
- 黑盒、无训练数据设定贴近第三方模型部署。
- 自适应攻击实验揭示规避与 ASR 的矛盾。

## 局限

- 背景混合和前景探针需要大量重复查询。
- 依赖公开背景、类别参考图和合适 NBO。
- 阈值仍需校准，物理触发器和视频连续性证据有限。

## 评分

- **创新性**：★★★★★
- **实验充分度**：★★★★☆
- **安全实用性**：★★★★☆
- **YOLO-Agent 参考价值**：4.8/5
