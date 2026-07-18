---
title: "Beyond RGB: Adaptive Parallel Processing for RAW Object Detection"
description: "详解 Raw Adaptation Module 的并行 ISP 参数预测与反沙漏特征融合。"
tags: ["ICCV 2025", "目标检测", "RAW 图像", "ISP", "低照度"]
---

# Beyond RGB: Adaptive Parallel Processing for RAW Object Detection

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Gamrian_Beyond_RGB_Adaptive_Parallel_Processing_for_RAW_Object_Detection_ICCV_2025_paper.html)  
**代码**：[官方代码](https://github.com/SonyResearch/RawAdaptationModule)

## 一句话总结

Raw Adaptation Module（RAM）把传统串行 ISP 改成任务驱动的并行处理：共享 Raw Parameter Encoder（RPE）读取 RGGB-stacked RAW，多个 Raw Parameter Decoder（RPD）分别预测 gamma、brightness、color correction 和 white balance 参数，各分支独立处理原输入，再由 reverse-hourglass Feature Fusion 输出三通道检测优化表示。

## 研究背景与问题

传统 ISP 为人眼优化，demosaicing、去噪、tone mapping 和 JPEG 会不可逆丢失纹理、阴影与高光信息；直接训练 detector 处理高动态 RAW 又会因像素分布偏向零而难以学习。已有 learned ISP 多按固定顺序串行执行，后续函数只能看到被前序修改过的结果，白平衡剪裁等损失无法恢复。论文主张让每个 ISP 函数平等读取原始 RAW，再学习任务相关融合。

## 方法总览

Bayer RAW 先堆叠成 RGGB。单个卷积 RPE 生成共享特征向量，四个 MLP 型 RPD 分别输出 ISP 参数；gamma、brightness、3×3 CCM、RGB white-balance gain 同时作用于原输入，得到四组表示。它们在通道维拼接后进入反沙漏 Feature Fusion，中部扩宽以保留丰富特征，再压回与输入同空间尺寸的三通道图像，送入 backbone 与 detector 端到端训练。

## 方法详解

自适应 gamma 以 RPD 输出为指数，brightness 预测加性偏移，CCM 预测九个颜色矩阵元素，WB 预测 R/G/B 增益。共享 RPE 避免每个函数重复抽取内容特征，也保证并行分支成本低于串行版本。Feature Fusion 不是简单平均，而是学习哪些处理结果对当前图像有用；论文额外加入可学习 normalization，RAM-T 则缩小同一架构以测试收益是否仅来自参数量。

## 实验与证据

数据覆盖 ROD-Day 4,053 张、ROD-Night 12,036 张，NOD 7,200 张、46,000 个实例，LOD 2,230 张，以及 PASCALRAW 4,259 张。主比较采用 Faster R-CNN/ResNet18，从头训练；天气和噪声实验使用 DINO/ResNet50。RAM 在 LOD-Dark、LOD-Normal、ROD-Day、ROD-Night、NOD-Nikon、NOD-Sony、PASCALRAW 的 mAP 分别为 34.9、40.1、28.3、44.5、31.0、32.4、66.8，全部为最佳或并列最佳。

ROD-Night 上 RAM 仅增加 0.54M 参数和 1.05 ms，却达到 44.5 mAP；更小的 RAM-T 增加 0.2M、1.01 ms，仍有 44.2。冻结 YOLOX-X、只训练预处理时，RAM 在 LOD-Dark/NOD-Sony 达 50.8/29.6 mAP，优于 AdaptiveISP、GenISP、RAOD。关键消融中，串行流水线在 LOD-Dark/Normal 为 29.9/35.1 mAP，参数 1.06M；并行 RAM 为 35.2/40.1，参数 0.54M。

强噪声下 RAM noisy/denoised 为 42.4/45.7 mAP，小目标 mAPs 为 30.5/33.4，均优于 RAW 与 sRGB。ROD-Night 合成雨雪雾中，RAM 从 clean 56.6 mAP 仅降至 53.1/53.2/54.1，而 sRGB 的雪天从 51.0 降至 37.4。Shapley 分析显示 Feature Fusion 是最关键组件，gamma、WB、brightness 的贡献随曝光和数据集改变。

RAM 的四个分支保持明确物理含义。Gamma 预测指数进行 tone mapping；brightness 预测加性偏移；CCM 输出 3×3 颜色矩阵；white balance 输出三个通道增益。它们都读取同一 RGGB 输入和共享 RPE 特征，因此不会像串行 ISP 那样让早期白平衡剪裁永久限制后续 gamma。反沙漏融合层先扩宽通道再压回三通道，负责在细节、动态范围和颜色校正之间选择。

数据集覆盖的传感器条件互补：ROD 为 24-bit HDR 日夜驾驶，NOD 为 Sony/Nikon 14-bit 低照户外，LOD 为同场景长短曝光八类目标，PASCALRAW 为 Nikon 12-bit 日光图像。这使 RAM 的结果不只代表夜间增强。它在 PASCALRAW 的 66.8 mAP 仅小幅超过 FeatEnHancer 66.5，却在 ROD-Night 从 sRGB 38.9 提升到 44.5，说明动态范围越困难，适配收益越明显。

效率表排除了“换更大 backbone”的解释。RAW ResNet50 比 ResNet18 增加 12.47M 参数和 2.33 ms，mAP 仅从 30.4 到 34.7；RAM 在 ResNet18 上只加 0.54M、1.05 ms，却达到 44.5。冻结 YOLOX-X 的实验进一步说明 RAM 能把 RAW 转到既有 sRGB 特征空间：LOD-Dark 50.8 mAP 超过 GenISP 49.5，NOD-Sony 29.6 超过 27.8。

Shapley 结果不是所有分支固定正贡献。LOD-Dark 的 brightness 有用，长曝光 LOD-Normal 中几乎无效；ROD 数据更依赖 gamma，ROD-Day 的 brightness 用于阴影，NOD-Sony 则来自所有函数和融合器的组合。Feature Fusion 被单层卷积替换时损失最大。因此自动裁剪应按相机/曝光配置保存，而不能在一个数据集上删分支后全局复用。

噪声实验区分 noisy 与 LED denoised 输入，避免把显式去噪收益算给 RAM。中等噪声时 RAW/sRGB/RAM noisy mAP 为 42.2/41.0/45.1，denoised 为 44.9/44.9/48.5；强噪声时 RAM noisy 42.4 甚至超过 sRGB denoised 42.3。小目标差距更明显，强噪声 RAM denoised mAPs 33.4，而 RAW/sRGB 为 30.2/28.7，说明并行 ISP 主要保住容易被去噪和 tone mapping 抹掉的细粒度结构。

恶劣天气实验每种天气都从头训练对应 DINO/ResNet50，而不是单一 clean 模型零样本测试，因此证明的是 RAM 能在该退化分布上学到更稳定表示。它在 snow 中比 sRGB 高 15.8 mAP，但不能直接等同跨天气泛化。实际 Harness 应同时加入“在 clean 训练、weather 测试”和“各天气专训”两套协议，分别衡量鲁棒性与适应上限。

## 对 YOLO-Agent 的启发

RAM 适合作为输入端可搜索的 task-aware ISP：Agent 可根据传感器和场景启用不同处理分支，并用 Shapley 或移除实验裁剪无效函数。关键约束是所有分支读取同一 RAW，融合器决定贡献，而不是把可逆性较差的操作固定成串行链。对边缘设备可优先尝试 RAM-T。

### 论文专属 Harness

- **对照组**：固定同一 YOLO detector，比较 Bayer RAW、RawPy sRGB、串行自适应 ISP、并行 RAM、RAM-T，并做逐个 ISP 分支移除。
- **观测指标**：总 mAP、mAPs、雨雪雾和三档噪声 AP、参数、预处理延迟、各分支 Shapley 值与输出动态范围。
- **通过阈值**：并行结构相对串行至少提升 3 mAP，同时参数更少；RAM-T 相对完整 RAM 下降不超过 0.5 mAP，预处理低于整帧预算 8%。
- **失败判断**：某处理分支在三个数据集均为负 Shapley，或融合图发生高光剪裁且恶劣天气 AP 下降超过 5 点，则移除该分支或停止部署。

RAW 部署还受相机黑电平、白电平、Bayer 排列和位深影响。论文将输入统一为 RGGB stacked，若传感器为 BGGR/GRBG 或元数据归一化不同，RPE 学到的分布会整体偏移。应在数据加载阶段把物理归一化写入可测试模块，并用同场景 RAW/sRGB 对检查饱和区域、暗部直方图和通道顺序；这些错误通常不会在网络结构检查中暴露。

定性图只显示置信度高于 0.5 的框，适合观察阴影小行人、透明瓶子和夜间遮挡，但不能替代完整 PR 曲线。复现截图应同时固定阈值并保存错误类别统计。

## 优点

- 并行与串行对照直接验证核心设计，不依赖单纯增大 backbone。
- 七个 RAW 子集及噪声、天气实验覆盖广，效率报告完整。

## 局限

- ISP 函数集合依赖数据集经验选择，未证明可自动扩展到所有传感器。
- 主训练从头进行，与大规模 RGB 预训练 detector 的适配仍需更多验证。

## 评分

- **问题重要性**：★★★★★
- **方法清晰度**：★★★★★
- **实验证据**：★★★★★
- **工程可迁移性**：★★★★☆
- **YOLO-Agent 参考价值**：★★★★★
