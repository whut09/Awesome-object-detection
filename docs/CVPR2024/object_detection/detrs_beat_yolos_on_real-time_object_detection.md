---
title: "[论文解读] DETRs Beat YOLOs on Real-Time Object Detection"
description: "RT-DETR 用高效混合编码器与不确定性最小查询选择，将免 NMS 的 DETR 推入实时区间。"
tags: ["CVPR 2024", "目标检测", "RT-DETR", "NMS-Free", "Hybrid Encoder"]
---

# DETRs Beat YOLOs on Real-Time Object Detection

**会议**: CVPR 2024  
**论文**: [arXiv](https://arxiv.org/abs/2304.08069)  
**代码**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)  
**任务**: 端到端实时目标检测

## 一句话总结

RT-DETR 只在最高层特征内用 AIFI 做注意力，再由 CCFM 进行卷积式跨尺度融合，并挑选分类与定位都更确定的 encoder token 初始化 object queries，以可裁剪 decoder 实现无需 NMS 的速度调节。

## 研究背景与问题

DETR 的集合预测消除了 anchor 和 NMS，但多尺度 Transformer encoder 的长序列与 decoder 迭代使其难以实时；YOLO 的后处理时间又随候选数、anchor 数和 NMS 阈值变化。论文先分解真实端到端延迟，发现 anchor-based YOLO 生成的候选更多，NMS 开销更高，再针对 DETR 的 encoder 瓶颈和 query 初值做优化。

## 方法总览

Backbone 输出 $S_3,S_4,S_5$。Efficient Hybrid Encoder 将 intra-scale interaction 与 cross-scale fusion 解耦：AIFI 仅展平低分辨率 $S_5$ 执行 self-attention；CCFM 通过 top-down、bottom-up 卷积路径把语义传给细粒度层。uncertainty-minimal query selection 从 encoder 特征中选固定数量候选，decoder 逐层更新类别和框，每层辅助头参与集合损失，最终直接输出预测集合。

## 方法详解

### AIFI 与 CCFM

三尺度统一 attention 时，高分辨率 token 数主导复杂度。AIFI 把全局实体交互限制在 $S_5$；Cross-scale Convolutional Fusion Module 以 Fusion Block、上采样和下采样完成尺度通信，让卷积承担更规则的细节处理。

### Query selection

只按分类分数挑 proposal 会选中类别自信却定位不准的 token。RT-DETR 以较小不确定性选择兼顾类别与框质量的 encoder 输出，为 decoder 提供更接近目标的初始 reference，从而减少后续修正难度。

### Decoder speed tuning

各 decoder 层都有预测头，部署时可保留不同层数而无需重训同一模型。算力紧张时提前退出，精度优先时使用完整迭代，形成连续速度档位。

## 实验与证据

COCO 上 RT-DETR-R50 报告 53.1 AP、T4 TensorRT FP16 108 FPS，R101 为 54.3 AP、74 FPS。论文比较单尺度/多尺度 Transformer、纯跨尺度融合以及 AIFI+CCFM；查询消融对照分类选择和质量联合选择；逐层裁剪 decoder 显示延迟下降与 AP 变化。与 YOLO 的速度表把 NMS 纳入端到端测量，是结论成立的关键口径。

## 对 YOLO-Agent 的启发

YOLO 与 RT-DETR 必须统一计入 resize、前向、解码、NMS/集合筛选，并固定 TensorRT 精度和 batch。AIFI 对照包含三尺度 attention、仅 $S_5$ attention、纯卷积 encoder；query 实验记录初始 top-k IoU、分类-IoU 相关及每层 decoder AP。若裁掉一层就造成 AP 急剧坍塌、初始 query IoU 不优于仅分类方案，或输出仍需额外去重才能控制重复率，则端到端优势没有被复现。

## 优点

- 依据计算瓶颈重构 DETR，而非简单缩小模型。
- 真正移除 NMS，并能用同一权重切换速度。
- 编码器和查询初始化均有独立证据。

## 局限

- 集合匹配训练仍比普通 YOLO 更复杂。
- attention 的后端效率依赖硬件与 TensorRT 版本。
- 极少 decoder 层时精度不可避免下降。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★★
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★★
