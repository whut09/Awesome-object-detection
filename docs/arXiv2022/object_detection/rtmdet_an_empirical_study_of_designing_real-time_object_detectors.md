---
title: "[论文解读] RTMDet: An Empirical Study of Designing Real-Time Object Detectors"
description: "RTMDet 以 CSPNeXt、大核深度卷积、SepBN Head 和缓存增强系统研究实时检测器的均衡设计。"
tags: ["arXiv 2022", "目标检测", "RTMDet", "CSPNeXt", "SepBN Head"]
---

# RTMDet: An Empirical Study of Designing Real-Time Object Detectors

**会议**: arXiv 2022  
**论文**: [arXiv](https://arxiv.org/abs/2212.07784)  
**代码**: [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo)  
**任务**: 单阶段实时检测与实例分割

## 一句话总结

RTMDet 用 CSPNeXt 的 5×5 depthwise convolution、共享卷积却分离 BN 的 SepBN Head、soft center prior 动态匹配和 cached Mosaic/MixUp，重新平衡 backbone、neck、head、训练 I/O 与实际 GPU 延迟。

## 研究背景与问题

许多实时检测器只更新 backbone，neck 和 head 仍沿用旧比例；FLOPs 也无法说明 depthwise、group convolution 在具体 GPU 上的真实效率。YOLOX 式 Mosaic/MixUp 每个 iteration 读取多张图，训练可能受磁盘吞吐限制。完全独立的多尺度头参数冗余，而全共享头又忽略尺度统计差异。RTMDet 因此把网络组成、标签质量和数据流水线放到同一经验研究中。

## 方法总览

CSPNeXt block 先调整通道，经 5×5 depthwise convolution 扩大感受野，再用 pointwise convolution 混合，最后与 CSP 短路分支融合；PAFPN 复用相似单元。三尺度特征进入 SepBN Head，各尺度共享分类/回归卷积权重但维护独立 BN。soft center prior 先限制候选，动态 soft label assignment 再按分类与 IoU 联合质量分配。增强图像从缓存队列取样，减少重复读盘。

## 方法详解

### CSPNeXt 与大核

作者比较普通 3×3、堆叠小核和大核 depthwise，最终把 5×5 放进基本块，以较低参数代价扩大有效感受野。CSP 部分连接控制梯度和激活量，SiLU 替换较弱激活；选择依据同时包含 AP 和 TensorRT 延迟，而非只看 GFLOPs。

### SepBN Head

不同 FPN 层的数据分布不同，所以 BN 统计不共享；分类或回归卷积承担相似变换，可以跨尺度复用。SepBN 位于“完全独立头”和“所有参数共享”之间，减小参数同时保留尺度专属性。分类标签还编码定位质量，降低排序错配。

### Cached augmentation

缓存保存近期样本及标注，Mosaic、MixUp 从池中组合，不再为每次增强额外触发多次存储读取。训练尾段切换到弱增强，使验证分布更接近真实图像。相同特征体系还能挂接动态 mask head 得到 RTMDet-Ins。

## 实验与证据

COCO 上 RTMDet-T/S/M/L/X 覆盖多预算；RTMDet-L 报告 51.3 box AP，T4 TensorRT FP16 约 1.5 ms，RTMDet-X 达 52.8 AP。消融按宽深比例、CSPNeXt block、5×5 depthwise、SepBN、assignment 与缓存增强展开，并与 YOLOX、YOLOv6、YOLOv7、PP-YOLOE 在相同输入尺度比较。论文同时提醒 TensorRT 版本、batch 与算子实现会改变速度排序。

## 对 YOLO-Agent 的启发

应把理论 FLOPs、CUDA kernel 数、显存带宽和端到端延迟分开记录，并为 3×3 普通卷积、3×3/5×5 depthwise 建同通道微基准。SepBN 对照包含独立头、全共享头、仅卷积共享三组，观察 APs/APm/APl 和 BN 漂移；缓存增强需统计 loader wait 与类别采样分布。若大核使 APs 下滑、SepBN 的尺度统计趋同却不省时，或缓存造成长尾类别曝光偏移，即判定该配置不适配目标数据与硬件。

## 优点

- 统一研究检测器结构、训练吞吐和后端时延。
- SepBN Head 给出参数复用与尺度适应的实用折中。
- 设计覆盖检测和实例分割，不局限单一任务。

## 局限

- 经验结论依赖 MMYOLO 与 NVIDIA 工具链。
- 大量组件选择缺少统一理论解释。
- 候选输出仍需 NMS 去重。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★★
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★★
