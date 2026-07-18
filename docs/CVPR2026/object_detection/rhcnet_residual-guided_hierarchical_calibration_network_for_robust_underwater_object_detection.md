---
title: "RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection"
description: "RHCNet 用 RGFE 恢复水下退化图像的边缘纹理，再由 HFCP 与二类聚类原型完成跨尺度语义校准。"
tags: ["CVPR 2026", "目标检测", "水下检测", "RHCNet", "特征校准", "K-means"]
---

# RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RHCNet_Residual-Guided_Hierarchical_Calibration_Network_for_Robust_Underwater_Object_Detection_CVPR_2026_paper.html)  
**代码**: [YitengGuo/RHCNet](https://github.com/YitengGuo/RHCNet)  
**任务**: 复杂退化环境下的水下目标检测

## 一句话总结

RHCNet 在 ResNet-50 中用 Location-Aware Module 与 Residual-Guided Feature Enhancement 主动补回高频结构，再让 Hierarchical Feature Calibration Pyramid 先以双向路径修正尺度特征、后用 K=2 的 cluster-guided calibration 分离前景和混乱背景。

## 研究背景与问题

水下散射和吸收会像物理低通滤波器一样削弱边缘与纹理，颜色、光照和悬浮颗粒又让前景背景高度相似。普通注意力往往只做隐式重加权，FPN/BiFPN 则假设不同尺度在空间上已经对齐；在折射、模糊和遮挡下，这两项假设都不稳。作者因此把问题拆成两个连续环节：先恢复深层特征丢失的结构，再在融合前校准跨尺度语义和空间位置。

## 方法总览

RHCNet 以 RetinaNet/ResNet-50 检测框架为基础。浅层位置先验由 LAM 提供；主干各阶段的特征进入 RGFE，通过 Semantic Convolution Transform 提取局部高频异常，并经通道与空间双注意力抑制噪声后残差注入语义流。处理后的多尺度特征交给 HFCP：底向上路径累积全局语义，顶向下路径通过 Position-Aware Module 补偿破碎区域，最后 CGCA 用聚类原型按内容相似度校准像素，再进入检测头。

## 方法详解

RGFE 的 SCT 由 depthwise separable convolution 加 pointwise convolution 构成，显式抽取梯度和边缘。SCT 特征一方面直接保留，另一方面进入 channel attention 与 spatial attention 的乘性调制，再与原语义特征拼接映射成残差。不同阶段先下采样到统一处理尺度、经 RGFE 后再上采样，通过可学习系数 λ 注入，目的不是无条件锐化，而是让网络控制结构补偿强度，避免把水下噪点一起放大。

HFCP 遵循 calibrate-then-fuse。底向上部分由全局平均池化通道分支和三个不同感受野卷积分支组成，积累跨层语义；顶向下的 PAM 把高层完整对象上下文投影到低层，修复遮挡造成的局部断裂。随后 cluster-guided feature calibration 在 detached 特征上做 K-means，固定 K=2 形成前景、背景原型；每个像素与原型点积后 softmax，按权重汇聚原型得到注意力增强特征。损失采用 Task-Adaptive Quality Focal Loss，把预测 IoU 与 centerness 以 ρ=0.5 组合为连续分类软标签，并配合 GIoU 回归，分类/回归权重为 1/2。

这里的聚类分配本身不可微，因此作者在 detached features 上计算 K-means，再用聚类统计和可学习权重生成可微注意力图，避免梯度穿过离散簇指派。与像素对像素 non-local attention 相比，两个原型把复杂度和噪声敏感度压低；与直接 FPN 求和相比，像素先根据内容相似性找到更合适的语义中心。LAM、RGFE、PAM、CGCA 因而形成由浅层位置、局部高频、顶向下结构补偿到全局原型过滤的逐级链路，而不是四个并列注意力插件。

## 实验与证据

论文在 DUO（7,782 张复杂水下图像）、UTDAC（5,643 张多类水下生物图像）和 COCO（118,287 张、80 类）上使用 COCO 指标。DUO 上 RHCNet 达到 AP 70.53、AP50 87.56、AP75 77.29，超过 CIDNet 的 68.83 AP；UTDAC 达到 53.35 AP，高于 YOLOv11 的 49.75 与 CIDNet 的 49.57。COCO 上为 45.68 AP、28.33 APS，论文表中优于同列比较的 YOLOv11 44.18 AP。代价为 70.04M 参数、145.16G FLOPs，明显不是轻量方案。

DUO 消融基线 RetinaNet-R50/FPN 为 57.06 AP。换 ResNet-101 并保留模块为 70.03，仍略低于完整 R50 的 70.53；用 BiFPN 替代 HFCP 仅 66.45；把 CGCA 换成 vanilla self-attention 为 68.12。去掉 LAM、RGFE、PAM、CGCA 后分别为 66.45、66.90、66.74、65.86，去掉 PAM 与 CGCA 同时降至 65.24，证明恢复结构与融合前校准都不是可忽略装饰。

所有实验基于 MMDetection，使用 RTX 4070 SUPER、SGD 动量 0.9、权重衰减 0.0001，初始学习率 0.001，共训练 35 epoch，并在第 27、32 epoch 阶梯衰减。比较方法既有 FRCNN、FCOS、GFL、RepPoints，也有 YOLOv7、YOLOv10、YOLOv11、DJENet、UDMD、CIDNet 等水下专用模型。UTDAC 上 RHCNet 的 AP50/AP75/APS 为 86.93/58.97/27.23，说明提升不只来自宽松 IoU；COCO 的 AP75 为 49.36、APL 为 59.46，则用于证明层级校准没有把网络限制在水下纹理。

从主表可见，RHCNet 在 DUO 的 APS/APM/APL 为 56.63/71.70/69.94，既改善小型生物，也没有牺牲大目标；UTDAC 的 APM/APL 为 48.90/59.29。与轻量 YOLOv11 的 2.60M 参数、6.50G FLOPs 相比，RHCNet 精度更高但计算量大一个数量级，因此论文更像验证结构校准上限，而非给出水下边缘设备的现成部署答案。定性图中主要优势是高浑浊、小目标和暗光下减少漏检与悬浮颗粒误报。

质量感知分类项也值得单独观察。软标签由 IoU 与 centerness 的几何加权产生，focusing factor 再根据预测概率和软标签差距强调难样本，因此低质量但分类分数高的水下框会受到更强惩罚。它与 RGFE/HFCP 的关系是：前者提供更可信的结构与语义特征，损失再迫使最终置信度反映定位质量。若只迁移模块而保留普通 Focal Loss，可能无法完整复现主表收益。

表中去除 LAM 与 RGFE 后仍有 65.26 AP，去除 PAM 与 CGCA 后为 65.24，两个阶段的联合损失几乎相同；但完整模型比两者均高约 5.3 点，说明局部恢复和层级校准呈互补而不是替代关系。相比单独换 R101 的结果，R50 完整版只多 0.50 AP，却用结构设计抵消了更深主干的容量优势，这正是作者用“更强 backbone”对照要证明的重点。

## 对 YOLO-Agent 的启发

更合理的迁移方式是把 SCT-RGFE 放进高分辨率主干层，把 CGCA 放在 PAN/FPN 融合前，而非整体复制 70M 模型。Harness 对照组为 YOLO 基线、仅 LAM、仅 RGFE、BiFPN、PAM+普通自注意力、PAM+K=2 CGCA。观测指标需分开记录 DUO/UTDAC 的 AP、APS、低对比子集召回、背景误报、边缘梯度响应、FLOPs 与端侧延迟。通过阈值建议水下 AP 至少提升 2 点、APS 提升 2.5 点，背景误报下降 10%，同时参数增长不超过 15%。失败判断是 COCO AP 下跌超过 0.5，或高频增强使悬浮颗粒误报上升，此时说明结构恢复过度。

## 优点

- 模块命名与数据流围绕“恢复—校准—融合”展开，适配水下退化机理。
- 与更强 backbone、通用 BiFPN、普通自注意力均做了直接对照。
- 除两套水下数据，还用 COCO 检查非水下场景的泛化损失。

## 局限

- 70.04M 参数和 145.16G FLOPs 与实时水下机器人预算存在距离。
- K=2 将语义简化为前景/背景，在多目标遮挡或复杂类别关系中可能不足。
- 统一 MMDetection 配方不代表与各专用方法完全同等训练资源，主表横向公平性仍需复核。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★☆☆
- **YOLO-Agent 参考价值**: ★★★★☆
