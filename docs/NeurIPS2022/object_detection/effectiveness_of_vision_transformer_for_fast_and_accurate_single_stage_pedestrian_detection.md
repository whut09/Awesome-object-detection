---

# Effectiveness of Vision Transformer for Fast and Accurate Single-Stage Pedestrian Detection
title: "Effectiveness of Vision Transformer for Fast and Accurate Single-Stage Pedestrian Detection"
description: "以单个可变形视觉 Transformer 编码器增强空间自适应与多尺度特征，在保持单阶段推理效率的同时提升遮挡行人检测精度。"
tags:
  - 行人检测
  - 单阶段检测
  - 可变形视觉Transformer
  - 多尺度特征
  - NeurIPS2022
---

## 一句话总结

该方法没有照搬 DETR 的六组编码器—解码器，而是把**单个 Deformable Vision Transformer Encoder**置于骨干网络与检测头之间。骨干输出先经卷积或反卷积形成下采样率为 \(1/4、1/8、1/16\) 的三尺度特征，并以 Group Normalization 稳定后续线性运算；编码器中的 Multi-Scale Deformable Attention 根据查询特征预测稀疏采样位置与注意力权重，从不同尺度聚合目标相关区域，避免完整自注意力在高分辨率行人图像上的二次方内存开销。

编码后的多尺度特征保持各自分辨率，随后统一上采样至 \(H/4\times W/4\)，逐尺度执行 L2Norm，再沿通道维拼接，并通过压缩层降至适合检测头的通道数。最终特征可直接接入 SSD、ALFNet 式渐进锚框细化器或无锚 CSP 检测头。实验使用 Caltech 与 Citypersons，核心对照包括 SSD300、progressive refinement、CSP，以及 Faster R-CNN、RepLoss、KGSNet、PRNet 等单阶段和两阶段方法；主指标是 FPPI 位于 \([10^{-2},10^0]\) 时的对数平均漏检率 MR，分别报告 Reasonable、Small、Heavy 与 All，并同时比较 FPS 和参数量。

关键消融明确支持这一数据流：Caltech 上 CSP 的 Reasonable、Heavy、All 分别为 6.8%、50.7%、62.3%，加入增强模块后降至 3.7%、42.8%、56.97%，其中 Heavy 下降 **7.9 个百分点**；速度由 39.2 FPS 降至 29.5 FPS。编码器数量从 1 增至 2、3、4 时，Reasonable MR 由 3.7%恶化为 4.9%、4.8%、5.5%，证明单编码器并非简单的速度折中，而是该结构中准确率最佳的配置。

## 研究背景与问题

单阶段检测的主要困难并不只是缺少更深的网络，而是没有 RPN 提供目标区域，分类器容易被大面积背景淹没，边界框也必须从零回归。作者据此把可变形注意力定位为一种“无 proposal 的空间过滤器”：每个查询只访问少量、可学习的跨尺度采样点，让行人及其有效上下文获得更高权重，同时压低遮挡物和背景响应。

与完整 Deformable DETR 相比，该设计删除了解码器和对象查询预测流程，只保留编码器的特征增强能力，因此仍属于可插拔的单阶段 neck。其价值尤其体现在 Heavy 子集：遮挡会破坏人体完整轮廓，而跨尺度稀疏采样可以从可见身体片段及邻近尺度补充判别信息。

## 方法总览

无锚 CSP 训练目标由中心热图损失 \(L_c\)、高度图损失 \(L_h\) 和偏移损失 \(L_o\) 构成，权重分别为 0.01、1、0.1；推理时宽度由预测高度乘固定宽高比 0.41 得到，保留分数高于 0.01 的框，并使用 IoU 阈值为 0.5 的 NMS。锚框方法则联合分类与定位损失，分类权重设为 0.01。

Caltech 输入尺寸为 336×448 进行主要训练，部分结构消融使用 480×640；Citypersons 输入为 640×1280，速度测试使用原表注明的 1024×2048。两者分别训练 10 与75个 epoch，批量大小为16与4，采用 ImageNet 预训练 ResNet50/VGG16、Adam、移动平均权重及阶梯学习率。

## 方法详解

在 Caltech 上，三尺度输入 \(1/4、1/8、1/16\) 得到 4.1%的 Reasonable MR 和44.7%的 Heavy MR，优于改变中高层尺度的其他组合。增强特征采用“上采样—L2Norm—拼接—通道压缩”时整体最佳；直接相加或各尺度独立预测不能同时维持 Reasonable 与 Heavy 表现。

参数量也没有随 Transformer 模块增加：CSP 总参数量为39.96M，增强版本为32.24M；neck 从14.68M降至8.73M，原因是融合后的压缩表示及更轻的检测头抵消了编码器成本。

## 实验与证据

Citypersons 验证集上，CSP 加模块取得 Reasonable 10.9%、Heavy 38.6%、All 37.2%，速度6.8 FPS；Heavy 比两阶段 KGSNet 的39.7%低1.1个百分点。Caltech 未跨数据集预训练时为3.7%/42.8%/56.97%，使用 Citypersons 预训练后进一步达到 Reasonable 2.6%、Heavy 28.0%、All 53.9%。

与 KGSNet 依赖候选区域、边界框细化、关键点检测和超分辨率网络不同，该方案在 Caltech 达到29.5 FPS；论文报告 KGSNet 在 Titan X 上仅5.9 FPS，且尚未计入 ALFNet 生成候选框的时间。不过 GPU 不同，因此该速度比较只能说明结构复杂度趋势，不能视为严格硬件等价基准。

## 对 YOLO-Agent 的启发

复现 Harness 应设置四组核心控制：原始 CSP；CSP 加参数量匹配的普通卷积 neck；CSP 加单个可变形编码器及完整多尺度融合；相同结构改为2、3、4个串联编码器。进一步分别移除编码前 GN、编码后 L2Norm、三尺度拼接或将跨尺度融合改为 Add/Sep，以定位收益来源。

统一报告 Reasonable、Heavy、All 的 MR、FPS、总参数和 neck 参数；Caltech 重点核对6.8%→3.7%与50.7%→42.8%，Citypersons 核对11.7%→10.9%与41.8%→38.6%。若完整模型在相同数据划分和输入尺度下无法使 Caltech Heavy 至少下降7.9个百分点，或 Citypersons Heavy 无法达到38.6%附近，同时速度低于6.8 FPS，则应判定实现未复现论文的精度—速度平衡，而不能仅以 Reasonable 的轻微改善宣告成功。

## 优点

优势是模块可同时适配锚框、渐进细化锚框和无锚检测器，并在减少参数的同时显著改善严重遮挡子集。选择单编码器也体现了明确的任务化设计：注意力用于增强卷积特征，而非把行人检测强行重写成完整 DETR 推理流程。

## 局限

论文未提供随机种子重复实验或误差条，速度又跨 RTX 3090 与 Titan X 比较，结论的统计稳定性和硬件公平性有限。固定宽高比0.41具有明显的行人类别先验，不宜直接推广到姿态变化更剧烈或通用目标检测场景。

作者还指出，当负样本参考点的注意力采样范围延伸至真实目标区域时会产生假阳性；未来需要约束采样位置和注意力权重，而不是单纯堆叠更多编码器。

## 评分

- 论文链接：[NeurIPS 官方页面](https://proceedings.neurips.cc/paper_files/paper/2022/hash/afb8caec018d3c8f6ef8b81fa52386fe-Abstract-Conference.html)

**???????**: [?Effectiveness of Vision Transformer for Fast and Accurate Single-Stage Pedestrian Detection?????????????????????????](https://proceedings.neurips.cc/paper_files/paper/2022/hash/afb8caec018d3c8f6ef8b81fa52386fe-Abstract-Conference.html)
- 官方代码：论文正文未给出作者 GitHub 或官方代码仓库，检查表亦明确标注未提供复现代码、数据与指令；可使用上述官方论文页面获取论文及补充材料。
