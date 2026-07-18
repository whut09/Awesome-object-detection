---

# Adaptive Sparse Convolutional Networks With Global Context Enhancement for Faster Object Detection on Drone Images
title: "Adaptive Sparse Convolutional Networks With Global Context Enhancement for Faster Object Detection on Drone Images"
description: "面向无人机高分辨率图像检测头加速的 CEASC 论文笔记，梳理 CESC、CE-GN、AMM 的数据流、实验结果及可复现实验设计。"
tags:
  - CVPR2023
  - 无人机目标检测
  - 稀疏卷积
  - 轻量化检测
  - 全局上下文
  - 动态掩码
---

- **论文链接**：https://openaccess.thecvf.com/content/CVPR2023/html/Du_Adaptive_Sparse_Convolutional_Networks_With_Global_Context_Enhancement_for_Faster_CVPR_2023_paper.html
- **官方代码**：https://github.com/Cuogeihong/CEASC

## 一句话总结

该工作不是压缩主干，而是直接替换 FPN 各层的重型检测头：输入特征 \(X_i\) 分成两路，一路经共享的 \(3\times3\) 掩码卷积得到软特征 \(S_i\)，训练时加入两组 Gumbel 噪声并以 0.5 硬阈值生成二值掩码 \(H_i\)，推理时则以 \(S_i>0\) 激活位置；另一路经 point-wise convolution 得到全局特征 \(G_i\)。四层 SparseConv–CE-GN–ReLU 只在 \(H_i=1\) 的位置计算，并在激活前执行 \(F_{i,j}\leftarrow F_{i,j}+G_i\)，分类头与回归头各自学习独立掩码。

其关键区别在于 CE-GN 不使用稀疏激活元素自身的统计量，而用全局特征 \(G_i\) 的均值和标准差归一化稀疏卷积输出 \(L_{i,j}\)，从而补偿小目标周围上下文。训练阶段额外保留同权重结构的稠密卷积分支 \(C_{i,j}\)，通过 \(L_{norm}\) 约束 \(C_{i,j}\times H_i\) 与稀疏特征一致；AMM 再利用标签分配结果，逐个 FPN 层计算真实前景比例 \(P_i\)，以 \(L_{amm}\) 迫使预测掩码激活比例接近该层前景比例。总损失为 \(L_{det}+\alpha L_{norm}+\beta L_{amm}\)，其中 \(\alpha=1,\beta=10\)。

实验覆盖 VisDrone 与 UAVDT，并以 GFL V1、RetinaNet、Faster R-CNN、FSAF 作为可插拔基线，评价 mAP、AP50、AP75、AR、GFLOPs 和 FPS。GFL V1–ResNet18 在 VisDrone 上由 28.4 mAP、524.95 GFLOPs、13.46 FPS 变为 28.7、150.18、21.55；UAVDT 上由 16.9 mAP、271.66 GFLOPs、20.49 FPS 变为 17.1、64.12、28.47。消融中仅加入 CESC 得到 28.6 mAP、158.23 GFLOPs、19.26 FPS，再加入 AMM 达到 28.7、150.18、21.55，说明逐层自适应比例同时改善精度与速度。

## 研究背景与问题

无人机图像中的目标大多尺寸极小，且飞行高度、俯视角度变化会造成前景占比剧烈波动。稠密检测头在大面积背景上执行卷积，浪费尤为明显：论文指出，采用 ResNet18、512 通道的 RetinaNet 中，检测头占整体 GFLOPs 的 82.3%。普通稀疏卷积虽然能跳过背景，却存在两个直接问题：只计算前景会丢失目标周围的上下文；固定掩码比例无法同时适配不同图像及不同 FPN 层，比例过高浪费计算，过低则漏掉前景。

CEASC 因此把问题定义为“保持检测精度的检测头稀疏化”，而非重新设计完整检测器。CESC 负责让稀疏特征仍能获得全局统计和残差信息，AMM 负责让每层掩码覆盖紧凑但充分的前景区域，两者分别处理信息缺失与稀疏率失配。

## 方法总览

CESC 的输入是第 \(i\) 个 FPN 层特征。掩码网络生成 \(H_i\) 后，四个检测头卷积层仅处理激活像素；point-wise 分支产生的 \(G_i\) 被同时用于 CE-GN 统计量和逐层残差。训练用稠密分支提供特征教师，但推理不需要该分支，因此不会保留其主要计算开销。

AMM 不直接手工指定 0.9 等固定掩码率，而从标签分配后的分类真值 \(C_i\) 统计正样本像素数，令 \(P_i=Pos(C_i)/Numel(C_i)\)。掩码损失最小化 \(Pos(H_i)/Numel(H_i)\) 与 \(P_i\) 的平方差。该监督是 layer-wise 的，因此 P3 至 P7 可以获得不同激活比例，也能随数据中的目标尺度变化。

## 方法详解

VisDrone 包含 7,019 张约 \(2000\times1500\) 图像和 10 类目标，论文采用 6,471 张训练、548 张测试；UAVDT 包含 23,258 张训练图像、15,069 张测试图像，分辨率为 \(1024\times540\)，共 3 类。默认模型为 GFL V1、ResNet18、512 通道；VisDrone 输入为 \(1333\times800\)，训练 15 个 epoch，UAVDT 输入为 \(1024\times540\)，训练 6 个 epoch。训练使用 SGD、初始学习率 0.01，Gumbel-Softmax 温度固定为 1；速度在单张 RTX 2080Ti 上测试。

跨检测器结果表明，CEASC 至少削减约 60% GFLOPs。VisDrone 上 RetinaNet 从 529.81 降至 157.41 GFLOPs，FPS 从 13.41 升至 20.10，mAP 从 21.8 轻微变为 21.6；Faster R-CNN 从 322.25 降至 132.91 GFLOPs，FPS 从 18.17 升至 21.71，mAP 从 24.8 变为 24.6。其优势主要是显著节省计算，同时将精度波动控制在较小范围。

## 实验与证据

CE-GN 是精度恢复的核心。无归一化稀疏版本只有 26.1 mAP；GN、BN、IN 分别得到 28.0、26.1、27.9，而 CE-GN 达到 28.7 mAP，并具有最低的 150.18 GFLOPs 和最高的 21.55 FPS。全局上下文编码方面，point-wise convolution 优于 \(3\times3\) 卷积、GhostModule、CBAM 和 Criss-Cross Attention：它在 28.7 mAP 下达到 21.55 FPS，说明简单全局分支比复杂注意力更适合此处的速度目标。

AMM 的 layer-wise 版本为 28.7 mAP、150.18 GFLOPs、21.55 FPS，统一 global 比例仅为 28.4、162.53、19.84。FPN 消融显示，只使用 P3 时 mAP 降至 26.9；加入 P4 后升至 28.5，加入 P5 后达到 28.7，而继续加入 P6、P7 不再提升 mAP，说明 P4 对该数据中的目标尺度尤为关键。

## 对 YOLO-Agent 的启发

可将该机制迁移到 YOLO 的解耦分类头和回归头：保留两套独立掩码预测器，在每个输出尺度上用 CESC 替换重复的 \(3\times3\) 卷积，并从 YOLO 的正样本分配结果计算各尺度 \(P_i\)。Agent 不应把“固定稀疏率”当作全局超参数搜索，而应记录每层实际激活率、目标尺度分布、mAP 与真实硬件 FPS，检查稀疏化是否只减少理论 FLOPs。

控制组必须包括：原始稠密 YOLO 头、仅 SparseConv、SparseConv+普通 GN、CESC 无 AMM、完整 CESC+AMM，以及统一全局掩码率版本。所有组保持主干、颈部、输入尺寸、训练轮数、标签分配和推理设备一致。主要判据为 mAP、AP50、AP75、GFLOPs、FPS和各尺度激活率；若完整方案相对稠密头的 GFLOPs 降幅不足 60%，或 FPS 未提升至少 20%，或 mAP 下降超过 0.5 个百分点，即判定迁移失败。

## 优点

该方法的强项是插件化：它直接针对高分辨率检测器中最昂贵的检测头，不依赖特定主干，也不需要对一张图像执行多次裁剪和重复推理。稠密教师分支只在训练期存在，AMM 又把稀疏率监督绑定到真实前景比例，因此推理结构较为明确。

## 局限

AMM 监督的是激活数量而非激活位置，位置是否准确仍主要依赖检测损失与掩码网络学习；当标签分配产生的正样本区域不能代表必要上下文时，最优比例估计也可能偏小。论文速度结果来自 RTX 2080Ti，未报告 UAV 端侧设备的内存访问、稀疏算子支持和真实延迟，因此 GFLOPs 降幅不能直接等价为嵌入式平台收益。

## 评分

CEASC 的核心结论是：无人机检测头的稀疏化不能只做像素筛选，还必须同时恢复全局上下文并按 FPN 层自适应控制激活比例。CESC 用全局统计、残差和训练期稠密约束稳定稀疏特征，AMM 用标签前景比例替代人工固定掩码率；两者结合后，在 VisDrone 和 UAVDT 上均以基本不损失 mAP 的代价显著降低计算量并提高推理速度。
