---
title: "When Transformers Meet Mamba: A Hybrid Transformer-Mamba Network for Video Object Detection"
description: "TMambaDet 用 SADT 建模帧内空间关系、TCBM 线性扫描跨帧动态，并在 MaET 中交织注意力与双向 Mamba。"
tags: ["CVPR 2026", "目标检测", "视频目标检测", "TMambaDet", "Mamba", "Deformable DETR"]
---

# When Transformers Meet Mamba: A Hybrid Transformer-Mamba Network for Video Object Detection

**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Qi_When_Transformers_Meet_Mamba_A_Hybrid_Transformer-Mamba_Network_for_Video_CVPR_2026_paper.html)  
**代码**: 未发现论文声明的官方代码  
**任务**: 视频目标检测

## 一句话总结

TMambaDet 把帧内长程关系交给 Spatial Adaptive Deformable Transformer，把跨帧序列交给 Temporal Cascaded Bidirectional Mamba，再用 Mamba Entangled Transformer Decoder 让 object queries 依次经历自注意力、可变形跨注意力和双向状态空间细化。

## 研究背景与问题

视频检测需要利用相邻帧补足模糊、遮挡和小目标信息。Transformer 能建立长距离依赖，但把多帧所有 token 放入注意力会产生二次复杂度；Mamba 对长序列为线性复杂度，却不如 Transformer 擅长精细上下文交互。论文没有让两者在同一位置竞争，而是按任务维度分工：空间内保留注意力的选择能力，时间上使用状态空间递推，实例查询阶段再把两类算子串联。

## 方法总览

每帧先经共享 ResNet-101 或 Swin-B 提取多尺度特征，投影、展平并加入 Deformable DETR 位置编码。SADT 在每帧内部聚合空间上下文；多帧 SADT 输出连接后送入 TCBM，以 temporal-prioritized bidirectional scan 形成时空编码特征。Deformable DETR 产生的 object queries 与编码特征共同进入 MaET：查询自注意力后执行 multi-scale adaptive deformable attention，再经 cascaded bidirectional Mamba，最后 FFN 输出分类与边框。

## 方法详解

Spatial Adaptive Deformable Transformer（SADT）的核心是 MADAttn。标准 deformable attention 给所有 query 固定 K 个采样点，本文用轻量 MLP 根据 query 特征预测 sigmoid 强度 α，再把它线性映射并四舍五入到 Kmin=2 与 Kmax=8 之间。复杂或遮挡目标可获得更多采样，简单区域少采样；采样偏移、权重和多尺度双线性插值仍沿用可变形注意力，因此保留稀疏空间交互。

Temporal Cascaded Bidirectional Mamba（TCBM）先对 SADT 特征做 LN，第一路经 Linear、Conv1d 和 temporal-prioritized forward SSM，第二路经 Linear 与激活，两路 Hadamard 相乘得到 forward feature；随后再级联反向 SSM。其 TPBS 顺序不是把一帧空间全部扫描完再换下一帧，而是在固定空间位置 j 上依时间遍历所有帧，再移动到下一个 j；反向支路从最后空间位置开始但仍优先跨时间，使同一位置的动态连续性在状态更新中相邻。

Mamba Entangled Transformer（MaET）先用 multi-head self-attention 交换 query 间信息，再以 MADAttn 从 TCBM 时空特征采样，最后用 cascaded bidirectional Mamba 沿查询序列补充长程实例上下文。SADT、TCBM、MaET 分别对应帧内、帧间和 query-feature 对齐，数据流和功能边界明确。

TCBM 采用级联而非把正反向 Mamba 简单并联求和：forward branch 先生成已经吸收过去与未来参考帧顺序的特征，backward branch 再在反向空间次序上继续校正。TPBS 把相同空间位置的多帧 token 排成连续片段，因此状态转移首先看到该位置随时间的变化，而不是被同帧其他位置打断。MaET 中的 Mamba 则处理 object query 序列，作用对象与 TCBM 不同；前者提高实例表示，后者提高视频 memory，二者不能互相替代。

## 实验与证据

ImageNet VID 含 3,862 个训练视频、555 个验证视频、30 类；EPIC-KITCHENS-55 含 272 个视频、290 类，均报告 IoU=0.5 的 mAP。训练随机从同一 clip 取当前帧外的 4 帧，测试使用 25 帧。ResNet-101 下 TMambaDet 在 ImageNet VID 达 87.9 mAP、20.6ms/RTX 5090，高于 TGBFormer 的 86.5 和 DGC-Net 的 86.3；Swin-B 为 92.1 mAP。EPIC-KITCHENS-55 上 ResNet-101/Swin-B 分别为 45.1/50.8，超过同 backbone 的 TransVOD Lite 与 YOLOV++。

组件消融以 Deformable DETR 的 78.3 mAP 为基线：单加 SADT 为 80.6，单加 TCBM 为 83.8，单加 MaET 为 80.9；TCBM+MaET 为 86.1，完整模型为 87.9。对快运动目标，基线 57.8，完整方案 73.1，增幅 15.3 点。SADT 固定 K=2/4/6/8 的 mAP 与延迟分别为 85.3/15.3ms、87.4/21.1ms、87.6/30.2ms、87.5/40.0ms，自适应采样达到 87.9/20.6ms。扫描消融中 S4、S6、Bi-S6、TPBS 为 85.7、86.4、87.3、87.9。

实现中 SADT 为四层、六头、四个特征尺度，TCBM 三层，MaET 四层；ImageNet VID 每帧使用 60 个 query，EPIC-KITCHENS-55 使用 300 个。ImageNet VID 同时利用 VID 与 DET 训练，短边缩放到 600，AdamW 训练 140K iteration；EPIC 训练 220K iteration。自适应采样还被移植到原生 Deformable DETR 和 TransVOD++：前者 ResNet-101 从 78.3 提到 79.6，后者从 82.0 提到 83.1，说明 MADAttn 的收益并非只在完整 TMambaDet 中成立。编码层数量分析显示 SADT 到四层后精度饱和，TCBM 三层最佳，继续堆叠反而下降。

运动分桶中，完整模型对 slow/medium/fast 分别达到 93.3/86.8/73.1，而基线为 86.4/76.5/57.8，时间模块对快速目标贡献最突出。运行时间比较也有双面性：ResNet-101 版本 20.6ms 明显快于 MEGA、HVRNet 等方法，Swin-B 版本则为 39.0ms，慢于 YOLOV++ 的 15.7ms，但精度高 1.4 点。EPIC 上同样出现 50.8 mAP 与 41.3ms 的权衡，说明“Mamba 线性复杂度”不等于整网必然最快，backbone 和 query 数仍决定总延迟。

SADT 的采样数量是离散整数，但由连续 α 预测后四舍五入；论文没有把这一分支描述为强化学习或独立监督，而是随检测损失端到端学习。复杂目标获得更多点只是作者对可视化行为的解释，工程复现还应统计 Kq 分布，确认模型没有长期退化为固定 K。MaET 输出仍沿用 Deformable DETR 的分类与框回归损失，因此主表变化可以归因于编码和查询交互，而非更换标签分配或损失函数。

可视化中 TransVOD++ 会漏掉快速出现的小狗和松鼠，TMambaDet 能恢复并给出更高置信度；但论文也展示特别小的松鼠仍被漏检，并把原因归为小目标动态建模有限。这个负例说明时序长度和状态空间能力不能弥补空间分辨率的一切损失，YOLO 迁移时仍需保留高分辨率检测层，而不能只增加 TCBM。

## 对 YOLO-Agent 的启发

可以把 TCBM 作为 YOLO 视频 neck 的跨帧聚合器，同时保留单帧检测头。Harness 对照组应为单帧 YOLO、固定 K 的时序 deformable attention、普通 S6 Mamba、Bi-S6、TPBS-TCBM，以及加入/不加入 query 级 MaET 的版本。观测指标必须按 slow/medium/fast 运动分桶记录 mAP，并记录 4/8/16/25 帧时的峰值显存、每帧延迟、状态长度和遮挡恢复率。通过阈值建议整体 mAP 至少 +2、fast 桶至少 +4，25 帧延迟低于同精度时序 Transformer 的 70%。失败判断是增益只来自更多参考帧，或小目标漏检率上升、状态随镜头切换污染超过三帧。

## 优点

- Transformer 与 Mamba 按空间、时间、查询三个层面明确分工，避免概念性拼接。
- 自适应采样与 TPBS 都有替代算法和延迟对照，能解释性能效率来源。
- 在 ImageNet VID 外加入 290 类厨房视频，验证复杂类别场景的迁移性。

## 局限

- 训练和测试参考帧数量不同，25 帧推理对在线系统的缓存要求较高。
- 论文未声明官方代码，TPBS 的张量布局和实际 kernel 效率难直接复现。
- 作者承认极小目标仍会漏检，说明当前时序状态对微小动态建模不足。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★☆
- **YOLO-Agent 参考价值**: ★★★★★
