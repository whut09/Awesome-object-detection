---
title: FlexEvent - Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies
publication: NeurIPS
year: 2025
description: "说明 FlexEvent 如何以 FlexFuse 动态分配帧与事件贡献，再用 FlexTune 将低频真值传播到高频时刻，从而训练单一跨频检测器。"
paper: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8064e4ebbcbe594628887b420956d8c3-Abstract-Conference.html
code: https://flexevent.github.io
tags: [事件相机, RGB事件融合, 高频检测, 伪标签, 多模态检测]
---

# FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies

## 一句话总结

FlexEvent 通过带噪声门控的 FlexFuse 融合 RGB 语义与多频事件特征，再用 FlexTune 从低频真值传播出时序一致的高频伪标签，使同一检测器从 20 Hz 扩展到 180 Hz 而不必人工逐频标注。

## 研究背景与问题

事件相机按亮度变化异步输出事件，具有微秒级时间分辨率，但现有检测数据的框标注通常跟随低频 RGB 帧。RVT、SAST、SSM 等模型于是把事件积累在固定时间窗内训练；频率升高后，每个时间窗事件更稀疏，语义和纹理不足，性能明显坍塌。逐频人工标注代价又抵消了事件相机的高频优势。

FlexEvent 要解决两个耦合问题：怎样让一个融合模型根据当前输入自适应分配事件与帧的贡献，以及怎样只用低频真值为未标注高频时间点生成可靠监督，而不是依赖容易漏掉“突然出现/消失目标”的线性插值框。

## 方法总览

训练样本先以标注频率 `a` 取得 RGB 帧 `F` 和事件窗 `Ea`，再把同一时间段切成 `b/a` 个高频子窗并随机抽取 `Eb`。RVT 事件分支和 ResNet-50 帧分支分别输出四尺度特征；**FlexFuse** 在每尺度计算事件/帧软门控并融合 `Ea`、`Eb`。随后 **FlexTune** 先在与真值时间戳对齐的高频末窗上稀疏训练，再执行高频启动、Temporal Consistency Calibration（TCC）和循环自训练，最终把跨频率能力固化进单模型。

## 方法详解

**FlexFuse 的数据流**是 `(F,Ea,Eb)` → 双分支四阶段特征 → 每尺度拼接事件与帧特征 → 门控矩阵产生事件权重 `α` 和帧权重 `β`。门控加入可学习标准差控制的高斯噪声，避免固定偏向某一模态；融合为 `α·hE+β·hF`，再把不同事件频率的融合结果相加，送入 YOLOX 风格检测头。正则项约束 `α、β` 的方差/均值比，防止模型退化成单模态。

**FlexTune 的数据流**首先只取高频集合中与低频框时间戳一致的最后一个事件窗，以真实框完成 Low-Frequency Sparse Training。然后模型对全部高频子窗推理得到初始伪框；TCC 依次做正向/反向事件增强、按类别置信度阈值与 NMS 过滤、基于 IoU 的检测关联，并删除短于 `Ltrack` 的轨迹。得到的高频标签与低频真值共同进入 Cyclic Self-Training，多轮更新模型和伪标签。

**跨频推理机制**不在运行时执行 TCC：FlexTune 是离线训练过程，推理时只需一个 RGB 帧加随后不同长度的事件片段。RGB 提供相对稳定的空间语义，事件窗长度决定运行频率，门控根据稀疏度自动重新配比。这一区别使 FlexEvent 可在 20、36、45、60、90、180 Hz 间切换，而不是为每个频率保存独立权重。

## 实验与证据

数据集包括 **DSEC-Det**（78,344 帧/60 序列）、**DSEC-Detection**（52,727 帧/41 序列）和 **DSEC-MOD**（13,314 帧/16 序列）。基线覆盖 RVT、SAST、SSM、LEOD、DAGr、HDI-Former、CAFR、RENet。DSEC-Det 验证集上 FlexEvent 达 **57.4 mAP、78.2 AP50、66.6 AP75**，DAGr-50 为 41.9 mAP；DSEC-Detection 三类平均 mAP 为 **47.4**，CAFR 为 38.0；DSEC-MOD 平均为 **36.9**，RENet 为 29.0。

频率切片最能验证主张：FlexEvent 从 20 Hz 的 58.7 mAP 到 180 Hz 仍有 **50.9**，而 RVT 从 39.5 降至 6.2。消融中，无 Tune/Fuse 的跨频平均为 43.7，加入 FlexTune 为 48.0，使用插值高频真值与融合为 54.7，完整方案为 **57.2**；FlexFuse 将平均 mAP 从 43.7 提升到 56.4，且 180 Hz 从 22.9 提升到 49.2。TCC 最佳参数为车/人阈值 0.6、`Ltrack=6`、IoU 0.6。

## 对 YOLO-Agent 的启发

1. agent 可把“传感器运行频率”作为可控上下文，自动选择事件窗而非维护多套模型。
2. 多模态融合应监控门控权重和模态退化，不能只比较最终 mAP。
3. 高频伪标签要依赖轨迹一致性与双向事件，而非简单框插值。

**YOLO-Agent Harness（FlexEvent）**：实验矩阵以相同 RGB 锚点生成 20/36/45/60/90/180 Hz 输入；融合轴的**对照组**包含 event-only、逐元素 Add、Concat、Attention 与带噪声门控的 FlexFuse，监督轴另比线性插值、只做高频启动、加入 TCC、完整 FlexTune，并以 DAGr/RVT 作为跨频参照。需要记录的**指标**是各频率 mAP、APS/APM/APL、车辆与行人 AP、新生/静止目标召回、单帧延迟、事件密度分桶性能，以及 FlexFuse 的 `α/β` 均值、方差和饱和比例。**失败判断**落在机制行为上：90 或 180 Hz 相对 20 Hz 的降幅不小于 DAGr，FlexTune 的轨迹剪枝令新出现目标召回下降超过 3 点，稀疏事件下静止目标持续漏检，或门控长期塌缩到单一模态时，即使总体 mAP 上升也不能通过。

## 优点

- 把自适应融合与跨频监督传播拆成两个可验证模块。
- 高频提升显著，且 FlexTune 不增加在线推理开销。
- 三个大规模事件数据集、频率曲线和融合消融较完整。

## 局限

- 依赖标定同步的 RGB 帧，纯事件系统无法直接使用。
- TCC 的阈值和轨迹长度仍需数据集调参，快速出现/消失目标可能被剪除。
- 模型 45.4M 参数，虽快于 DAGr，但明显大于主流 event-only 基线。

## 评分

- 创新性：9/10
- 技术完整性：9/10
- 实验说服力：9/10
- 工程可迁移性：8/10
- 对 YOLO-Agent 价值：8.5/10
