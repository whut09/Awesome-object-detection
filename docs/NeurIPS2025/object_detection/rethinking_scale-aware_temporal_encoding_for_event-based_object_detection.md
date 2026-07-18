---
title: "Rethinking Scale-Aware Temporal Encoding for Event-based Object Detection"
description: "SATE 将 DDRL 循环层前移到高分辨率低层特征，以解耦形变卷积完成运动对齐和历史状态融合，再通过独立尺度分支服务事件目标检测。"
tags: ["NeurIPS 2025", "事件相机", "目标检测", "SATE", "ConvLSTM", "Deformable Convolution"]
---

# Rethinking Scale-Aware Temporal Encoding for Event-based Object Detection

**会议**：NeurIPS 2025  
**论文**：[NeurIPS 官方页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d450dceeacd6083d1d550247377f2320-Abstract-Conference.html)  
**官方代码**：[BIT-Vision/SATE](https://github.com/BIT-Vision/SATE)  
**任务**：事件流目标检测

## 一句话总结

SATE 反对只在深层特征上堆 RNN 或全局注意力：它把三层 **Decoupled Deformable-enhanced Recurrent Layer（DDRL）** 放在 2×、4×、8× 的高分辨率阶段，用 ConvLSTM 保存时间状态、用解耦调制形变卷积做运动对齐，再让三个尺度分支独立下采样并经 FPN+YOLOv6 检测头融合。

## 研究背景与问题

事件相机输出 `(x,y,t,p)` 异步亮度变化，具有高动态范围与微秒级延迟，但静止或慢速目标产生的事件极少，当前时刻的稀疏体素可能缺失完整轮廓。已有 RED、RVT、SAST 多在经过显著下采样后的高层特征上级联循环模块；到此时细粒度边缘运动已经被空间压缩。Transformer 又引入全局注意力成本，却未必更适合局部、稀疏的运动轨迹。

论文的关键判断是：时间建模的位置比模块复杂度更重要。事件密度最高、空间细节最完整的浅层阶段应先完成历史信息恢复；不同目标尺度和速度则应由独立时空分支处理，避免一条连续深链把所有尺度绑定到同样的下采样节奏。作者还认为浅层特征包含任务无关噪声，需要借助形变卷积的平滑与自适应采样抑制冗余传播。

## 方法总览

50 ms 事件窗口被离散成 5 通道 voxel grid，经 stride=1 stem 后连续进入三层 DDRL。每层先用 5×5、stride=2 的 Conv-BN-ReLU 下采样，再由 **DDRB** 将当前特征与前一时刻 hidden state 或 cell state 拼接、3×3 压缩；当前事件特征另行预测 deformable offset 和 modulation mask，调制形变卷积对融合特征运动对齐，之后经 SE Block 和 ConvLSTM 更新状态。三层输出分别进入独立下采样分支，最终形成 8×、16×、32×、通道 128/256/512 的特征，交给 FPN 与 YOLOv6 head。

## 方法详解

**DDRL** 包含 downsampling 和 **Decoupled Deformable-enhanced Recurrent Block（DDRB）**。所谓 decoupled，是把“从当前事件估计像素运动”与“融合当前/历史语义”分开：offset `Δxk` 和 mask `Δmk` 仅由当前 `Fs_t` 的两条 3×3 卷积分支产生；被采样的特征则来自 `Concat(Fs_t,Ts_t-1)` 压缩后的表示。3×3 核有 9 个采样点，使用 8 组 grouped deformable convolution，使不同通道组学习不同形变模式。

形变融合结果经 SE 做通道重标定，再输入 ConvLSTM。论文分别测试上一时刻 hidden state `H` 与长期 cell state `C` 作为 `Ts_t-1`，cell 通常更好。把循环层放在 2×、4×、8× 而非 8×、16×、32×，不仅保留小目标和快速边缘，还避免高通道深层 DDRL 的参数膨胀。三个 **Scale-Specific Spatiotemporal Encoding** 分支各自继续下采样，因而浅层时间信息不会被迫沿同一串行路径传播。

训练使用 ADAM、OneCycle，半个 batch 做完整 BPTT、另半做 TBPTT，并使用水平翻转、zoom-in、zoom-out。检测端不是新设计：FPN 负责多尺度融合，YOLOv6 head 使用 distribution focal loss、classification loss 和 regression loss。这样消融可以把收益主要归因于事件 backbone，而非检测头变化。

## 实验与证据

Gen1 含 39 小时、304×240 事件数据，约 228K vehicle 框与 28K pedestrian 框；过滤边长小于 10 像素或对角线小于 30 的框。1 Mpx 含约 15 小时、1280×720、约 2500 万框和 vehicle/pedestrian/two-wheeler 三类目标，输入降至 640×360。eTram 为路侧视角，约 10 小时、200 万框、8 类，更稀疏。三者均以 mAP 为主指标。

Gen1 训练 batch 6、序列长 21、学习率 2e-4、400K iteration；1 Mpx 为 batch 8、序列长 5、3e-4、800K；eTram 训练 400K iteration。SATE 在 Gen1 达 **52.7 mAP、8.80 ms**，超过 ASTMNet 的 46.7、RVT-B 的 47.2、SAST-CB 的 48.2；1 Mpx 达 49.1 mAP，比 ASTMNet 高 0.8；eTram 达 **33.0 mAP、13.05 ms**，比第二名 SAST-CB 高 3.0。模型从零训练，不依赖预训练权重。

DDRB 消融中，基础 ConvLSTM backbone 为 50.7 mAP；直接融合 hidden/cell 为 51.7/52.0；使用 DDConv 为 52.2/52.7；DDConv+SE 的 hidden 方案为 52.3，cell 方案保持 52.7 且 AP50 提到 81.7。形变融合相比直接 5×5 convolution 在参数更少的同时继续提升，支持“运动估计与特征融合解耦”的主张。

骨干消融更关键：高层时间建模 49.5 mAP、53.8M 参数；单分支 51.3；RVT backbone 48.0；完整方法 52.7、26.4M。附录进一步把 DDRL 放在 High/Mid/Low，结果为 49.5/51.0/52.0，低层位置同时只需 23.5M，而高层方案达到 53.8M，直接支持循环层前移。可视化显示低层历史特征能在遮挡和低运动时恢复目标响应。

速度比较也避免了“更准但完全不可用”的误解：Gen1 上 S5-ViT-B 为 47.7 mAP、8.16 ms，SATE 为 52.7、8.80 ms；RVT-B 虽有 10.2 ms 延迟，但精度只有 47.2。1 Mpx 上 SATE 的 13.3 ms 慢于 S5-ViT-B 的 9.57 ms，却显著快于 ASTMNet 的 72.3 ms。因而它的优势更接近用少量循环与形变成本换取稳定精度，而不是所有数据集上的绝对最低延迟。

三套数据的标注频率和观察视角差异很大：Gen1 是低分辨率车载，1 Mpx 同时覆盖昼夜高分辨率驾驶，eTram 则是路侧稀疏监控。SATE 在三者上都领先，说明低层时间编码并非只针对某一种事件密度调出的特例。

## 对 YOLO-Agent 的启发

对视频、事件或多帧 YOLO，代理不应默认把 ConvLSTM 插在 neck 顶部。可把循环位置、历史状态类型、尺度分支是否共享、运动对齐算子列为结构搜索变量，并按目标速度和尺寸分桶评估。SATE 也提醒代理：注意力并非时间建模的默认答案，浅层局部运动对齐可能在精度和延迟上更合适。若输入是普通视频，还需先验证帧间稠密纹理是否会让浅层循环显存失控。

### Harness

对照组为相同 voxel、FPN、YOLOv6 head 的无循环 CNN 与高层 ConvLSTM；实验组比较低层 ConvLSTM、低层直接融合、DDConv、DDConv+SE，并固定输出尺度与通道。观测总体 mAP/AP50、small/slow-object AP、遮挡后重现召回、单步延迟、参数量、序列显存和状态漂移。通过阈值设为总体 mAP 提升≥1.0，small 或 slow 子集至少提升 1.5，同时延迟增幅≤25%；若仅靠参数增长、长序列显存超基线 40%，或快速运动提升但静止目标召回下降超过 2 点，则判定时序模块不满足部署要求。

## 优点

- 用位置消融证明浅层时间编码优于深层堆叠。
- DDRB 明确分离运动估计与历史特征融合，数据流可解释。
- Gen1、1 Mpx、eTram 覆盖车载和路侧两种事件场景。
- 检测头保持常规 FPN+YOLOv6，便于隔离 backbone 收益。

## 局限

- DDRL 仍增加循环状态和形变卷积计算，26.4M 参数并非极轻量。
- 5-bin、50 ms voxel 是较简单的事件表示，未充分利用连续时间信息。
- 训练时间长，Gen1 单张 RTX 3090 约需 4 天。
- 稀疏卷积、量化和实际事件芯片部署尚未验证。

## 评分

- **创新性**：8/10
- **实验充分度**：9/10
- **工程可迁移性**：8/10
- **YOLO-Agent 时序架构价值**：9/10
