---

# PolarStream: Streaming Object Detection and Segmentation with Polar Pillars
title: "PolarStream: Streaming Object Detection and Segmentation with Polar Pillars"
description: "面向旋转式激光雷达流式扇区输入，以极坐标柱、双向多尺度上下文填充、特征去畸变和距离分层卷积实现低延迟三维检测、语义分割与全景分割。"
tags:
  - 论文笔记
  - 流式感知
  - 激光雷达
  - 三维目标检测
  - 全景分割
  - NeurIPS2021
---

- **论文链接**：[NeurIPS 官方论文页](https://proceedings.neurips.cc/paper_files/paper/2021/hash/e1e32e235eee1f970470a3a6658dfdd5-Abstract.html)

**???????**: [?PolarStream: Streaming Object Detection and Segmentation with Polar Pillars?????????????????????????](https://proceedings.neurips.cc/paper_files/paper/2021/hash/e1e32e235eee1f970470a3a6658dfdd5-Abstract.html)
- **官方代码**：论文正文未提供作者 GitHub URL，亦未声明官方代码仓库；实现细节与补充材料请从上述官方论文页获取。

## 一句话总结

该方法不等待激光雷达完成整圈扫描，而是把一次扫描按方位角切成 \(n=1,2,4,8,16,32\) 个楔形扇区，扇区到达后立即处理。它以 **Polar Pillars** 替代 Han et al. 与 STROBE 使用的笛卡尔矩形区域：每点表示为 \((r,\theta,z,x,y,i,t)\)，连续十帧点云先经自车运动补偿并变换到当前帧，再按柱分辨率 \((\delta_r,\delta_\theta,\delta_z)\) 动态体素化。由于极坐标网格可直接展开成紧凑的 \(r\)-\(\theta\) 特征图，模型无需计算包围楔形扇区的大量空白矩形区域。

数据流为：动态 Polar Pillar → Pillar Feature Encoder → PointPillars 风格二维 CNN 主干与 U-Net 式上采样结构 → 并行检测、语义分割头。检测端改造 CenterPoint：十类中心热图的高斯半径由目标框在距离与方位角上的跨度决定，并回归中心偏移、尺寸、高度、方向和速度；语义头把柱编码特征与主干双线性上采样特征拼接，再经单层 \(1\times1\) 卷积输出。全景融合则把“thing”点分配给同类别且中心最近的检测框；流式情况下需收齐同一扫描各扇区的框后执行全局融合。

实验基于 nuScenes：700/150/150 个场景用于训练、验证和测试，覆盖十类检测目标及十六类语义/全景类别；对照包括 Cartesian Pillars、Polar Pillars、Han et al. 的 stateful-RNN/stateful-NMS，以及 STROBE 的多尺度记忆模块，指标为检测 mAP、NDS、语义 mIoU、全景 PQ 和端到端延迟。在全扫描消融中，普通 Polar Pillars 检测 mAP 为 48.2；仅加入 Range Stratified Convolution & Normalization 提升至 49.1，仅加 Feature Undistortion 提升至 48.6，两者联合达到 50.3，额外运行时间仅约 0.5 ms。

## 研究背景与问题

核心矛盾不是单纯减少点数，而是同时解决“楔形输入与矩形卷积不匹配”及“扇区视野过窄”。笛卡尔柱必须处理楔形扇区的最小包围矩形，约一半区域可能为空；扇区边界还会切断近车目标。以循环状态或整圈多尺度记忆补充上下文虽可行，却引入额外融合计算，并可能破坏上下文与当前目标之间的空间关系。

论文将端到端延迟定义为 \(50/n\) ms 的数据采集时间加算法总运行时间。这比只报告网络推理时间更适合旋转雷达：切分越细，采集等待越短，但单扇区可见范围也越窄，因而真正问题是寻找精度、上下文和延迟的联合最优点。

## 方法总览

**Multi-Scale Context Padding** 利用极坐标展开图在 \(\theta\) 方向上的天然邻接性。Trailing-Edge Padding 在主干每次卷积前，用当前扫描中前一扇区的边缘特征替换零填充；Bidirectional Padding 进一步聚合上一整圈的多尺度特征，经自车运动补偿后，从“未来方位”对应的上一圈扇区填充当前扇区前缘。它只复制卷积感受野需要的少数列，而非保存并融合完整大特征图。

在 32 扇区时，裸 Polar Pillars 的 PQ、mIoU、mAP、NDS 分别为 60.3、67.1、46.7、55.8；双向填充后分别达到 68.3、73.1、52.4、60.0。相对既有流式方法，论文报告 PQ 与 mIoU 分别提高 6.7 和 6.6 个点，说明边缘填充在极窄视野下尤其有效。

## 方法详解

**Feature Undistortion** 针对同一物体在极坐标图中近端被拉宽、远端被压缩的问题。它用位置编码 \((r,\cos\theta,\sin\theta,x,y)\) 生成位置相关的权重与偏置，使普通卷积近似从极坐标邻域向笛卡尔柱位置做双线性采样；辅助网络与主网联合训练，推理时每个位置的参数固定，因此不需在线运行辅助网络。该模块仅用于中心热图预测。

**Range Stratified Convolution & Normalization** 针对中心偏移统计随距离和方位变化的问题：极坐标偏移范围约为 \([-2,2]\)，标准差 0.64，明显大于笛卡尔柱的 0.28。模型按距离区间使用独立卷积核，并只在对应距离区域内归一化；分层卷积与归一化用于中心偏移回归，分层归一化还作用于检测头共享卷积。

## 实验与证据

训练采用分类 Focal Loss，边界框、方向和速度采用 L1 Loss，语义分割采用加权交叉熵与 Lovász-Softmax。增强包括 CBGS 类平衡采样、横纵轴翻转、0.95–1.05 缩放、绕 \(z\) 轴旋转及平移；不使用数据库采样。所有运行时间在单张 V100、PyTorch 环境测量。

重实现 Han et al. 与 STROBE 时统一主干和输入分辨率，并移除 STROBE 的 HD 地图分支。STROBE 的 Feature Warp 在速度估计上明显弱于十帧点级运动补偿：其平均速度误差为 0.607 m/s，而 Cartesian Pillars 的 Point Warp 为 0.358 m/s。

## 对 YOLO-Agent 的启发

若将该思路接入 YOLO-Agent，代理不应把任务抽象为一般点云检测，而应显式维护“扫描编号、扇区编号、前一扇区多尺度缓存、上一圈运动补偿特征”四类状态。模型规划应固定执行顺序：点级时间补偿 → 极坐标动态柱化 → 每层边缘上下文填充 → 联合检测与分割 → stateful-NMS → 整圈全景融合。

代理还应区分在线输出与延迟输出：检测和语义结果可随扇区产生；全景实例编号依赖同一扫描全部检测框，不能伪装成严格即时输出。部署策略可根据延迟预算选择 \(n\)，而不是默认使用越多扇区越好。

## 优点

复现实验必须设置三组纸面一致的控制变量：第一组固定主干、输入尺寸和十帧累积，仅比较 Cartesian Pillars 与 Polar Pillars；第二组固定 Polar Pillars，比较无填充、单向填充、双向填充；第三组在 \(n=1\) 下分别开关 Feature Undistortion 与 Range Stratified Conv/Norm。所有组同时报告 mAP、NDS、mIoU、PQ、算法运行时间及含 \(50/n\) ms 采集时间的端到端延迟。

## 局限

具体失败判据：在 \(n=32\) 时，若双向填充不能同时超过裸 Polar Pillars 的 PQ 60.3、mIoU 67.1、mAP 46.7，或若特征去畸变与距离分层联合后 mAP 未达到 50.3、额外耗时超过 0.5 ms，则核心结论未复现。还应检查缓存是否严格来自前一扇区和上一圈对应方位，任何未来扇区泄漏都会使流式评估无效。

## 评分

主要局限是全景融合仍需等待整圈检测框，因而并非所有输出都获得同等流式延迟；双向填充依赖上一圈特征与准确自车运动补偿，快速动态目标可能留下时序错位。实验以模拟切分的 nuScenes 为主，尚未验证真实传感器传输抖动、丢包以及不同扫描模式。尽管如此，该工作清楚证明：顺应旋转雷达的极坐标几何，比在笛卡尔矩形上堆叠复杂记忆模块更可能获得稳定的精度—延迟折中。
