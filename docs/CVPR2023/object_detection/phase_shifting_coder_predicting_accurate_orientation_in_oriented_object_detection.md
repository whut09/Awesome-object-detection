---

# Phase-Shifting Coder: Predicting Accurate Orientation in Oriented Object Detection
title: "Phase-Shifting Coder：面向旋转目标检测的精确方向预测"
description: "解析 PSC 与双频 PSCD 如何以相位编码统一处理角度边界不连续和类正方形歧义，并总结其在 DOTA、HRSC、OCDPCB 上的高 IoU 定位表现与复现实验设计。"
tags:
  - 旋转目标检测
  - 航拍图像
  - 角度编码
  - 相位展开
  - PSC
  - PSCD
  - CVPR2023
---

- **论文链接**：[CVPR 2023 官方论文页](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Phase-Shifting_Coder_Predicting_Accurate_Orientation_in_Oriented_Object_Detection_CVPR_2023_paper.html)
- **官方代码**：[open-mmlab/mmrotate](https://github.com/open-mmlab/mmrotate)

## 一句话总结

该方法不直接回归旋转角，而是把长边 90° 定义下的方向角 \(\theta\in[-\pi/2,\pi/2)\) 映射为主相位 \(\varphi=2\theta\)，再由 **Phase-Shifting Coder（PSC）**生成长度为 \(N_{\text{step}}\) 的连续余弦编码。检测头的数据流为：Backbone/Neck 特征进入分类、框回归与角度卷积分支；角度分支输出经 \(2\operatorname{sigmoid}(\cdot)-1\) 限制到 \([-1,1]\)，与标注角度生成的 phase-shifting patterns 计算 L1 损失；推理时再以加权正余弦和及 `atan2` 解码主相位，最后恢复 \(\theta=\varphi/2\)。因此 \(-\pi/2\) 与 \(\pi/2\) 在编码空间自然连续，不需要离散角度类别。

为同时处理 180° 边界周期与 90° 类正方形周期，**Dual-Frequency Phase-Shifting Coder（PSCD）**并行构造 \(\varphi_1=2\theta\) 和 \(\varphi_2=4\theta\)，分别编码为 \(X_1、X_2\)，角度分支输出长度由 \(N_{\text{step}}\) 增至 \(2N_{\text{step}}\)。解码后以 \(\varphi_1\) 作为绝对相位、\(\varphi_2\) 作为包裹相位，计算 \(\delta=\cos\varphi_1\cos(\varphi_2/2)+\sin\varphi_1\sin(\varphi_2/2)\)；若 \(\delta<0\)，令最终相位为 \(\pi+\varphi_2/2\)，否则取 \(\varphi_2/2\)，并归一化回 \([-\pi,\pi)\)。这一步是真正针对类正方形歧义的双频相位展开，而非普通角度平滑。

实验覆盖 **DOTA、HRSC 与 OCDPCB**，控制骨干为 FCOS-R50、RetinaNet-R50、YOLOv5s 和 YOLOv5m，基线为离散角度编码 **CSL** 与高斯分布损失 **KLD**，主要指标是 mAP50、mAP75 和 mAP50:95。关键消融显示，在 RetinaNet+PSCD 上将 \(N_{\text{step}}\) 从 3 改为 4，DOTA mAP75 从 41.17 升至 41.82，但 mAP50 从 71.09 降至 70.96；HRSC 三种步数的 mAP50 仅在 85.46–85.53 间波动，说明步数影响有限，论文最终推荐计算量更低的 \(N_{\text{step}}=3\)。

## 研究背景与问题

旋转框参数化中的问题本质上不是数值回归难，而是一个输入可能对应多个等价角度表示。普通 L1/Smooth L1 会把几何上相同的 \(-90°\) 与 \(90°\) 视为相距 180°；近正方形框旋转 90° 后仍近似同一目标，却会因宽高交换和角度跳变产生很大损失。CSL 用圆形平滑标签缓解边界问题，但引入角度量化、类别通道和数据集相关平滑半径；KLD 将旋转框转为二维高斯分布，能统一等价框，却在高 IoU 阈值下可能缺少足够精细的方向约束。

PSC 的核心判断是：目标的旋转对称周期可以映射为相位频率。若目标旋转 \(s\) 弧度后等价，则一般形式为 \(\varphi=k\theta\)，其中 \(k=2\pi/s\)。普通矩形对应 \(k=2\)，正方形对应 \(k=4\)，而飞机航向等 360° 方向任务可取 \(k=1\)。多类目标具有不同旋转周期时，可用多频率相位共同消歧，这比为每种角度定义单独设计边界规则更统一。

## 方法总览

单频编码定义为：

\[
x_n=\cos\left(\varphi+\frac{2n\pi}{N_{\text{step}}}\right),\quad
n=1,\ldots,N_{\text{step}}.
\]

解码通过各相移样本与对应正弦、余弦基函数的加权和恢复相位，并必须使用 `atan2` 保留象限。编码是连续、可微且无量化的；角度误差会平滑地反映到多个通道，而不会像角度分类那样受分桶精度限制。

总损失为 \(L=w_1L_{\text{cls}}+w_2L_{\text{box}}+w_3L_{\text{ang}}\)，默认 \(w_3=0.2w_1\)。PSC 只替换 \((x,y,w,h,\theta)\) 中的角度预测，不改变中心点、宽高及检测器原有分类逻辑，因此能附加到 anchor-based、anchor-free 与 YOLO 系列检测头。

## 方法详解

DOTA 使用 2,806 张航拍图像和 15 类共 188,282 个实例，训练时切为重叠 200 像素的 \(1024\times1024\) 图块。HRSC 含 436/181/444 张训练、验证、测试船舶图像；OCDPCB 含 445 张训练和 191 张测试 PCB 图像，分辨率为 \(1280\times1280\)。后者沿用 HRSC 超参数而不专门调参，用于检验迁移到陌生场景时的稳定性。

DOTA 上，YOLOv5s+PSC/PSCD 的 mAP75 为 47.56/51.50，双频带来 **3.94 个百分点**；YOLOv5m 上由 50.50 增至 54.10，提升 **3.60 点**。PSCD 相比 KLD 的 DOTA 平均 mAP75 高 2.36 点，相比 CSL 平均 mAP50、mAP75 分别高 1.15、2.59 点，优势集中在高质量旋转框而非单纯召回。

## 实验与证据

HRSC 中 FCOS+PSC 达到 90.06 mAP50、78.56 mAP75，PSCD 的 mAP75 为 79.20；但 RetinaNet 上 PSC 的 61.30 高于 PSCD 的 59.57。船舶很少呈类正方形，额外频率反而可能引入不必要的解包错误，因此 PSCD 并非始终优于 PSC。

OCDPCB 上 RetinaNet+PSC 达到 77.35/65.61/57.58 的 mAP50/mAP75/mAP50:95，而 KLD 为 76.30/54.27/49.06；论文统计 PSC 的 mAP75 平均领先 KLD 9.83 点，显示无需数据集专属平滑半径的优势。RetinaNet-R50 推理时，PSC 为 43.1 ms、36.79M 参数，PSCD 为 43.4 ms、36.88M；相较基线 42.9 ms，额外成本很小，也明显低于 CSL 的 46.3–50.5 ms。

## 对 YOLO-Agent 的启发

在 YOLO-Agent 中可把角度标量输出替换为 PSC 的 3 通道，或 PSCD 的 6 通道；匹配与类别分支保持不变，标注分配后按目标角度在线生成余弦编码。推理阶段应在 NMS 前解码角度，再组成旋转框执行 rotated NMS。若任务以车辆、舰船等细长目标为主，默认 PSC；若包含储罐、球场、近方形建筑等 90° 对称实例，则启用 PSCD。Agent 应把“是否存在大量类正方形目标”和“是否重视 AP75”作为结构选择信号，而不是自动选择参数更多的双频版本。

## 优点

复现 Harness 设置三组严格控制实验：①同一 FCOS-R50 或 RetinaNet-R50、同一增强和训练轮次，仅替换 Rotated、CSL、KLD、PSC、PSCD；②固定 PSC/PSCD，比较 \(N_{\text{step}}=3,4,5\)；③固定 YOLOv5s/m，比较 PSC 与 PSCD。DOTA 训练 12 epoch，YOLO 训练 120 epoch，HRSC/OCDPCB 训练 72 epoch；学习率从 \(10^{-3}\) 降至 \(10^{-5}\)，禁用 SWA 与多尺度测试。

## 局限

Harness 必须报告 mAP50、mAP75、mAP50:95、单图推理时间和参数量，并按“类正方形类别”与“细长类别”拆分 AP。成功标准不是仅提高 mAP50，而是在相同检测器下 PSC 相对 CSL/KLD 稳定提高 mAP75，且 PSCD 在 DOTA-YOLOv5s 上应复现接近 3.94 点的 PSC→PSCD 增益；同时监测边界附近角度样本与宽高比接近 1 的样本。

## 评分

具体失败判据：若 PSCD 在含方形目标的 DOTA 上连续多次训练均不能超过 PSC 的 mAP75，或增益低于 1 点且角度解包错误集中在 \(\pm\pi\) 附近，则实现可能存在相位范围、`atan2` 象限或归一化错误；若 HRSC/OCDPCB 上 PSCD 低于 PSC，则不应立即判为实现失败，因为论文已观察到该现象。若 PSC 的推理增量显著超过 0.2 ms、参数增量超过约 0.08M，或输出仍出现固定角度分桶，则说明角度头没有按连续三步相移编码实现。
