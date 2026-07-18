---

# ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector
title: "ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector"
description: "面向小目标检测的尺度解耦特征蒸馏与跨尺度预测校正方法，通过训练期知识迁移提升紧凑检测器，且不增加推理开销。"
tags:
  - 小目标检测
  - 知识蒸馏
  - 尺度感知
  - 特征蒸馏
  - CVPR2023
---

- **论文链接**：https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html

**???????**: [?ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector?????????????????????????](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html)
- **官方代码**：论文抽取内容中未出现作者 GitHub URL，因而没有可确认的官方代码仓库；可使用上述官方论文页获取论文及补充材料。

## 一句话总结

方法首先在教师与学生骨干网络最后阶段取得特征 \(Z^T\) 和 \(Z^S\)，经适配器对齐维度后送入 Scale-Decoupled Feature（SDF）。SDF 设置三个并行 ResNet 式残差分支，每个分支均由 \(1\times1\)、\(3\times3\)、\(1\times1\) 卷积组成，但中间卷积的膨胀率分别为 \(\{1,2,3\}\)：小膨胀率侧重细粒度小目标，大膨胀率覆盖更大感受野。各分支输出经 MLP 展平并拼接，教师与学生只使用一个 \(L_2\) 特征损失匹配；三个同构分支共享权重，以降低训练显存，而学生端完整镜像教师端的尺度解耦结构。

Cross-Scale Assistant（CSA）不直接把教师的噪声框交给学生，而是联合教师、学生特征构造可学习的 Multi-Scale Cross-Attention（MSC）。按文中公式，学生特征形成查询 \(Q_i\)，教师特征经过不同下采样率 \(r_i\) 的 MSC 聚合形成键 \(K_i\) 与值 \(V_i\)，值分支再加入深度卷积位置投影；随后计算多头注意力，使不同头覆盖不同尺度，而不是反复集中于显著大目标。CSA 后接分类与回归分支并由真实标签优化，训练蒸馏时，学生模仿的是 CSA 校正后的分类和框知识，而非原始教师输出。

实验覆盖 COCO 2017 与以航拍小目标为主的 VisDrone，包含 Faster R-CNN、Cascade R-CNN、RetinaNet、FCOS、RepPoints、GFL，以及 ResNet50、MobileNetV2 等学生，主要报告 AP、AP50、AP75、APS、APM、APL；教师通常采用 ResNet101 和两倍输入分辨率。COCO 上 RetinaNet-ResNet50 的 AP/APS 从 37.4/20.0 提升到 42.8/25.9，FCOS 的 APS 从 21.9 提升到 29.1。消融中，普通 CSA 得到 40.2 AP、23.5 APS，加入 MSC 后达到 41.6 AP、24.8 APS，即分别再增 1.4 和 1.3。

## 研究背景与问题

核心问题并非一般意义上的“教师不够准确”，而是小目标对尺度混叠和框坐标扰动极其敏感。小目标只占少量像素，其特征容易被大面积背景及更大实例污染；若所有尺度共享单一嵌入，学生很难从中提取专属于小目标的知识。同时，教师框的少量像素偏移可能显著改变小目标 IoU，使常规输出蒸馏把错误监督进一步放大。

论文的扰动实验直接验证了第二点：RetinaNet 基线为 36.9 AP、21.2 APS；采用输出蒸馏后为 38.1 AP、21.9 APS，但给教师框沿对角线加入 6 像素扰动时降至 37.7 AP、21.1 APS，12 像素扰动时进一步降至 37.0 AP、19.9 APS。总体 AP 尚未完全崩溃时，APS 已低于无蒸馏基线，说明小目标蒸馏必须显式处理框噪声。

## 方法总览

学生总目标为：

\[
L_{\text{total}}=\alpha L_{\text{feat}}+\beta L_{\text{cls}}+\gamma L_{\text{bbox}}+L_{\text{det}}
\]

其中 \(L_{\text{feat}}\) 来自 SDF 拼接嵌入的距离，\(L_{\text{cls}}\) 与 \(L_{\text{bbox}}\) 蒸馏 CSA 的分类和回归输出，\(L_{\text{det}}\) 是原检测器损失。两阶段模型使用 \(\alpha=0.07,\beta=0.5,\gamma=0.2\)；单阶段模型使用 \(\alpha=0.01,\beta=0.2,\gamma=0.05\)。CSA 采用 Adam，初始学习率 \(3\times10^{-4}\)，权重衰减 \(10^{-4}\)。

## 方法详解

CSA 依赖尚未稳定的学生特征，若从随机初始化阶段同步学习，辅助监督本身会漂移。因此论文先让学生预热 30k 次迭代；在早期还冻结学生骨干，1×训练日程冻结 10k 次，2×日程冻结 20k 次。每次迭代先更新 CSA，再更新学生。SDF、CSA 和高分辨率教师均只服务于训练，测试时保留原学生检测器，不增加推理计算路径。

## 实验与证据

COCO 的同分辨率公平比较中，GFL-ResNet50 从 40.1 AP、23.3 APS 提升到 42.5、25.9，超过 LD 的 42.1、24.5；RetinaNet 从 37.4、20.0 提升到 41.7、24.8；FCOS 从 38.5、21.9 提升到 42.7、27.8。VisDrone 上 RetinaNet-MobileNetV2 从 21.73 AP 提升至 26.08，接近更重的 RetinaNet-ResNet50 基线 26.21，并明显高于 ZoomInNet 的 22.49。

SDF 消融显示，独立分支权重与共享权重均为 41.6 AP，APS 分别为 24.9 和 24.8，几乎无总体精度差异，因此共享参数更具训练效率。不同容量学生也保持收益：ResNet18 从 35.8/18.9 AP/APS 提升至 37.2/20.1；ResNet34 从 38.9/21.5 提升至 40.8/23.1。

## 对 YOLO-Agent 的启发

在 YOLO-Agent 中，可把教师设为更大骨干或高分辨率版本，学生保持目标部署结构。SDF 应接在教师与学生骨干末级或进入颈部前的对应特征上，并保留膨胀率 \(\{1,2,3\}\) 的共享权重分支；CSA 则应逐个学生金字塔尺度读取学生查询和教师多尺度键值，再输出与 YOLO 分类、框回归头兼容的监督。若检测头包含独立 objectness，应将其视作额外分类监督，而不能用未经验证的权重直接并入论文的三个系数。

## 优点

Harness 必须至少设置四组：原始学生、常规输出蒸馏、仅 SDF、仅 CSA，以及完整 SDF+CSA+MSC；另设“CSA+普通交叉注意力”以隔离 MSC 的贡献，并比较 SDF 独立权重和共享权重。教师架构、输入分辨率、训练日程、增强和随机种子必须一致，避免把高分辨率教师收益误记为模块收益。

## 局限

主指标使用 COCO AP 与 APS，同时记录 AP50、AP75、APM、APL；航拍数据还应报告 AP、AP50、AP75、AR1、AR10、AR100、AR500。工程侧记录学生推理 FPS、参数量和 FLOPs，确认训练期模块已被移除。另复现 0、6、12 像素教师框对角线扰动，观察 APS 是否比总体 AP 更快恶化。

## 评分

具体失败判据：完整方法在相同教师和训练预算下，APS 未同时超过原始学生与普通输出蒸馏；MSC 相比普通交叉注意力不能带来可重复的 APS 增益；或导出学生后 FPS、参数量、FLOPs 任一增加，均说明实现偏离论文。若总体 AP 上升但 12 像素扰动下 APS 仍低于无蒸馏基线，则 CSA 没有真正过滤小目标框噪声，也应判定复现失败。
