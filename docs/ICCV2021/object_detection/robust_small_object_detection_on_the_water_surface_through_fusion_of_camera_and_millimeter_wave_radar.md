---

# Robust Small Object Detection on the Water Surface Through Fusion of Camera and Millimeter Wave Radar
title: "Robust Small Object Detection on the Water Surface Through Fusion of Camera and Millimeter Wave Radar"
description: "ICCV 2021 论文笔记：RISFNet 通过 RPDM、时序自注意力与全局通道注意力，融合相机和毫米波雷达，实现水面小目标的鲁棒检测。"
tags:
  - ICCV2021
  - 小目标检测
  - 多模态融合
  - 毫米波雷达
  - 水面无人艇
  - RISFNet
---

## 一句话总结

该工作面向内河与湖泊中的无人水面艇，以漂浮塑料瓶为检测目标，提出 **Radar-Image Spatiotemporal Fusion Network（RISFNet）**。完整数据流是：AWR1843 雷达原始回波经 Range FFT、Doppler FFT、CFAR 检测和 DOA 估计生成含位置、速度与能量的点云；利用 IMU 俯仰角执行位置补偿投影；再把点云转换为三通道 **Radar Point Density Map（RPDM）**。连续三帧 RPDM 进入 VGG-13 雷达分支，当前 RGB 图像进入 CSPDarknet53 图像分支，雷达特征依次经过 Temporal Position Encoding 和 Self-Attention，随后与图像特征在三个尺度上通过 Global Attention 融合，最后送入 FPN 完成大、中、小目标预测。

作者采集了真实内陆水域浮瓶数据：相机分辨率为 1280×720、15 Hz，IMU 为 10 Hz，77 GHz FMCW 雷达为 10 Hz，最大探测距离 30 m、距离分辨率 0.04 m。原始同步数据共 12000 帧，降采样后保留 1895 帧，含 3164 个标注目标，其中 1946 个目标面积小于 32×32 像素；训练集与测试集按 4∶1 划分。RISFNet 达到 **AP35 90.05%、AP50 75.09%**，超过 YOLOv4 的 78.46%/57.04%、EfficientDet 的 78.62%/58.52%、Faster R-CNN 的 77.35%/57.58%，也超过融合方法 CRF-Net 的 79.63%/57.74% 和 Li 等方法的 85.28%/64.64%。

消融结果明确说明增益来源：完整 RISFNet 为 90.05%/75.09%；仅用单帧雷达降至 88.34%/68.83%，移除时序位置编码降至 89.72%/72.24%，移除 Self-Attention 降至 88.72%/71.38%，移除 Global Attention 降至 88.95%/70.40%，而把双分支改成单一共享骨干仅有 82.81%/63.68%。RPDM 同样不是简单的雷达可视化：完整三通道 RPDM 达到 90.05%/75.09%，只保留密度通道时仅为 82.48%/63.93%。

## 研究背景与问题

水面小目标检测同时受到强反光、岸边建筑与植被倒影、目标像素过少以及波浪杂波影响。视觉传感器语义丰富，却容易在过曝、暗光和远距离条件下丢失瓶体；毫米波雷达不敏感于光照且探测距离较远，但浮瓶雷达散射截面低，点迹会闪烁、断续，并混有随机水面杂波。因此，问题不是把两种输入直接拼接，而是先稳定雷达的时空表示，再让网络依据传感器质量自适应分配权重。

## 方法总览

位置补偿投影利用固定相机高度 \(h\)、IMU 俯仰角 \(\theta\) 和雷达纵向距离 \(y\)，以  
\[
z_r=\frac{h}{\cos\theta}+y\tan\theta
\]
修正雷达点的高度。它针对水面航行中相机视角持续变化、毫米波雷达高度坐标不准确的问题，比直接透视投影和固定高度投影更贴合图像目标位置。

RPDM 将每个投影点用高斯核扩散，并分别编码距离 \(r\)、多普勒速度 \(v\) 和能量 \(p\)，形成 \(3\times H\times W\) 输入。实验采用 101×101 高斯核、方差 30。相较二值稀疏图，密度表示提供了连续空间分布与更丰富的梯度，同时保存雷达物理属性。

## 方法详解

雷达分支对过去三帧分别提取 52×52、26×26、13×13 多尺度特征。Temporal Position Encoding 用与帧序相关的正弦权重标识时间距离，避免早期雷达帧与当前图像之间的空间误差被等同处理。各帧随后经过独立 MLP、残差连接和通道压缩，再拼接成聚合雷达特征。

Self-Attention 的任务是增强跨帧反复出现的真实目标点迹并削弱随机杂波；Global Attention 则对图像和雷达特征分别执行全局池化，并通过共享 MLP、ReLU 与 Sigmoid 产生通道权重。当相机亮度异常或雷达 CFAR 阈值导致点云过密、过疏时，融合比例可随输入质量调整。

## 实验与证据

训练沿用 YOLOv4 的 CIoU 定位损失、置信度损失与分类损失，图像骨干使用 VOC 预训练 CSPDarknet53。网络训练 100 个 epoch，优化器为 Adam，批量大小为 4，StepLR 的 step size 为 1、gamma 为 0.9；部署于 Nvidia Jetson TX2 时约为 6 FPS。

在 nuScenes 的同一 mini-dataset 设置下，RISFNet 的 mAP 为 28.25%，高于对比融合方法的 24.3%。平台速度由采集时最高 2 m/s 模拟提升至 4 m/s 后，AP35 仍为 89.98%；单独测试波浪场景时为 89.22%，说明方法并非只在平静水面有效。

## 对 YOLO-Agent 的启发

可把 RISFNet 视为 YOLO 检测头之前的多模态特征适配器：保留 YOLO 的多尺度预测与损失函数，将 RGB backbone、轻量雷达 backbone、时序雷达净化模块和质量感知融合模块解耦。若用于 YOLO-Agent，控制策略不应只决定“是否调用雷达”，还应读取 CFAR 阈值、点数密度、图像亮度和跨帧稳定度，为各尺度动态选择雷达权重。

建议设置四组论文特定控制组：纯 RGB YOLOv4；RGB 加单帧雷达稀疏图；RGB 加三帧 RPDM 但无注意力；完整三帧 RPDM、时序编码、自注意力和全局注意力。主指标固定为 AP35、AP50，同时报告 Jetson TX2 FPS；压力测试分别扫描 CFAR 阈值、图像亮度、4 m/s 模拟航速与波浪子集。具体失败判据为：任一退化条件下融合模型 AP35 不再高于纯视觉 YOLOv4，或完整模型相对“无 Global Attention”不能恢复论文所示的鲁棒性优势。

## 优点

完整 RPDM 相较雷达稀疏图，在 AP35/AP50 上由 87.12%/69.58% 提升至 90.05%/75.09%；相较 PointNet++ 的 87.64%/69.55%也更优。距离密度单通道达到 88.80%/72.20%，明显强于速度通道的 83.67%/64.01%和能量通道的 84.59%/66.85%，说明距离及空间分布是主要信息，而速度、能量提供互补增益。

## 局限

局限首先是数据集规模较小且类别单一，仅围绕内陆水域漂浮瓶展开；论文未报告跨季节、极端天气、不同雷达型号或多类别障碍物上的泛化。其次，位置补偿依赖相机高度近似固定和可靠的 IMU 俯仰角，剧烈横滚、安装偏差及同步误差可能破坏投影。三帧融合还可能在高速运动、急转弯或目标快速横移时引入重影。

## 评分

- **论文链接**：[CVF 官方论文页面](https://openaccess.thecvf.com/content/ICCV2021/html/Cheng_Robust_Small_Object_Detection_on_the_Water_Surface_Through_Fusion_ICCV_2021_paper.html)
- **官方代码**：论文正文声称发布代码与雷达—视觉数据集，但所给官方论文提取中未出现作者 GitHub URL，因此无法确认官方代码仓库；请以[作者论文页面及其补充材料入口](https://openaccess.thecvf.com/content/ICCV2021/html/Cheng_Robust_Small_Object_Detection_on_the_Water_Surface_Through_Fusion_ICCV_2021_paper.html)为准。
