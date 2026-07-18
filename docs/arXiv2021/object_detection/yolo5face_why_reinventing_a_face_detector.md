---
title: "[论文解读] YOLO5Face: Why Reinventing a Face Detector"
description: "YOLO5Face 通过五点关键点头、Wing Loss、P6 和 ShuffleNetV2 变体把 YOLOv5 适配到多尺度人脸检测。"
tags: ["arXiv 2021", "人脸检测", "YOLO5Face", "Wing Loss", "WIDER FACE"]
---

# YOLO5Face: Why Reinventing a Face Detector

**会议**: arXiv 2021  
**论文**: [arXiv](https://arxiv.org/abs/2105.12931)  
**代码**: [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)  
**任务**: 实时人脸与五点关键点检测

## 一句话总结

YOLO5Face 把人脸视作通用目标，在 YOLOv5 检测头旁增加五点 landmark 回归与 Wing Loss，并以 stem、较小 SPP 核、P6 stride-64 输出、大小脸增强和 ShuffleNetV2 backbone 覆盖服务器到移动端。

## 研究背景与问题

专用人脸检测器常引入复杂 anchor、上下文或级联结构，但人脸面临的尺度、遮挡、姿态和密集问题也存在于通用检测。YOLOv5 原版没有关键点监督，Focus 切片可能损失局部连续性，默认 P3–P5 对极大脸和极小脸的覆盖也不理想。论文验证是否只需针对人脸作有限、可解释的改造。

## 方法总览

输入首先通过 stride-2 stem convolution 代替 Focus；CSP backbone 与 PAN 融合 P3/P4/P5，配置可再增加 P6 输出。每个 anchor 预测 box、objectness、face class 和 5×2 landmark 坐标。框损失沿用 IoU 类目标，关键点采用 Wing Loss，在小误差区对数放大梯度、大误差区转为线性。轻量版把主干替换为 ShuffleNetV2。

## 方法详解

P6 的 stride 为 64，补充大感受野层，论文消融显示对 WIDER FACE mAP 有益；SPP 使用更小池化核以减少开销。增强策略不仅随机翻转，还在裁剪时考虑小脸，避免 tiny face 被当成无效区域。五点标签同时提供眼、鼻、嘴角几何，可帮助分类头学习姿态稳定特征。

## 实验与证据

WIDER FACE 的 Easy/Medium/Hard 三子集是主评测，论文逐项比较 Focus/stem、SPP 核、P6、landmark、增强和不同 backbone。多数 VGA 输入模型在三个子集达到当时有竞争力结果；ShuffleNetV2 版本强调移动速度，大型 YOLOv5x6 版本追求 Hard 子集精度。关键证据是同一检测框架可沿复杂度连续缩放，而非只给单个高分模型。

## 对 YOLO-Agent 的启发

应按人脸像素高度、偏航角、遮挡等级分桶，同时报告 WIDER AP、五点 NME、漏检率和移动端延迟。对照原 Focus、stem、加 P6、加 landmark 四级，并单独测试“关键点缺失/遮挡”样本。若 P6 只提高 Easy 大脸却拖慢 Hard 小脸，或 Wing Loss 使检测 AP 上升但 NME 恶化，应回退对应分支；移动端需确认 ShuffleNetV2 的 channel shuffle 没被后端拆成昂贵内存操作。

## 优点

- 用少量任务特定改造复用成熟 YOLO 流程。
- 同时输出框和五点，利于后续对齐识别。
- 模型族覆盖高精度与嵌入式需求。

## 局限

- WIDER FACE 不能代表所有红外、口罩和超广角场景。
- anchor 与 NMS 仍需专门调节。
- 五点标注成本高于纯框数据。

## 评分

- **创新性**: ★★★☆☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★☆
