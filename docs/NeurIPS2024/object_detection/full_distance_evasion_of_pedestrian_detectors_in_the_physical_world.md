---
title: Full-Distance Evasion of Pedestrian Detectors in the Physical World
publication: NeurIPS
year: 2024
description: "梳理 FDA 从远距成像转换、跨尺度频率优化到服装部署的物理攻击链，并给出面向行人检测器的全距离红队验收标准。"
paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/b99d6cc40b05809c3d84b57a165448cd-Abstract-Conference.html
code: https://github.com/zhicheng2T0/Full-Distance-Attack
tags: [物理对抗攻击, 行人检测, 距离鲁棒性, 成像模拟, 对抗服装]
---

# Full-Distance Evasion of Pedestrian Detectors in the Physical World

## 一句话总结

FDA 用仅 15 个可学习参数的 Distant Image Converter 重建远距离成像退化，再以 Multi-Frequency Optimization 分离远近距离所需的低频与高频图案，使实体补丁和服装在 3.5–41 米范围内持续逃逸行人检测器。

## 研究背景与问题

Adv-Patch、Adv-Tshirt、TCA 等物理攻击通常在近距离有效，距离增大后攻击率快速下降。论文指出原因并非检测器天然具备远距鲁棒性，而是攻击训练中的“缩小后粘贴”与真实远距照片存在越来越大的外观鸿沟：大气散射改变颜色，相机抗混叠滤镜和成像芯片改变空间频率，机内效果滤镜继续改变色彩与纹理。

即便能正确模拟远距外观，优化仍有第二个冲突：近距离图像保留高低频信息，远距离只剩低频结构；若远近攻击需要不同低频图案，同一补丁的梯度会互相抵消。FDA 因此同时解决成像模拟错误与跨距离频率竞争。

## 方法总览

完整链路为：对抗图案经 **Multi-Scale Cropping（MSC）** 取子块并贴到近距行人 → 用人体掩码抠出前景 → **Distant Image Converter（DIC）** 按指定距离、天空光和浑浊度生成远距外观 → 与随机背景合成 → 送入目标检测器，以置信度和 IoU 联合攻击损失更新图案。训练覆盖 4、8、14、20、26、34、40 米；**Two-Stage Optimization（TSO）** 先偏向远距学习低频结构，再偏向近距学习高频纹理，并约束第二阶段低频分量不要漂移。

## 方法详解

**DIC 的独有数据流**由 Atmospheric Perspective、Camera Simulation、Effect Filter 三段组成。第一段依据距离、天空光 RGB 与浑浊度模拟目标颜色向天空光指数偏移；第二段串联两个逐通道卷积，分别模拟抗混叠滤镜和矩形成像传感器，卷积核由可微几何函数生成，模糊强度随距离缩放；第三段模拟亮度、饱和度、锐化、曝光、对比度等机内滤镜。作者打印 45 张训练图、9 张测试图，在 7 个距离、5 天拍摄配对图，以 MSE 拟合 15 个参数。

**MFO 的独有机制**不是普通 EOT。MSC 在近距训练时只裁取较小图案区域，使近距梯度难以整体改写补丁，保护服务于远距的低频布局；TSO 第一阶段增加远距样本比重，第二阶段增加近距样本并加入低频保持损失。DIC 参数还接受 EOT 扰动，天空光与浑浊度随机采样，因此攻击同时覆盖距离、背景、天气和成像偏差。

**实体攻击数据流**使用 1100 张 INRIA、PennFudan、COCO 行人图，加 1000 张网络行人图和既有背景集优化补丁；测试则由 5 名受试者、不同地点和不同手机实拍，每个距离约 30 张、3 次试验。攻击成功率 ASR 定义为 `1-TP/GT`，置信度与 IoU 阈值均为 0.5，且测试距离与训练距离错开，避免仅验证离散点记忆。

## 实验与证据

YOLOv5 白盒纸质补丁平均 ASR 为 **74%**，NAP 与 T-SEA 分别为 19% 和 42%；换 Huawei Nova 11 SE、OPPO A9 后仍有 68% 与 72%。不加 DIC/MFO 为 22%，加 DIC 到 65%，再加 MFO 到 74%。DIC 测试 L2 误差约 0.11，而朴素缩放和七个独立 FCN 随距离增大到约 0.16。

更细消融中，依次移除效果滤镜、相机模拟、大气透视后 DIC 平均误差从 0.11 升至 0.12、0.13、0.14；完整 MFO 数字攻击 ASR 为 73%，去 MSC 为 66%，再去 TSO 为 56%。服装正背面攻击 YOLOv5 达 76%，TCA 为 37%；侧面为 61%，比 TCA 高 15 点。单模型补丁跨黑盒迁移有限，但联合 Mask R-CNN ResNet/Swin 优化后，对六个黑盒模型平均 ASR 均超过 75%。

## 对 YOLO-Agent 的启发

1. 鲁棒性 agent 必须把物理距离建模为完整成像链，而不是只做 resize、blur 和颜色抖动。
2. 不同尺度可能要求冲突的频率特征，可采用分阶段课程和频带保持约束诊断梯度冲突。
3. 红队评估要区分白盒、单源迁移和集成迁移，不能用单个近距补丁代表真实安全性。

**YOLO-Agent Harness（物理攻击）**：固定补丁面积、打印材料和拍摄路线，先锁定**对照组**为无攻击行人、Adv-Tshirt、增强 EOT、仅 Distant Image Converter（DIC）、DIC+Multi-Scale Frequency Optimization（MSC）以及完整 DIC+MSC+TSO；白盒 YOLOv5 与迁移到 YOLOv8、Deformable DETR、Mask R-CNN 的结果分开统计。验收**指标**采用 3.5–41 米七个距离段的 ASR、目标像素高度分桶 ASR、不同相机/朝向的最差组 ASR、检测置信度、误触发补丁框比例和 DIC 重建 L2 误差。以下任一项构成**失败判断**：24 米后连续两个距离段不优于强 EOT，换相机或侧身使 ASR 跌落超过 10 个百分点，降低 IoU 阈值后收益主要来自补丁自身被框成“小行人”，或数字模拟优势无法在三次实体试验中复现。

## 优点

- 从光学和相机硬件出发解释远距失效，根因清晰。
- 物理测试距离宽、设备和检测器多，消融能对应两个核心假设。
- 同时提供补丁、服装和黑盒迁移证据。

## 局限

- DIC 需针对目标相机采集打印配对数据，跨镜头自动校准尚未解决。
- 攻击训练和服装制作成本高，人体姿态、运动模糊和更复杂遮挡覆盖有限。
- 工作具有明显滥用风险，且 YOLOv5 在低 IoU 阈值下可能被图案诱发的小框影响评估。

## 评分

- 创新性：9/10
- 技术完整性：9/10
- 实验说服力：9/10
- 工程可迁移性：7.5/10
- 对 YOLO-Agent 价值：9/10
