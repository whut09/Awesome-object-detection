---
title: "The Translucent Patch: A Physical and Universal Attack on Object Detectors"
description: "通过贴在镜头上的半透明椭圆形区域扰动，对目标类别实施物理通用攻击。"
tags: ["CVPR 2021", "目标检测", "对抗攻击", "物理攻击", "YOLOv5"]
---

# The Translucent Patch: A Physical and Universal Attack on Object Detectors

**论文页面**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Zolfi_The_Translucent_Patch_A_Physical_and_Universal_Attack_on_Object_CVPR_2021_paper.html)  
**官方 PDF**：[CVPR 2021 Paper PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zolfi_The_Translucent_Patch_A_Physical_and_Universal_Attack_on_Object_CVPR_2021_paper.pdf)  
**官方代码**：论文正文与 CVF 官方页面未声明代码仓库，未发现可核验的官方实现链接。

## 一句话总结

The Translucent Patch 不在停车标志上贴扰动，而把由多个可微椭圆形状组成的 RGBA 贴片固定到相机镜头，通过目标置信度、目标框 IoU、非目标置信度保持和 Non-Printability Score 联合优化，使 YOLOv5 的 stop-sign AP 从 95.17% 降到 52.70%，实体贴片漏检率为 42.27%。

## 研究背景与问题

传统物理对抗贴片要求接触每个被攻击物体：隐藏多人或多个标志就要逐个布置。本文把攻击面转移到传感器，镜头上的一个固定半透明图案会叠加到所有帧、所有位置，因此必须同时压制图像中任意 stop sign 候选，又避免把 person、car、traffic light 等非目标类别全部破坏；这比只改变单一图像分类结果更受候选框数量和定位约束影响。

方法数据流为：**原始驾驶图像 → n 个椭圆 shape 的连续参数 `(xc,yc,r,shx,shy,γ)` → 距离衰减 alpha channel → affine positioning/shearing → RGBA patch → alpha blending 到整幅图像 → YOLOv5 objectness/class/box 输出 → 四项损失反向传播更新 shape 参数 → 打印到透明纸并贴在镜头上**。形状数量、最大透明度和半径范围人工限定，位置、半径、剪切与 RGB 颜色由 Adam 优化。

## 方法总览

每个 shape 由中心、半径、x/y shear、RGB 颜色和透明度定义。作者不用像素级噪声，而令 alpha 随归一化距离 `d(i,j)` 平滑衰减：中心接近 `αmax`，边缘接近 `αmax(1-s)`；`s=0.9、β=2.5、r∈[0.03,0.25]`。连续中心与 shear 通过二维 affine matrix 实现，使所有自由参数都能对检测损失求梯度。最终物理样品为约 **0.6×0.33 英寸**透明贴纸。

总损失是 `w1 L_target_conf + w2 L_IoU + w3 L_untargeted_conf + w4 L_nps`。`L_target_conf` 最小化 YOLO 的 `Pr(objectness)×Pr(target class)`；`L_IoU` 最小化目标预测框与 GT 的 IoU，既压分数也破坏定位；`L_untargeted_conf` 约束 clean 与 patched 图像中非目标类别置信度差；`L_nps` 把颜色拉向打印机可实现色集合。网格搜索得到权重 **0.74/0.15/0.10/0.01**。

## 方法详解

### 数字攻击与物理落地

数字阶段直接按 `patched=(1-α)·original+α·γ` 混合。自由参数用 Adam 更新，普通参数初始学习率 `5e-3`，半径为 `8e-4`。攻击目标不是让 stop sign 变成指定类别，而是让候选置信度低于检测阈值或使框位置失配。打印阶段使用 Xerox 6605DN 激光打印机，将贴片装到 Logitech C930 镜头；屏幕播放 LISA 视频模拟行驶，YOLOv5 模拟 ADAS，检测阈值为 0.4。

### 数据与评价协议

训练数据汇合 LISA（约 500 张 stop-sign 图）、MTSD（约 750）和 BDD（约 500），总计约 **1750** 张。BDD+MTSD 按 90/10 划训练与验证，LISA 独立作测试并提供视频帧。因 LISA/MTSD 只标交通标志，作者用 YOLOv5 给其他七类生成标签，因此非目标类别的 clean AP 被定义为 100%，这会使相关结果依赖教师自身预测。数字攻击用 AP，物理攻击用 `fooled objects / total objects`。

### 复现边界

贴片坐标固定在相机平面，不随 stop sign 框移动；若实现把 shape 裁到目标区域，就退化成普通 object-attached patch，失去“一次覆盖所有实例”的威胁模型。多个 shape 的 alpha 叠加还要保持论文的平滑距离衰减，否则硬边缘可能在数字域产生更强但不可打印的高频信号。损失必须从 NMS 前的 YOLO 网格候选取得目标 objectness 与类别概率，因为 NMS 后离散框集合不适合稳定反传；非目标保持项则比较 clean 与 patched 图像中的类别置信差。物理复现需要记录打印尺寸、透明介质、镜头相对位置与重装偏移，不能只报告一张成功帧。论文的 Tesla 演示使用纯色贴片，是动机实验，不应与后续优化八形状贴片的定量结果混为一组。

## 实验与证据

白盒攻击使用 8 个 shape、`αmax=0.4`，YOLOv5 stop-sign AP 从 **95.17% 降至 52.70%**，下降 42.47 个百分点；非目标类别 AP 为 **82.69%**。黑盒迁移中，同一贴片在 YOLOv2 上把 stop-sign AP 从 81.54 降至 57.36，非目标从 59.13 降至 54.92；在 Faster R-CNN 上从 94.31 降至 54.53，非目标从 78.31 降至 70.36，说明攻击并非只拟合 YOLOv5 的 head。

形状数量消融表明 n=3/5/7/10/15 时，stop-sign AP 为 91.15/77.45/65.01/53.11/**47.90**，但非目标 AP 同时从 95.44 降到 72.49。透明度消融更清楚地暴露攻击-可见性权衡：`αmax=0.1/0.3/0.5/0.7/0.9` 时目标 AP 为 93.85/70.13/51.75/38.61/**36.55**，非目标 AP 为 98.26/88.25/81.93/78.76/**70.45**。更强扰动并非免费。

物理实验中，优化贴片对 stop sign 的 fooling rate 为 **42.27%**，对其他类为 21.54%，即仍检测到约 78.46% 非目标实例；随机贴片仅骗过 20.57% stop sign。红色和青色整片虽然分别达到 93.3% 与 98.9% 目标漏检，却也让其他类漏检 82.7% 与 81.6%，近似遮挡镜头。专门优化的贴片因此取得更合理的定向性。

## 对 YOLO-Agent 的启发

YOLO-Agent 应把该论文转成传感器级鲁棒性测试器，而不是默认攻击模块：生成受打印约束的 RGBA shape，固定在图像坐标系而非目标坐标系，分别记录目标类抑制与非目标副作用。训练侧可加入镜头污渍/贴膜增强、跨帧固定扰动检测或“clean 与 patch 非目标一致性”正则；部署侧应检测相机平面上跨场景不动的半透明模式。

**专属 Harness**：白盒对照为 CLEAN、随机同形状贴片、纯 RED、纯 CYAN、优化 PATCH；黑盒在 YOLOv2/Faster R-CNN 或不同 YOLO 版本测试；物理组固定摄像头、屏幕、距离与照明，重复装卸贴片。观测 stop-sign AP/漏检率、其他七类 AP/漏检率、不同 n 与 `αmax` 的 Pareto 曲线、跨模型迁移和跨帧固定性。防御通过标准：相对无防御模型，PATCH 的目标漏检率至少降低一半，同时 clean AP 与非目标 AP 下降均不超过 2 点；失败判断：只对数字 alpha blending 有效、换打印机/位置即失效、通过整体降低置信阈值“恢复”目标却引入大量误报，或防御仅识别红/青纯色而漏掉优化 shape。

## 优点

- 攻击从物体表面转到镜头，一个贴片可作用于所有目标实例。
- 损失同时约束目标隐藏、定位偏移、非目标保持与可打印性。
- 包含白盒、黑盒迁移、参数消融和实体装置实验。
- 与随机/纯色贴片比较，明确展示定向攻击与遮挡镜头的区别。

## 局限

- 威胁模型仍要求攻击者接触相机镜头，且贴片位置与打印误差会影响效果。
- 非目标类别 GT 由 YOLOv5 生成，100% clean AP 不是独立人工标注基准。
- 物理实验是摄像头拍摄屏幕，并非真实道路、距离、天气和振动条件。
- 目标为 stop sign，其他类别、相机 ISP 与现代检测器上的普适性仍需验证。

## 评分

- **创新性：8.5/10**：镜头级、类别定向、通用物理贴片具有鲜明攻击面。
- **实验充分性：8/10**：数字、迁移和物理证据完整，但实景规模有限。
- **YOLO 防御价值：8.5/10**：很适合构建固定传感器扰动 Harness。
- **综合：8.3/10**。
