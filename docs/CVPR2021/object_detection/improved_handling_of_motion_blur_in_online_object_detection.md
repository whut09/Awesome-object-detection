---
title: "Improved Handling of Motion Blur in Online Object Detection"
description: "围绕 egomotion 模糊的核居中、框扩张与按曝光专门化检测器的系统研究。"
tags: ["CVPR 2021", "目标检测", "运动模糊", "鲁棒性", "在线视觉"]
---

# Improved Handling of Motion Blur in Online Object Detection

**论文页面**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Sayed_Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR_2021_paper.html)  
**官方 PDF**：[CVPR 2021 Paper PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Sayed_Improved_Handling_of_Motion_Blur_in_Online_Object_Detection_CVPR_2021_paper.pdf)  
**官方代码**：[mohammed-amr/detectInBlur](https://github.com/mohammed-amr/detectInBlur)

## 一句话总结

这篇论文把相机自运动模糊拆成 3 类轨迹 P1–P3 与 5 档曝光 E1–E5，发现最有效的不是先去模糊，而是先把 blur kernel 重心移到滤波器中心、按核的最大位移扩张 GT 框，再由 ResNet-18 blur estimator 选择按长曝光类型专门训练的 Faster R-CNN。

## 研究背景与问题

COCO-C 等鲁棒性基准通常把直线运动模糊视为普通图像腐蚀，却继续沿用锐利图像的框。对检测任务，这会制造空间标签错误：非居中的曝光轨迹把目标整体拖向一侧；即使核已居中，目标像素仍扩散到原框之外。论文因此不把问题简化成“纹理变差”，而是分别检查去模糊、各向异性尺度、分布外适应、标签生成和模糊类型条件化五个假设。

完整数据流为：**COCO 图像 → Boracchi–Foi 2D camera trajectory → P1/P2/P3 与 E1–E5 blur kernel → kernel barycenter centering → GPU 稀疏卷积 → 按核支撑范围扩张 bounding box → blur-augmented Faster R-CNN-R50-FPN**。在专门化版本中，测试图像还进入 **ResNet-18 blur estimator → 低曝光通用网络或三个长曝光 P 专家 → 检测框**；这条路与 Deblur、Squint、AugMix、minibatch-statistics 四类对照逐一比较。

## 方法总览

作者为每个 `(P,E)` 组合预生成 12,000 个 128×128 核，共 **180,000** 个训练核；轨迹长度为 96。P1 表示最抖的“nervous camera”，P2 偏往复，P3 多为近直线；E1–E5 通过提前截断轨迹控制曝光。图像不先统一缩放，直接以 reflection padding 做卷积，避免把焦距变化与核缩放混在一起。

标签修正包含两步。第一步按非零核权重计算 barycenter，并把它平移到滤波器中心，使模糊目标的平均位置与原框对齐。第二步取核在 x/y 方向的非零极值 `x−,x+,y−,y+`，把框改为 `bx-|x−|, by-|y−|, bw+|x−|+|x+|, bh+|y−|+|y+|`，覆盖曝光期间目标可能出现的位置并用于相应测试评估。

## 方法详解

### 五类 remedy 的真实接口

**Deblur** 用 Nah 等人的 GoPro 去模糊网络预处理测试图，再送原始或模糊增强检测器，代价约为检测流程的 12 倍。**Squint** 假设 oracle 知道核方向，沿核主成分对输入下采样，经过 backbone 后对每层激活做倒数尺度恢复，再送 detection head。**OOD** 包含三种 AugMix（非空间、空间但不改框、空间且扩框）以及测试时把 batch-size-1 统计与训练统计按 `n=1,N=16` 混合的 BatchNorm 适应。

**Blur specialization** 有两套模型袋：按 P1–P3 各训一个专家并配一个全模糊通用网络；或让三个专家只负责各自 P 的长曝光，另一个 Low-Exposure Net 负责锐利与短曝光。对应的 16 类或 4 类 ResNet-18 模糊估计器负责路由。论文最终偏好第二套 Spec by Exposure，因为它既保住锐利/低曝光准确率，又在重模糊下利用更强的形状偏置。

### 复现边界

“Expanded Labels”不能脱离 kernel centering 单独照搬。若核重心仍偏在滤波器一侧，扩框只会把错误中心周围的标签放大，无法消除输入与标注的系统平移。训练核来自预生成库，但评估核按固定随机种子在线生成，目的是避免测试重复训练轨迹；卷积前也不能把所有图像压到固定分辨率，否则同一个 128×128 核对应的视场比例被改写。作者用 10% 锐利、90% 模糊图训练 Standard Augmented，专门化模型则按 P/E 子空间取样。真实模糊表中的“Sharp/Blurry”共享 DetectoRS 伪框来源，因此应比较同一数据集内模型差值，而不宜把这些绝对 AP 与人工标注 COCO AP 直接横比。

## 实验与证据

基础检测器是 torchvision 的 Faster R-CNN-R50-FPN，COCO 上为 **58.5 mAP@0.5、37.0 mAP@0.5:0.95**。COCO minival 的曲线显示，核不居中在最强模糊下可损失约 **8–10 mAP@50**；采用 expanded labels 的模型整体优于对应标准框版本，而 blur augmentation 后再叠加 minibatch statistics 或 deblurring 没有收益。论文也明确指出 Deblur 既慢又会改变训练分布。

跨真实/准真实数据使用 DetectoRS 在锐利帧上的预测作 pseudo-ground-truth，以 mAP@0.5 评估。`Spec by Exposure + NonSpatial AugMix` 在 GOPRO blurry 为 **28.15**，原始模型为 19.64；RealBlur blurry 为 **36.37**，原始模型 29.00；REDS blurry 为 **32.55**，原始模型 24.38。去模糊后再用原模型在 GOPRO blurry 仅 2.24，在 RealBlur 为 28.76，证明视觉上更清晰并不等于检测更可靠。这组消融也说明任务指标必须优先于重建观感。

锐利图像精度同样要纳入判断：Spec by Exposure 在 GOPRO sharp 为 34.80，带 NonSpatial AugMix 后为 34.89，接近原始模型 35.85；Standard Augmented 只有 32.42。按曝光路由的意义正是把 sharp/low-exposure 交给专门网络，减少统一模糊增强造成的清晰域遗忘。

组件对照同样支持专门化：RealBlur blurry 上 Spec by Exposure 为 35.26，Spec by Type 为 36.11，Standard Augmented 为 35.63；加入 NonSpatial AugMix 后 Spec by Exposure 达 36.37。GOPRO Expanded blurry 上 Low-Exposure Net 为 31.60、Spec by Exposure 为 31.14，而带 NonSpatial AugMix 的版本约 30.9，说明 AugMix 的主要价值是跨数据集泛化，并非所有模糊域都提升。

## 对 YOLO-Agent 的启发

对 YOLO-Agent，最重要的迁移点是“标签生成器必须理解传感器退化”。可在增强模块输出 blur kernel 的同时输出 centered/expanded boxes，并让路由器按估计的 P/E 选择 LoRA、BN 参数或轻量 expert head；不应先默认接一个 deblurrer。若追求单模型，可把论文建议的多个专家改成共享 backbone、按模糊类别门控的 head。

**专属 Harness**：A 为锐利训练 YOLO，B 为随机模糊但原框，C 为 centered kernel+原框，D 为 centered+expanded boxes，E 为 D 加统一模糊模型，F 为 D 加按曝光专家与 blur estimator；另设 Deblur+A、Squint-oracle、AugMix、test-BN 对照。观测每个 P1–P3/E1–E5 的 mAP@50、锐利集 AP、框中心误差、GT 覆盖率、路由准确率、端到端延迟和模型内存。通过标准：C 明显优于 B，D 在高曝光继续优于 C，F 在不显著损伤锐利 AP 的前提下提高 E4/E5，并在 GOPRO/RealBlur/REDS 至少两域复现方向；失败判断：扩框只放宽评测而不改善训练、oracle Squint 才有效、路由错误抵消专家收益，或 Deblur 延迟使系统失去在线性。

## 优点

- 把运动模糊的空间标签歧义与纹理退化明确分开。
- 同时报告有效方案和去模糊、OOD 适应等负结果。
- 合成 COCO、GOPRO、REDS、RealBlur 上均有验证。
- 核参数、曝光分档和模型路由接口清晰，可做工程复现。

## 局限

- 真实数据没有人工检测框，跨域评估依赖 DetectoRS 伪标签。
- 多专家模型增加内存，blur estimator 错误会直接选错检测器。
- 研究聚焦相机 egomotion；动态目标的深度相关、空间变化模糊未解决。
- 锐利图像基础检测准确率本身低于 60 mAP@0.5，仍限制最终上限。

## 评分

- **创新性：8.5/10**：核居中与扩框直击检测标签歧义。
- **实验充分性：8.5/10**：五类假设、组合与多模糊数据集齐全。
- **YOLO 可迁移性：8/10**：标签增强容易，专家路由有额外成本。
- **综合：8.3/10**。
