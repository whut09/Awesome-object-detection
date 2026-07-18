---
title: "[论文解读] Dual Domain Control via Active Learning for Remote Sensing Domain Incremental Object Detection"
description: "原创中文解读 Active-DDC：用 ALER 控制数据分布漂移，用 ADSC 对齐 DETR 查询特征，缓解遥感域增量遗忘。"
tags: ["ICCV 2025", "遥感检测", "域增量学习", "主动学习", "知识蒸馏"]
---

# Dual Domain Control via Active Learning for Remote Sensing Domain Incremental Object Detection

**论文**：[CVF 官方论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Dual_Domain_Control_via_Active_Learning_for_Remote_Sensing_Domain_ICCV_2025_paper.html)  
**代码**：作者在论文正文和 CVF 条目中均未列出项目仓库，本文不附推测性实现链接  
**发表**：ICCV 2025  
**类别**：Domain-Incremental Detection, Remote Sensing

## 一句话总结

Active-DDC 把遥感域增量遗忘拆成数据分布漂移与 DETR 查询特征漂移：ALER 主动挑选高信息、高风险旧样本回放，ADSC 先筛选语义相关查询，再用最优传输完成新旧模型的多对多对齐。

## 研究背景与问题

遥感域会随传感器、飞行高度、地理区域和分辨率变化，同一类别在 HRRSD、LEVIR、DIOR、DOTA 中呈现不同尺度与布局。连续微调新域会遗忘旧域；随机回放无法在有限内存中覆盖代表性样本；传统 logit 或伪标签蒸馏又忽视 DETR 查询无序、动态、跨任务语义位置会变化的特性。论文因此要求同时控制输入样本分布和模型内部查询分布，而不是只在一个层面追加正则。

## 方法总览

Active-DDC 以 Deformable DETR 为主干。任务 t 到来前，Data-based Active Learning Example Replay（ALER）根据上一模型的假阳性率、假阴性率预筛困难样本，计算图像信息熵 H(x) 与遥感前景/背景面积比对应的风险 R(x)，以 `S(x)=αH(x)+βR(x)` 排序写入有限 memory bank；新旧任务难度比决定新样本占比，低分旧样本被淘汰。联合回放后，Query-based Active Domain Shift Control（ADSC）用教师与学生预测框 IoU 预筛查询，再以框 IoU 和查询余弦距离构建代价矩阵，经带熵正则的 Sinkhorn 最优传输得到多对多匹配，最后对匹配查询施加蒸馏损失。

## 方法详解

ALER 的风险设计针对遥感极端前景/背景不平衡：前景面积比越低，样本越容易因小目标和大背景而被忽视，映射函数会给出更高风险。它不是固定保存“最难”样本，而是在每个任务检查 memory bank 旧样本难度 `p_{t-1}` 与新域难度 `p_t`，动态确定载入数量，再综合信息量和风险更新存储，使联合训练数据靠近多个域的有效覆盖。

ADSC 先消除无效查询。只有学生框与教师框 IoU 超过阈值的查询进入蒸馏；随后代价由 IoU 项和 `1-cosine` 项加权，Sinkhorn 求解平滑传输计划，允许多个新查询承接多个旧查询知识，避免按索引一一对应。总损失是 Deformable DETR 的分类/框回归损失与匹配查询蒸馏损失之和，因此推理结构不增加模块，主要成本发生在增量训练阶段。

## 实验与证据

实验将 HRRSD、LEVIR、DIOR、DOTA 裁剪到共同的 ship、airplane、oil tank 三类，DOTA 切为 1024×1024，评测 mAP、AP50、AP75 与联合测试。四域顺序 I 中，Fine-tuning 联合 AP50 为 60.3，Active-DDC 达 67.8；联合 mAP 从 31.3 提到 36.2，优于 CL-DETR、Incre-DETR、DyQ-DETR 和 LDB。不同顺序仍提升：顺序 II 联合 mAP 24.5→32.7，顺序 III 36.6→42.5，但结果也显示顺序敏感。模块消融中，顺序 I 基线 31.3 mAP；仅 ALER 为 34.7，仅 ADSC 为 33.9，二者合用 36.2。ALER 联合 AP50 65.1，高于 ER 61.1、RPC 64.7；ADSC 为 64.0，高于 DETRDistill 61.0 和 KD-DETR 63.4。

## 对 YOLO-Agent 的启发

若 Agent 使用 YOLO，ALER 可直接复用，但 ADSC 依赖 DETR object query，不能无损移植到密集检测头；应把它列为“架构限定组件”，或改用实例原型/候选集合对齐并重新验证。

**机制特定 Harness**：**对照组**在固定域序列及其逆序上运行 Fine-tune、等容量随机回放、仅 ALER、常规 logit KD、仅 ADSC 与完整 Active-DDC，所有方法共享 memory bank 容量和逐域训练预算。**指标**同时报告当前域 mAP、旧域平均 mAP、平均遗忘量与 forward transfer，并按尺度、旋转角、传感器分辨率和拥挤度切片；另追踪 ALER 样本熵/前景比/梯度贡献，以及 ADSC 的查询预筛率、OT 代价和跨轮匹配稳定性。**失败判断**要求完整方法相对随机回放至少减少 1.5 mAP 的平均遗忘；若等量随机样本可追回其中 80% 以上收益，或打乱 object query 后旧域 mAP 变化小于 0.2，或逆序实验有旧域低于随机回放 1.0 mAP 以上，则认定双域控制无效。

## 优点

- 明确区分数据漂移与查询特征漂移，模块职责清晰。
- ALER 针对遥感前景稀疏特点，而非直接套用自然图像回放策略。
- ADSC 适配 DETR 查询无序性，训练后不增加推理参数。

## 局限

- 类别被裁成三类，尚不能说明细粒度类别增量与域增量同时发生时的表现。
- 检测结果对域顺序仍敏感，联合训练上界差距明显。
- ADSC 与 DETR 查询强绑定，迁移到 YOLO 等密集头需要重新定义对齐对象。

## 评分

**8.4/10**。问题拆解合理，四数据集和多顺序实验有说服力；但类别简化、顺序敏感和架构依赖限制了直接部署范围。
