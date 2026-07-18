---
title: "[论文解读] Interactive Multi-Class Tiny-Object Detection"
description: "C3Det 将用户点击与全图特征做局部 late fusion 和全局 feature correlation，交互补全多类微小目标。"
tags: ["CVPR 2022", "微小目标", "C3Det", "交互标注", "点提示"]
---

# Interactive Multi-Class Tiny-Object Detection

**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Interactive_Multi-Class_Tiny-Object_Detection_CVPR_2022_paper.html)  
**代码**: [Interactive-Multi-Class-Tiny-Object-Detection](https://github.com/ChungYi347/Interactive-Multi-Class-Tiny-Object-Detection)  
**任务**: 多类别微小目标交互式标注

## 一句话总结

C3Det 接收少量带类别的点击：局部路径把 click heatmap 与检测特征 late-fuse，全局路径用点击位置特征与整图做 feature correlation，从而一次点击召回同类多个微小实例，并允许下一轮补点纠错。

## 研究背景与问题

遥感和细胞图像一张图常有数十到数百个小实例，逐框标注极慢。交互分割多针对单对象，传统点监督 detector 训练后不能在标注过程中根据点击即时改变类别与实例集合。多类场景还需区分相似微小纹理。

## 方法总览

图像先经过 one-stage 或 two-stage detector backbone。用户点击编码为类别相关点图；local late-fusion 在点击邻域强化对应类别，global feature-correlation 抽取点击处 embedding，与全图特征计算相似响应，发现远处同类实例。检测头输出候选，用户只对漏检或错类再点击，模型迭代更新结果。

## 方法详解

局部分支保证被点击对象本身稳定命中，全局相关分支承担“一点带多例”。多类别点击使用独立类别编码，避免不同类别原型混合。训练时模拟不同点击轮次和正负点击，让模型在少点击与后续纠错两种状态都可工作。

## 实验与证据

Tiny-DOTA 与 LCell 上，C3Det 在相同点击数下比已有 interactive annotation 方法取得更高 mAP，并可接入两阶段和一阶段 detector。真实用户研究显示相对手工框标注快 2.85×，NASA-TLX 任务负荷仅 0.36×。消融确认 late fusion 稳定局部框，feature correlation 决定跨图同类召回。

## 对 YOLO-Agent 的启发

Harness 应画 mAP-vs-click curve，而不是只报最终 AP；对照仅点击热图、仅相关图、C3Det 双分支，并记录每类平均点击、误扩散率和交互延迟。若一个点击召回大量相似背景，或多类点击后特征原型互相污染，即判失败。用户研究需固定图像和停止条件，防止因标注协议不同虚增 2.85×。

## 优点

- 一次点击可传播到多个同类实例。
- 同时支持多类与多轮纠错。
- 有真实标注员效率和负荷证据。

## 局限

- 类内外观差异大时相关传播有限。
- 密集粘连细胞仍难得到精确框。
- 需要交互式推理界面和低延迟后端。

## 评分

- **创新性**: ★★★★★
- **实验充分度**: ★★★★★
- **工程价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★☆
