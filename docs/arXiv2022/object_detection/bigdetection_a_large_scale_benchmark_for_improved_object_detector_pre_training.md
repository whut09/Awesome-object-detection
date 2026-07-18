---
title: "[论文解读] BigDetection: A Large-scale Benchmark for Improved Object Detector Pre-training"
description: "BigDetection 统一 LVIS、OpenImages 与 Objects365 的类别和标注，构建 600 类检测预训练集。"
tags: ["arXiv 2022", "目标检测", "BigDetection", "预训练", "数据集"]
---

# BigDetection: A Large-scale Benchmark for Improved Object Detector Pre-training

**论文**: [arXiv](https://arxiv.org/abs/2203.13249)  
**代码**: [amazon-science/bigdetection](https://github.com/amazon-science/bigdetection)  
**任务**: 大规模目标检测预训练

## 一句话总结

BigDetection 将 LVIS、OpenImages 和 Objects365 的异构类别空间映射为 600 类统一 taxonomy，整理出 340 万训练图像与 3600 万框，用更大且更多样的检测监督替代单纯 ImageNet 分类初始化。

## 研究背景与问题

检测预训练长期依赖 ImageNet，分类标签却不提供实例位置；直接混合多个检测集又会遇到同义类、父子类和缺失标注冲突。例如一个数据集标“vehicle”，另一个细分为 car/bus，未标类别还可能被错误当背景。论文的核心工作不是简单拼接，而是定义可审计的类别合并和标注清洗原则。

## 方法总览

数据管线先规范 LVIS、OpenImages、Objects365 的类别名称与 WordNet/语义关系，再合并同义类、处理层级冲突并过滤质量不足的框。统一后的 BigDetection 保留 600 类长尾分布。检测器先在该集合训练，再将分类头替换为下游类别并在 COCO、LVIS 等数据上微调；论文同时把它作为独立 benchmark 比较 Faster R-CNN、RetinaNet、FCOS 等结构。

## 方法详解

类别统一需要处理“完全同义”“包含关系”和“仅局部重叠”三种情况。作者避免把粗粒度父类与细粒度子类盲目压成一类，并对不同来源的 missing annotation 设置忽略逻辑，降低假负样本。预训练输出的是完整 detector 权重，包括 backbone、FPN、RPN/检测头，而不是只迁移分类 backbone。

## 实验与证据

数据规模达到 3.4M images、36M boxes、600 categories，明显超过三个单独来源。论文在 COCO 和 LVIS 下游比较 ImageNet 初始化、Objects365 预训练及 BigDetection 预训练；统一大集在多种 detector/backbone 上稳定提高 AP，长尾和大模型受益更明显。消融显示，仅把数据机械合并不如经过 taxonomy reconciliation 与标注清理的版本，说明收益不只来自样本数量。

## 对 YOLO-Agent 的启发

Harness 应设置 ImageNet、Objects365、naive-union、BigDetection 四个初始化，并固定下游 epoch、增强与输入尺度。记录 COCO AP、LVIS APr/APc/APf、收敛速度和预训练存储成本；另统计每类假负样本率。若统一 taxonomy 后常见类提升但 rare 类下降，或相同训练步数下增益仅由更多预训练迭代造成，就不能接受该数据配方。YOLO 接入时还应检查检测头类别映射是否被正确重置。

## 优点

- 把多个高价值检测集转成可复用的统一预训练资源。
- 预训练完整检测器，位置知识可以直接迁移。
- 数据、模型和代码公开，便于建立强初始化基线。

## 局限

- 类别合并仍包含人工语义判断。
- 长尾、来源域差异与漏标问题没有完全消失。
- 预训练 340 万图像需要较大算力和存储。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **工程价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★★
