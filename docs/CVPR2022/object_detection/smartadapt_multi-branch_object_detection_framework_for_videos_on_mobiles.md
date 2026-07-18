---
title: "[论文解读] SmartAdapt: Multi-Branch Object Detection Framework for Videos on Mobiles"
description: "SmartAdapt 在 MBODF 的五维执行分支中，由内容感知 scheduler 为每个 GoF 选择检测/跟踪配置。"
tags: ["CVPR 2022", "视频目标检测", "SmartAdapt", "MBODF", "移动 GPU"]
---

# SmartAdapt: Multi-Branch Object Detection Framework for Videos on Mobiles

**会议**: CVPR 2022  
**论文**: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SmartAdapt_Multi-Branch_Object_Detection_Framework_for_Videos_on_Mobiles_CVPR_2022_paper.html)  
**代码**: 论文未提供官方仓库  
**任务**: 移动视频自适应检测

## 一句话总结

SmartAdapt 把连续视频切成 Group of Frames，在 Multi-branch Object Detection Framework 中联合配置 detector interval、detector resolution、proposal 数、tracker resolution 和 tracking confidence，再由 content-aware scheduler 选择满足延迟预算的分支。

## 研究背景与问题

固定分辨率和固定检测间隔无法同时处理静态街景与快速运动；ApproxDet/FastAdapt 的粗粒度策略也难覆盖检测器与 tracker 的组合空间。移动 GPU 还要求每次切换都守住如 33ms 的 deadline，而非平均够快。

## 方法总览

每个 GoF 首帧运行 Faster R-CNN、EfficientDet 或 YOLO，后续帧由 MedianFlow 等 tracker 更新。MBODF 将五个 knob 的一个取值组合视为 execution branch，并离线剔除被支配配置。轻量 scheduler 从当前内容和历史质量特征预测各分支准确率/延迟，在线选择预算内预期最准者。

## 方法详解

MBODF 的一个 branch 是五元组 $(d_i,r_d,n_{prop},r_t,c_t)$：检测间隔决定每个 GoF 何时重检，detector/tracker resolution 控制两条计算链，proposal 上限和 tracking confidence 控制候选规模。离线 profiling 先删除 accuracy-latency 被支配分支，scheduler 只在 Pareto 候选中预测内容相关收益，避免在线遍历组合。


## 实验与证据

论文在移动 GPU 和多个视频集上与固定 detector、tracking-by-detection、ApproxDet/FastAdapt 比较，重点报告 deadline 下 mAP。消融拆分分支空间大小、scheduler 特征和切换频率；结果显示细粒度五维配置比只调分辨率或检测间隔更能适应内容变化。

## 对 YOLO-Agent 的启发

Harness 应回放同一视频并记录每个 GoF 的分支、deadline miss rate、切换开销和跟踪漂移；oracle branch 作为上界，固定最快/最准与单 knob 策略作基线。若 scheduler 选择准确率远低于 oracle，或 p95 超 33ms 即使平均 mAP 更高也应失败；频繁切换造成缓存抖动时需加入 hysteresis。

## 优点

- 把检测与跟踪的多维旋钮统一成可搜索分支。
- 内容感知而非固定策略。
- 直接围绕移动 deadline 优化。

## 局限

- 分支 profiling 与设备绑定。
- tracker 漂移会污染后续 GoF。
- scheduler 本身需要训练数据和在线开销。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★☆
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★☆
