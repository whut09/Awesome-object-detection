---
title: "Speed Up Object Detection on Gigapixel-Level Images With Patch Arrangement"
description: "解析 PAN 如何用多粒度 patch tree、PFM 与 PPM 把海量局部区域压缩成少量 canvas，降低千兆像素检测的前向次数。"
tags: ["CVPR 2022", "千兆像素检测", "Patch Arrangement", "强化学习", "推理加速"]
---

# Speed Up Object Detection on Gigapixel-Level Images With Patch Arrangement

**官方论文**：[CVF 论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Fan_Speed_Up_Object_Detection_on_Gigapixel-Level_Images_With_Patch_Arrangement_CVPR_2022_paper.html) · [官方 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_Speed_Up_Object_Detection_on_Gigapixel-Level_Images_With_Patch_Arrangement_CVPR_2022_paper.pdf)  
**官方代码**：论文与 CVF 页面未提供作者仓库；PAN 的树编码、PointerNet 解码器和强化学习目标只能依据公开描述复现。

## 一句话总结

PAN 先在多粒度 patch tree 中决定哪些相邻细块可以被一个粗块替代，再把剩余任意位置的 patch 按序装入共享 canvas，让一次 detector forward 覆盖多个高分辨率区域，在 PANDA 上以约 0.702 AP50 把速度从 0.07 提升到 0.37 FPS。

## 研究背景与问题

PANDA 图像约为 `25,000×14,000` 像素。整图缩放会让远处行人失去可辨细节，滑窗或 ClusDet、DMNet 一类候选区域方法虽然避开空背景，却仍要逐 patch 调用检测器；当候选数量达到上万次时，网络本身再快也无法承受。论文抓住了一个调度层面的浪费：局部 patch 的内容需要保持高分辨率，但它们没有必要分别占用一张 detector 输入。因而目标从“找到更少 patch”扩展为“选择合适粒度，并提高每次前向中的有效像素占比”。

## 方法总览

测试时先把原图下采样，由 coarse Faster R-CNN 得到粗框，Mean Shift 按空间位置聚类生成 fine-grained 初始 patch；这些叶节点再次聚类形成 middle-layer coarse patch，整图作为根，构成多粒度树。Tree-LSTM 自底向上聚合父子信息，按先序遍历展开后的隐藏状态再进入 Chain-LSTM，补充同层关系。Patch Filter Module（PFM）用 PointerNet 风格解码器从粗、细节点中互斥选择；Patch Packing Module（PPM）把筛后节点重组为两层树，输出装箱顺序，并用贪心布局把多个 patch 放到容量受限的 canvas。所有 canvas 才送入最终检测器，结果坐标再映射回原图。

## 方法详解

### 多粒度树与 Mixed-LSTM

每个 patch 节点用八维向量描述：中心 `x/y`、宽高、长宽比、面积、块内平均目标面积和目标数量；后两项由粗检测结果估计。全连接层先嵌入节点，Child-Sum Tree-LSTM 把所有子节点隐藏状态求和后传给父节点，显式表达“粗块覆盖哪些细块”；Chain-LSTM 再按树的先序序列编码同层 patch 的相互关系。最终表示同时携带空间包含、目标密度和跨区域上下文，供两个解码器共享。

### PFM：在粗块速度与细块精度之间选择

PFM 将 tree-to-set 写成序列决策，每一步指向一个节点。若选择 middle 节点，其覆盖的 leaf 节点会被 mask；选中任一 leaf，也会屏蔽对应 coarse 父节点，避免同一区域重复检测。过滤奖励为 `1-(|S|-|Vmid|)/(|Vleaf|-|Vmid|)`，鼓励用较少 coarse patch 替代细块，但最终策略还接收后续检测 AP 的回报，因此不会无条件追求最少节点。

### PPM：全局排序并生成 canvas 布局

PPM 不再关心 patch 在原图中是否相邻，而把所有保留区域放在同一个全局集合中。Pointer decoder 给出装箱顺序；贪心布局让每个 patch 优先占据 canvas 底部，当前画布空间不足时再开启新画布。最大容量 `C` 限制一张 canvas 可容纳的 patch 数，也直接控制理论加速比。PFM 的策略回报采用 `λ·Rfilter+Rpack`，PPM 使用 detector AP 作为 `Rpack`，二者通过 REINFORCE、批均值 baseline 与 Monte Carlo batch 联合优化。

## 实验与证据

- PANDA 含 18 个场景，13 个训练、5 个测试，每场景约 30 张图；评估 full-body detection，并分别报告 small `<96×96`、middle `96×96–288×288`、large `>288×288` 的 AP50、检测器调用数 `#Pass` 与 FPS。
- 粗检测器为 ResNet-50 Faster R-CNN：训练 coarse detector 时原图下采样 4 倍并切 `2048×1024` 窗；最终评测下采样 2 倍。FC 为 64 维，Tree/Chain-LSTM 和两个 decoder 的 hidden size 均为 128；Adam 学习率 0.001，Monte Carlo batch 64，单张 GTX 1080Ti。
- DS+SW(FR) 需要 13,620 次 pass，AP50 0.705、FPS 0.07；PAN `C≈4` 为 3,671 pass、0.715 AP50、0.23 FPS；PAN `6×` 为 2,565 pass、0.702 AP50、0.37 FPS，即约 2.7 秒一张，速度约为前者五倍。
- 对比 ClusDet 的 7,871 pass、0.718 AP50、0.12 FPS，ClusDet+PAN 降至 3,683 pass，仍有 0.713 AP50 和 0.23 FPS；说明框架可以接收其他方法生成的初始 patch。
- 过滤/打包消融显示，仅 coarse filter 为 2,838 pass、0.687 AP50；仅 fine filter 为 15,019 pass、0.715；multi-grained filter 后再 packing 为 3,671 pass、0.715，证明跨粒度选择保住精度，装箱负责主要调用压缩。
- 编码器消融中，Tree-LSTM、Chain-LSTM、Mixed-LSTM 分别得到 0.692/0.690/0.715 AP50；`λ=0.05/0.10/0.20` 对应 0.690/0.715/0.672，最佳折中为 0.1。接入 YOLOv3 后 FPS 从 0.518 升至 2.115，SSD 从 0.515 升至 2.088。

## 对 YOLO-Agent 的启发

PAN 可以被抽象成 YOLO 前端的“区域调度 agent”，优化目标必须是原图级端到端吞吐，而不是单个 canvas 的 kernel 时间。**Harness** 需要固定同一 coarse proposal 源与同一 YOLO 权重。**对照组**包含全图缩放、固定滑窗、仅候选裁块、仅多粒度 PFM、随机或面积排序 packing、PFM+学习式 PPM，并扫描 `C=2/4/6/8` 和 `λ=0.05/0.1/0.2`。**指标**：逐张原图统计 mAP50-95、small AP、漏检数、`#Pass`、canvas 像素利用率、patch 缩放倍率、坐标回映错误、跨 canvas NMS 开销、P50/P95 延迟、峰值显存与 coarse-stage 时间。**失败判断**：加速主要来自把小目标缩得更小、small AP 相对滑窗下降超过 2 点、端到端速度不足 2 倍、回映/NMS 产生超过 0.5% 异常框，或随机装箱与学习 PPM 的差距小于 5%，任一发生即拒绝该策略；只有在精度约束内真实减少 detector pass，才允许 agent 采用。

## 优点

- 把 patch 粒度选择与画布利用率统一优化，解决了既有方法只减少候选、不减少前向浪费的问题。
- PFM 处理局部父子替代，PPM 处理全局组合，两个模块的职责和数据流可区分。
- 验证了 Faster R-CNN、YOLOv3、SSD 以及 ClusDet 入口，表明它是检测器无关的输入调度层。

## 局限

- 上游粗检测漏掉的对象不会进入 patch tree，后续安排无法补救，论文没有单独量化这一召回上限。
- 贪心 canvas 布局与 resize/padding 可能改变目标尺度分布；对跨 patch 大目标和非人类类别的证据有限。
- 强化学习回报依赖完整 detector AP，训练成本与稳定性未详细报告，也缺少作者代码供核验。
- `#Pass` 是重要但不充分的代理量，真实系统还受裁剪、拷贝、拼接、坐标恢复和 NMS 限制。

## 评分

- **创新性：8.5/10**——将候选区域问题推进为可学习的多粒度选择与二维装箱问题。
- **工程证据：8/10**——给出 pass、FPS、尺度 AP、容量和多检测器扩展，但端到端耗时拆分不足。
- **YOLO-Agent 适配度：9/10**——天然适合作为超高分辨率场景的前置调度策略。
- **综合：8.5/10**——在千兆像素人群检测上非常有价值，复现时必须严防“通过缩小目标换速度”的伪加速。
