---
title: "Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset"
description: "解析 MvHeat-DET 的多专家热传导算子、IoU 查询选择，以及 EvDET200K 事件检测基准。"
tags: ["CVPR 2025", "目标检测", "事件相机", "MvHeat-DET", "EvDET200K"]
---

# Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset

**会议**：CVPR 2025  
**论文**：[arXiv](https://arxiv.org/abs/2412.06647)  
**代码**：[Event-AHU/OpenEvDET](https://github.com/Event-AHU/OpenEvDET)

## 一句话总结

MvHeat-DET 把事件帧特征视为二维热量分布，用 DCT、DFT、Haar 三种频域“传热专家”动态扩散信息，再以 IoU-based Query Selection 提升 DETR 查询质量，并配套发布 EvDET200K。

## 研究背景与问题

现有事件检测基准常分辨率低、类别少或只覆盖单一光照，难以评估密集、小目标和复杂天气。Transformer 自注意力在高分辨率事件帧上代价高，单一 DCT 热传导算子又隐含固定边界与频率偏好，未必同时适配低运动稀疏区域和高运动跨 patch 交互。论文同时解决模型与数据两端：构建更丰富的基准，并让骨干按输入内容选择频域扩散机制。

## 方法总览

事件流先堆叠成 event frames，经 stem 得到 embedding，随后进入四阶段 MvHeat Encoder。每阶段包含若干 MoE Heat Conduction Operation（MHCO）层，空间分辨率逐级减半；MHCO 用 policy network 和 Gumbel-Softmax 在 DCT-IDCT、DFT-IDFT、HT-IHT 三个专家中选择路径，Frequency Embeddings（FEs）预测热扩散率 `k`，频域特征乘 `exp(-k(vx²+vy²)t)` 后逆变换。编码结果进入 IoU-based Query Selection（IQS），再由 DETR 风格 decoder 输出框与类别。

## 方法详解

三类专家对应不同传播边界。DCT 和 Haar 在 patch 内满足近似 Neumann 边界，适合低运动、大片空白事件区域；DFT 允许全局频率交互，更适合高速目标或密集场景。policy network 为当前特征产生三路分数，Gumbel-Softmax 保证离散选择可反向传播。FEs 与频域张量同形，经线性层预测内容相关的 `k`，使“扩散多远”不再是固定超参数。

IQS 针对 DETR 仅按分类分数选 top-K query 的定位偏差：训练时把 IoU 融入分类目标，使高分类分数同时代表更高定位质量，再沿用 top-K 选择。完整流向是“事件帧—stem—四阶段 MHCO—IoU 对齐查询—decoder”。EvDET200K 由 Prophesee EVK4-HD 以 1280×720 拍摄，含 10,054 段 2–5 秒视频、202,260 个框、10 类，训练/验证/测试为 6031/1002/3021，51% 为小目标。

## 实验与证据

EvDET200K 上，MvHeat-DET 达到 52.9 mAP、80.4 mAP50、55.9 mAP75，参数 47.5M、56.4G FLOPs、58 FPS；优于 vHeat encoder 基线的 50.3/72.2/54.2，也高于 Faster R-CNN 46.0、Swin-T 49.0 和 YOLOv10-B 44.1 mAP。N-Caltech101 上达到 55.7 mAP，超过 EAS-SNN 53.8。

组件递增为 DETR baseline 40.9，加入 IQS 41.6，替换 vHeat Encoder 后 50.3，再加入 MoE 达 52.9。专家由 DCT 单路 50.3 提升到 DCT+DFT 的 52.7，再到三专家 52.9；固定 `k`、直接学习 `k`、由 FEs 预测分别为 48.2、49.1、49.7。输入从 256² 增至 640²，mAP 40.0 升至 52.9，但 FLOPs 从 11.9G 增至 73.4G；作者最终选 `(2,2,12,2)`，在 52.9 mAP 与 56.4G 间折中。

## 对 YOLO-Agent 的启发

MHCO 可作为 backbone block，IQS 则用于 query-based 分支。**Harness** 设置原骨干、单 DCT-HCO、固定三专家平均、Gumbel 动态三专家；对 DETR 另开关 IQS。观测 mAP、mAPs、专家使用熵、天气路由比例、FLOPs、FPS 与显存。动态 MoE 相对单 DCT 至少提升 2.0 mAP，且 512² 保持 ≥50 mAP、速度下降不超过 25% 才通过；若路由塌缩到一路、低分辨率下降超过 10 mAP，或 IQS 只提高分类而 mAP75 不升，则失败。

## 优点

- 模型创新与新基准同步给出，可检验高分辨率事件检测。
- 专家数、扩散率、分辨率、深度和通道均有消融。
- EvDET200K 覆盖多光照、多运动、动态背景及密集小目标。

## 局限

- 频域专家与 DETR decoder 使 640² 配置成本明显。
- 标注由每段视频转换五帧完成，未充分利用连续高频标注。
- 缺少逐样本专家路由可视化与负载均衡分析。

## 评分

- **创新性**：★★★★☆
- **实验充分度**：★★★★★
- **数据集价值**：★★★★★
- **YOLO-Agent 参考价值**：4.1/5
