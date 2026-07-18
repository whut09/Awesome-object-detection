---
title: "Crowd-SAM:SAM as a smart annotator for object detection in crowded scenes"
description: "详解 Crowd-SAM 如何用 DINOv2、EPS 与 PWD-Net 完成拥挤场景少样本检测。"
tags: ["ECCV 2024", "目标检测", "SAM", "拥挤场景"]
---

# Crowd-SAM:SAM as a smart annotator for object detection in crowded scenes

**论文**：[官方论文页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8766_ECCV_2024_paper.php)  
**代码**：[官方代码](https://github.com/FelixCaae/CrowdSAM)

## 一句话总结

Crowd-SAM 冻结 SAM ViT-L 与 DINOv2 ViT-L，只用少量框标注训练前景定位头和 Part-Whole Discrimination Network（PWD-Net），再由 Efficient Prompt Sampler（EPS）删除已被高置信掩码覆盖的密集点提示，使 SAM 能在遮挡、密集行人场景中充当检测器与智能标注器。

## 研究背景与问题

CrowdHuman 平均每图约 22 个目标，逐框标注昂贵。直接用 SAM 自动掩码生成器时，稀疏网格召回不足，类别无关提示产生背景假阳性，密集网格又使解码成本暴涨。论文测得 16×16 网格召回 33.6，128×128 达 76.0，但平均假阳性由 51 增至 485，单图解码由 0.059 秒升至 3.2 秒。

## 方法总览

图像分别进入冻结的 DINOv2 与 SAM encoder。DINOv2 特征经 MLP 和像素二分类头产生类别热图，阈值化后得到候选点 `PG`；EPS 分批取点送入 SAM decoder；PWD-Net 联合 SAM 的 Mask Token、IoU Token 与 DINOv2 Semantic Token 重估掩码；有效掩码覆盖的剩余点被删除，循环至候选耗尽或采样达到 `K=500`。多裁剪结果经 NMS 合并并转为框。

## 方法详解

Class-specific Prompt Generation 以框中心提示 SAM 生成伪掩码，再合并为二值前景监督，用 Dice loss 训练 DINOv2 适配器和分类头；推理阈值为 0.5。EPS 每轮均匀抽取 `PB`，SAM 生成 `M`，PWD-Net 筛出分数超过 `T` 的 `M'`，凡落入有效掩码的未处理点都被剪除。

PWD-Net 的精细 IoU 分数由冻结的 SAM 原 IoU Head 与并行 MLP 相加；并行头接收 IoU Token `U` 和四个 Mask Token `M`。语义分数用 SAM 掩码加权 DINOv2 特征，池化成 Semantic Token `O`，再复用前景分类头。最终 `S=Siou·Scls`，以前景真实 IoU 或背景零分为 MSE 目标。

## 实验与证据

主实验覆盖 CrowdHuman、CityPersons、OCC-Human，并用 COCO 0.1% train-val 和 COCO-OCC 检验多类扩展。两大 ViT-L 均冻结，只训练 2,000 次迭代。CrowdHuman 10-shot 下，单裁剪为 71.4 AP、83.9 Recall、1.7 秒/图；多裁剪为 78.4 AP、85.6 Recall、8.1 秒/图，超过全监督 FCOS 的 76.3 AP，但低于 DINO 的 86.7 AP。

OCC-Human 上达到 31.4 AP，高于 Pose2Seg 的 22.2；CityPersons 50-shot 为 33.3 AP。多类版本在 COCO/COCO-OCC 为 22.0/20.6 AP。移除前景定位、EPS、PWD-Net、多裁剪后 AP 分别为 71.0、77.8、17.0、71.4。去掉 Mask Token 后 AP 暴跌至 38.4；直接微调原 IoU Head 为 75.6。EPS 在 192×192 网格达 73.2 AP，而全量采样在 128×128 已 OOM。

训练阶段并不需要真实实例掩码。作者用框内提示让 SAM 生成伪掩码，随机从伪前景取正点、从背景取负点来训练 PWD-Net；优化器为 Adam，学习率 `1e-5`、权重衰减 `1e-4`、`β1=0.9`、`β2=0.99`，单张 3090 Ti 数分钟即可完成。冻结 SAM 与 DINOv2 既减少可学习参数，也避免 10-shot 条件下基础模型被过拟合破坏。

指标必须结合阅读。CrowdHuman 采用 IoU 0.5 的 AP、MR−2 与 Recall，最佳 Crowd-SAM 的 78.4 AP 并不等价于 COCO 风格 AP；其 MR−2 为 74.8，仍显著落后全监督 ATSS 的 59.7 和 DINO 的 57.6。论文声称“可比全监督”主要指 AP 和少样本效率，而不是所有拥挤行人指标全面领先。SAM baseline 没有可靠分类分数，所以表中只报告 Recall，这也是 PWD-Net 不可缺少的原因。

EPS 的结果说明更密网格并非单调更好。128×128、192×192、256×256 时 EPS 的 AP 为 72.4、73.2、72.3，提示过密点会重新引入边界或背景歧义。随机采样受 `K=500` 限制，在 64 以上几乎停滞；EPS 通过高置信掩码覆盖删除，把预算集中到尚未解释的区域。实际部署还应记录每轮剩余 `PG` 曲线，检查算法是否被少量错误大掩码过早清空。

多裁剪带来 7.0 AP，却把时间从 1.7 秒推到 8.1 秒。它适合离线标注，但如果用于在线伪标签生成，切片重叠率、NMS 阈值和小目标收益必须共同调节。多类 COCO 仅使用 0.1% 训练数据，结果低于 Faster R-CNN 和 BCNet，但 COCO 到 COCO-OCC 只降 1.4 AP，较好支持其遮挡鲁棒性而非通用多类 SOTA 的定位。

PWD-Net 的乘法融合也有明确含义：`Siou` 评估形状与边界质量，`Scls` 判断掩码是否属于前景，任一项低都会压低总分。只靠 IoU Token 容易让背景区域中的完整物体样式掩码获得高分，只靠 DINOv2 语义又无法区分一个人的局部与整体。Mask Token 消融下降 40 AP，Semantic Token 或 IoU Token 消融分别下降 2.1、1.1 AP，证明三种信息的重要性不对称但互补。

OCC-Human 的 Moderate/Hard AP 为 26.5/17.7，均高于 Pose2Seg 的 26.1/15.0，说明优势在严重遮挡下仍存在；CityPersons 的 MR−2 为 31.7，虽然 AP 高于几种 few-shot detector，但与全监督 ATSS 的 27.8 尚有差距。将 Crowd-SAM 作为“替代全部人工标注”的结论过强，更合理的用途是快速生成可复核初始框，再由人工修正极端遮挡样本。

## 对 YOLO-Agent 的启发

可迁移的是“提示生成—昂贵解码—质量复核”的预算流水线。YOLO-Agent 可用 EPS 的覆盖后删除策略分配伪标签或困难区域再检测预算；PWD-Net 则提示质量头应同时读取几何 token 与类别语义，不能只相信原生 objectness。

### 论文专属 Harness

- **对照组**：固定 10 张 CrowdHuman 标注，比较标准 SAM、前景热图+随机上限 500、前景热图+EPS、完整 EPS+PWD-Net；下游训练同一 YOLO。
- **观测指标**：AP、Recall、MR−2、提示数、解码次数、秒/图、显存峰值及下游 YOLO 遮挡分档召回。
- **通过阈值**：EPS 提升至少 2 AP 或 3 Recall，解码次数减少 30%；PWD-Net 加入后 AP 至少提升 10 点。
- **失败判断**：单图超过 2 秒在线预算，或伪标签 YOLO 比人工框基线低 3 AP 以上，则仅作离线标注。

从标注工作流看，掩码转框还会放大边界误差：细长遮挡区域的可见掩码框与 full-body 框定义不同。论文在人群数据上统一采用 visible annotation 训练和评估，因此迁移到使用 full box 的行人检测器时，必须重新生成监督或做框扩张；否则 AP 下降可能来自标注协议而非 Crowd-SAM 失效。人工复核界面也应展示原掩码，而不只显示外接框。

EPS 采用均匀随机批采样，因此结果存在顺序随机性；高质量提示先被抽到时能快速覆盖实例，低质量提示先出现则会消耗预算。复现应固定至少三个随机种子，并报告达到 80% 最终 Recall 所需解码数，避免单次幸运顺序夸大效率。

## 优点

- 每个模块对应明确瓶颈，token、采样器和多裁剪消融充分。
- 少量框监督即可在拥挤、遮挡数据上取得强结果。

## 局限

- 最佳版本需 8.1 秒/图，不适合实时检测。
- COCO 多类性能有限，同时依赖两个 ViT-L。

## 评分

- **问题重要性**：★★★★★
- **方法清晰度**：★★★★★
- **实验证据**：★★★★★
- **工程可迁移性**：★★★☆☆
- **YOLO-Agent 参考价值**：★★★★☆
