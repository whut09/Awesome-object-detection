---
title: "[论文解读] Benchmarking Object Detectors with COCO: A New Path Forward"
description: "解析 COCO-ReM 如何用 SAM、LVIS 与人工核验修复 COCO-2017 实例掩码并改变模型排名。"
tags: ["ECCV 2024", "目标检测", "实例分割", "COCO-ReM", "数据质量"]
---

# Benchmarking Object Detectors with COCO: A New Path Forward

**论文**：https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6053_ECCV_2024_paper.php  
**官方代码**：未发现论文声明的官方代码；官方数据集页面为 https://cocorem.xyz  
**会议**：ECCV 2024

## 一句话总结

COCO-ReM 通过 SAM 边界细化、LVIS/检测器补全实例和人工纠错，重做 COCO-2017 的训练与验证实例掩码，证明原标注会惩罚更锐利的预测、改变五十个模型排名，而干净训练标注还能让 ViTDet-B 更快收敛并接近三倍参数的 ViTDet-L。

## 研究背景与问题

COCO 多边形 mask 存在粗边界、孔洞与遮挡处理不一致、实例漏标、误标、重复 mask 和 crowd 错误。早期模型能力有限时这些噪声可能不是瓶颈；当 SAM、Mask2Former、ViTDet 能预测更细边缘时，预测比真值更准确反而降低 IoU，导致 benchmark 与人工判断错位。

作者保持 COCO 图像、80 类和评测接口，只替换实例标注，以检验数据质量能否改变“哪个模型更好”和“怎样训练更有效”。

## 方法总览

半自动流水线有三阶段。Stage 1 用 COCO mask 采样点/框提示 SAM，细化边界与孔洞，并人工检查验证集每个 mask；Stage 2 从 LVIS 导入同图同类但更完整的实例，再借助 LVIS 训练检测器发现漏标候选并人工确认；Stage 3 人工删除重复标注、修正类别，并把成组实例标为 crowd。训练集主要走自动流程，验证集彻底核验。

## 方法详解

作者处理约 86 万训练实例和约 3.6 万验证实例。Stage 1 的 SAM 自动结果中，仅约 900 个验证 mask 被判低质量，占 36K 的 2.4%，再通过交互点提示修正。导入 LVIS 时，只在 LVIS 对 `(image, category)` 的实例数多于 COCO 且该对未被标为非穷尽时替换，避免把未确认类别当成完整真值。

最终 COCO-ReM 仍使用标准 COCO AP，但 mask 更贴合可见区域、正确保留孔洞并统一遮挡规则。它不是检测模块，而是重新定义可靠证据：同一 checkpoint 在 COCO-2017 与 COCO-ReM 的差异可揭示模型是否迎合旧边界偏差。

## 实验与证据

作者评估五十个检测/实例分割模型，包括 Mask R-CNN、Cascade R-CNN、Mask2Former、OneFormer、ViTDet。高质量掩码模型在 COCO-ReM 上得分更高，排名显著变化；69/80 类平均 AP 上升，carrot、orange、fork 等类别提升明显。bowl、dining table、scissors 等带孔洞或重遮挡类别降低，说明 COCO-2017 训练使模型学会填孔或吞并遮挡物。

消融以 Cascade ViTDet-B/L 和 Mask2Former Swin-B/L 四模型平均：COCO-2017 AP 49.6；仅边界细化 ReM-S1 为 57.3；导入 LVIS 后 ReM-S2a 为 55.6；最终 ReM 为 55.5。最大变化来自 Stage 1，证明粗边界是主噪声。训练保持官方 ViTDet 超参数，用 8 张 A40 与梯度累积模拟 batch 64；COCO-ReM 训练的 Mask R-CNN ViTDet-B 约 54.5 AP，对应旧标注约 50.5；Cascade ViTDet-B 为 55.4 对 51.4，并接近 3 倍规模 ViTDet-L 的 56.3。

## 对 YOLO-Agent 的启发

Agent 比较策略前应先做标注可信度审计。对 YOLO-seg 可用高质量分割模型检查边界、孔洞和漏标；对 box 数据可借助实例 mask 判断框内可见区域和遮挡规则。模型应同时在原真值与精修真值上排序，避免把迎合脏边界当成改进。

### Harness

对照组为原 COCO、仅 SAM 边界精修、精修+漏标补全、最终人工核验；固定同一组 YOLO-seg/Mask R-CNN checkpoint，不重调阈值。观测模型排名 Kendall 相关、mask AP、boundary AP、漏标诱发假 FP、孔洞/遮挡类别 AP，以及达到同 AP 所需 epoch。通过条件：精修集人工抽检错误率低于 1%，boundary AP 与人工偏好一致率提升至少 10%，训练达到原最终 AP 的 epoch 减少 20%；若排名变化只来自实例数量而边界指标不改善，则失败。

## 优点

- 保持 COCO 图像与接口，可直接复评历史模型。
- 自动效率与验证集人工质量控制结合。
- 同时证明评测标注和训练标注限制进步。
- 五十模型和训练消融证据广泛。

## 局限

- 重点是实例 mask，未解决全部框与类别体系问题。
- 训练集未逐实例人工核验。
- 新 AP 与旧 COCO 数字的直接可比性降低。

## 评分

| 维度 | 评分 |
|---|---:|
| 数据贡献 | 9.5/10 |
| 评测价值 | 9.5/10 |
| 实验证据 | 9.0/10 |
| 数据审计价值 | 9.0/10 |
| 综合 | 9.3/10 |
