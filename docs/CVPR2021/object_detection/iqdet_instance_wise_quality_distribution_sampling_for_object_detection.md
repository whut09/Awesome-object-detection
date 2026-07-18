---
title: "IQDet: Instance-Wise Quality Distribution Sampling for Object Detection"
description: "IQDet 通过实例级质量分布编码与连续坐标采样，改写密集检测器的正样本分配。"
tags: ["CVPR 2021", "目标检测", "标签分配", "样本采样", "FCOS"]
---

# IQDet: Instance-Wise Quality Distribution Sampling for Object Detection

**论文页面**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_IQDet_Instance-Wise_Quality_Distribution_Sampling_for_Object_Detection_CVPR_2021_paper.html)  
**官方 PDF**：[CVPR 2021 Paper PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_IQDet_Instance-Wise_Quality_Distribution_Sampling_for_Object_Detection_CVPR_2021_paper.pdf)  
**官方代码**：论文正文与 CVF 官方页面未声明独立代码仓库，未发现可核验的官方代码链接。

## 一句话总结

IQDet 从每个 GT 在 FPN 上的 RoIAlign 区域特征预测高斯混合质量分布，再由 Quality Distribution Sampling 在连续坐标上抽取高质量正样本并用分布值作软标签，使 FCOS 的 COCO AP 从 38.7 提升到 41.1，推理时删除全部新增分支。

## 研究背景与问题

密集检测器的标签分配大致沿两条路线演进：RetinaNet 按预设 IoU 阈值逐锚点静态判断，PAA 根据当前预测损失动态划分样本；FCOS、ATSS 又引入实例中心区域或实例统计。IQDet 指出这些方案仍可能是“逐样本”的：质量相近的框会因阈值被硬拆成正负，固定中心也无法适应偏心目标，而且规则网格上的有限正样本会造成质量不足或多样性不足。

论文把输入图像送入 ResNet-FPN 得到 P3–P7；对每个 GT、每个金字塔层用 **RoIAlign → Quality Distribution Encoder（QDE）→ GMM 参数 μ、σ、π** 得到实例级二维质量分布；随后 **Quality Distribution Sampling（QDS）→ 连续坐标重采样 → 双线性插值分类/回归预测 → 分布软标签与框回归目标**。普通 FCOS Class+Box Subnet 仍输出类别、框和 IoU 辅助分数，QDE/QDS 只参与训练。

## 方法总览

QDE 的输入不是单个候选框，而是 GT 覆盖区域的 FPN 特征。RoIAlign 保留空间布局，两个全连接层提取实例表示，再分别预测 K 个高斯分量的位置 μ∈R^(K×2)、尺度 σ∈R^(K×2) 与混合系数 π∈R^K。样本相对 GT 中心的二维偏移先按目标宽高归一化到 [-1,1]，不同 FPN 层独立形成分布，但共享编码器参数。

QDS 按分布概率为每个实例跨金字塔抽取固定数量的浮点坐标，而不是把坐标取整到特征网格。分类图与回归图在这些位置做双线性插值，分类目标直接取质量分布值；GT 外部网格仍作为负样本。回归分支预测采样点到 GT 边界的偏移，并按 FPN 下采样倍率归一化。默认每个实例重采样 12 个正样本，GMM 分量数为 2。

## 方法详解

### 质量分布如何被监督

GMM 不是无监督拟合目标形状。论文用当前预测框与对应 GT 的 IoU 作为网格质量目标，以 BCE 训练 `L_IQ`，因此 QDE 会随检测器状态动态变化。完整目标为 `L_cls + L_reg + L_aux + λ_IQ L_IQ`：分类使用 focal loss，回归使用 IoU loss，辅助 IoU 预测与质量分布监督均用 BCE，`λ_IQ=1`。这使实例整体的空间质量模式平滑掉单点 IoU 噪声，同时保留不同语义区域可能成为质量峰值的多峰表达。

### 训练与推理边界

训练时，GT 框进入 RoIAlign/QDE，分布同时决定采样位置和分类软标签；检测头在规则网格与插值位置上共同接收梯度。推理时没有 GT，QDE、QDS、RoIAlign 和 `L_IQ` 全部移除，只保留 FCOS 式检测头、IoU 辅助分数与 RetinaNet 相同的后处理；每类 NMS IoU 阈值为 0.6，最终保留每图前 100 个检测。因此“cost-free”严格指无新增推理结构，而非训练免费。

### 复现边界

实现中最容易混淆的是“候选数量 K”和“高斯分量数 K”在论文叙述里使用了相近符号：默认 GMM 分量为 2，而表 5 扫描的 4、8、12、16、20 是每个实例重采样的正预测数量。另一个陷阱是把 QDS 写成对现有网格点按分布加权；论文实际先从连续分布采坐标，再对分类图和回归图做双线性插值，因此四个相邻网格会按插值系数收到不同梯度。跨 FPN 层也不能先融合成一张共同分布，因为作者让每层独立估计 μ、σ、π，仅共享 QDE 参数，并把相对位移按该层下采样倍率归一化。若这些细节被省略，结果更接近软标签版 FCOS，而不是 IQDet。

## 实验与证据

主消融在 COCO trainval35k（约 115K 图）训练、val 5K 测试，ResNet-50-FPN、90k iteration、短边 800。原始 FCOS centerness 基线为 **38.7 AP / 57.5 AP50 / 41.7 AP75**；换成 IoU 辅助预测为 39.4 AP；仅 QDE 达 40.4 AP，仅连续 QDS 达 39.6 AP，二者结合为 **41.1 AP / 58.9 AP50 / 44.8 AP75**。高阈值指标提升 3.1 AP75，说明收益确实指向定位质量。

组件证据很具体：RoIAlign 保留空间信息得到 41.1 AP，RoIPool 为 40.6，RoIAlign 后再池化仅 40.1；质量分布取整采样为 40.8 AP，连续坐标双线性插值为 41.1；每实例 K=4/8/12/16/20 时 AP 分别为 40.6/40.6/**41.1**/40.6/40.5。用分类损失、回归损失或组合损失作质量目标只得 40.7/40.6/40.6 AP，IoU 目标最好。

COCO test-dev 的 ResNet-101-FPN IQDet 为 **45.1 AP**，高于同骨干 PAA 的 44.6；ResNeXt-101-DCN 单尺度为 49.0 AP，多尺度为 **51.6 AP**。迁移到 PASCAL VOC 得 59.0 AP，对比 PAA 58.3；WiderFace 得 52.4 AP，对比 ATSS 51.6。

## 对 YOLO-Agent 的启发

YOLO-Agent 可把 IQDet 视为“训练期 assigner 插件”，而不是新检测头：从 YOLO neck 的多尺度特征按 GT 做 RoIAlign，预测实例质量分布，再在分类/回归图上连续取样。最值得迁移的是分布监督与连续采样的联动；若只把分布值乘进 loss，而没有改变采样坐标，就没有复现 QDS 的梯度路径。

**专属 Harness**：明确对照组并固定同一 YOLO 初始化、训练轮数、增强和 NMS，A 为原生 assigner，B 为仅 QDE 软标签，C 为仅连续 QDS，D 为 QDE+QDS；额外做 D-quantized、D-no-RoIAlign 与 K∈{4,8,12,16}。观测 COCO AP、AP75、正样本平均 IoU、每 GT 正样本数、训练显存及推理延迟。通过标准：D 在至少 3 个种子上相对 A 的 AP 与 AP75 同向提升，D 优于 B/C，连续采样优于取整，推理延迟差异落在噪声内；失败判断：增益仅来自更多正样本、关闭 QDE/QDS 后不回落、K 变化无响应，或导出模型仍包含 RoIAlign/QDE。还应核对训练日志中的插值坐标确实为非整数。

## 优点

- 把标签分配从阈值规则提升为 GT 级、预测感知的空间分布建模。
- 连续坐标采样真正改变检测头梯度，而不只是重加权现有网格点。
- 训练辅助分支可完全移除，推理成本与基础检测器一致。
- 多个数据集、骨干和采样目标消融较完整。

## 局限

- 训练依赖 GT RoIAlign、GMM 预测和插值采样，实现复杂度高于普通 assigner。
- “高斯分量对应语义部位”主要来自可视化，缺少显式部件监督。
- 实验基于较早的 FCOS/ATSS 体系，迁移到现代 YOLO head 时需重新校准质量目标和正样本数量。

## 评分

- **创新性：8.5/10**：实例级质量分布与连续采样组合明确。
- **实验充分性：8.5/10**：模块、编码器、坐标、K 值和目标均有消融。
- **YOLO 可迁移性：8/10**：适合训练期 assigner，但需要 RoIAlign 与插值接口。
- **综合：8.4/10**。
