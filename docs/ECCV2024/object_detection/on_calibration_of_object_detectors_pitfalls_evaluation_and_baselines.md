---
title: "[论文解读] On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines"
description: "论文系统分析 D-ECE、AP 与 Temperature Scaling 在检测器校准评估中的陷阱，并提出 LaECE0、LaACE0、LRP 阈值选择及检测专用 Platt Scaling、Isotonic Regression 基线。"
tags: ["ECCV 2024", "目标检测", "模型校准", "后处理"]
---

# On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines

**论文**：[ECCV 官方论文页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3148_ECCV_2024_paper.php)  
**代码**：官方抽取文本列出 `fiveai/detection_calibration`，题目分配未提供可确认的官方代码链接  
**发表**：ECCV 2024  
**类别**：检测器校准、评估协议与后处理

## 一句话总结

论文指出“D-ECE 加 AP、固定置信度阈值、跨域验证集拟合 Temperature Scaling”会对检测器校准得出错误结论，并以 LaECE0/LaACE0 衡量置信度与 IoU 的一致性、以 LRP 为每个模型选择工作阈值，再用类别级 Platt Scaling（PS）或 Isotonic Regression（IR）建立更强且廉价的后校准基线。

## 研究背景与问题

分类校准只需讨论“置信度是否等于正确率”，检测却同时包含类别、定位、重复框和漏检。不同检测器对同一图像输出的框数不同，置信度阈值又直接改变 FP、FN 与定位误差，因此把所有低分框都保留后计算 D-ECE，再另报一个不负责选择工作点的 AP，不能代表部署时的联合性能。

论文逐项指出常见协议的缺陷。D-ECE 只让置信度拟合 precision，忽略真阳性框的 IoU；LaECE 虽加入 IoU，却在正样本判定阈值以下把检测直接视为零质量，丢失“这个框仍有一定重叠”的细粒度信息；AP 随保留检测数量增加而倾向上升，不适合交叉验证部署阈值；用 Objects365、BDD100K 等域偏移验证集拟合校准器和阈值，还会让测试时的工作点失配。

## 方法总览

作者提出一套联合评估流程：从同分布验证集取得检测结果；用 LRP 分别选择校准前筛选阈值与校准后工作阈值；按类别拟合单调 PS 或 IR；在同分布和域偏移测试集同时报告 LaECE0、LaACE0、LRP，并保留 AP 作为排序精度参考。核心目标不是只把一个误差数字压低，而是让输出置信度具有“预测框与其匹配真值的 IoU”这一明确语义。

## 方法详解

### 1. 从 D-ECE 到定位感知误差

设检测 $i$ 的类别、框和置信度为 $(\hat c_i,\hat b_i,\hat p_i)$，匹配函数 $\psi(i)$ 返回对应真值框；FP 的 IoU 定义为零。论文令匹配阈值 $\tau=0$，要求

$$\mathbb{E}[\operatorname{IoU}(\hat b_i,b_{\psi(i)})\mid \hat p_i=p]=p.$$

按类别和等宽置信度区间计算得到

$$\operatorname{LaECE}_0=\frac{1}{K}\sum_{c=1}^{K}\sum_{j=1}^{J}\frac{|\hat D_j^c|}{|\hat D^c|}\left|\bar p_j^c-\overline{\operatorname{IoU}}_j^c\right|,$$

其中 $K$ 是类别数，$J=25$，$\hat D_j^c$ 是类别 $c$ 在第 $j$ 个区间的检测集合。LaACE0 采用每个样本单独成箱的极端自适应分箱：

$$\operatorname{LaACE}_0=\frac{1}{K}\sum_c\frac{1}{|\hat D^c|}\sum_i\left|\hat p_i-\operatorname{IoU}(\hat b_i,b_{\psi(i)})\right|.$$

两者都在 $\hat p_i=\operatorname{IoU}$ 时达到最小，但 LaACE0 更直接反映逐框绝对误差。

### 2. 用 LRP 选择工作点

LRP 将 FP 数、FN 数和真阳性定位误差 $E_{loc}(i)$ 合并：

$$\operatorname{LRP}=\frac{N_{FP}+N_{FN}+\sum_{\psi(i)>0}E_{loc}(i)}{N_{FP}+N_{FN}+N_{TP}}.$$

与 AP 不同，LRP 会同时惩罚低 precision、低 recall 和差定位，因而其曲线存在适合部署的最优阈值。论文要求每个检测器独立在同分布验证集上选阈值，避免用同一固定阈值比较输出规模不同的模型。

### 3. 检测专用 PS 与 IR

对类别 $c$，先用 LRP 选校准输入阈值 $\bar u_c$，仅保留更接近推理分布的检测，再拟合类别专用映射 $\zeta^c$；校准后再用 LRP 选输出阈值 $\bar v_c$。Platt Scaling 为

$$\hat p_i^{cal}=\sigma(a\sigma^{-1}(\hat p_i)+b),\quad a\ge 0,$$

其中 $a$ 为尺度、$b$ 为偏置，目标以 IoU 作为软标签优化二元交叉熵。IR 则直接用 $(\hat p_i,\operatorname{IoU}_i)$ 对拟合单调分段线性回归，不限定为单一 sigmoid 形状，也尽量保持原有排序。

论文还强调校准目标必须与评价指标对齐：若最终比较 D-ECE，就应把 TP/FP 指示量作为目标；若比较 LaECE0 或 LaACE0，就应把匹配 IoU 作为目标。PS 与 IR 不是固定配方，而是同一数据构造、阈值选择和目标定义框架下的两种单调映射器。

由于两种映射都保持单调，原检测排序不会被任意打乱；真正改变输出集合的是按类别交叉验证得到的校准前后阈值。这使“分数映射”和“部署工作点”成为两个可分别审计的步骤。

## 实验与证据

- **数据**：构建 Common Objects（COCO）、Autonomous Driving（Cityscapes）和 Long-tailed Objects（LVIS）设置；同分布验证/测试来自同一分布，域偏移测试包含 COCO-C、Obj45K、Foggy Cityscapes 等。
- **主要基线**：未校准检测器、Temperature Scaling，以及训练期 MbLS、MDCA、TCD、BPC、Cal-DETR；还覆盖 PAA、ATSS、GFL、VFNet、Faster R-CNN、D-DETR、DINO、GLIP、Grounding DINO、Co-DETR、EVA、MoCaE 等 14 个检测器。
- **对 Cal-DETR**：在 COCO minitest 上，面向 D-ECE 优化的 PS 得到 0.9 D-ECE，而 Cal-DETR 为 8.7；面向 LaECE 优化的 IR 在 LRP 阈值下取得 8.2 LaECE，Cal-DETR 为 11.8，同时检测精度基本保持。
- **验证集陷阱**：D-DETR 的 IR 若用 Objects365 验证集，D-ECE 为 14.2；改用同分布 COCO 验证集后为 1.3，AP 均为 44.1。
- **广泛适用性**：14 个检测器未经校准时 LaECE0 范围为 10.6–34.4，IR 后收敛到 6.7–8.9，且 LRP 基本不变。UP-DETR 的 LaECE0 从 34.4 降至 8.2。
- **消融**：类别级校准、检测阈值和 PS 偏置项均有贡献；COCO 上 PS 为 9.6 LaECE0、IR 为 7.7，当前 TS 基线为 12.3。域偏移验证集可使 LRP 最多恶化 4。

## 对 YOLO-Agent 的启发

YOLO-Agent 可把该方法实现为训练后独立的 `calibration` 阶段：冻结 YOLO，保存同分布验证集上 NMS 后的类别、框、分数与真值匹配；对每类用 LRP 搜索 $\bar u_c$，分别拟合 TS、PS、IR，再搜索 $\bar v_c$。对照组必须包含未校准、论文指出的 TS 基线、带偏置 PS 和 IR；报告 LaECE0、LaACE0、LRP 与 AP，不能只报 ECE 或只看可靠性图。

失败阈值应直接对齐论文证据：若 IR 不能把未校准模型的 LaECE0 从论文观察到的 10.6–34.4 区间压到 6.7–8.9，或像 D-DETR 那样未能在 LRP 约 57.3 基本保持时从 12.7 降到 7.7，则不应把校准器合入默认部署链路；若跨域验证集导致 LRP 接近论文报告的 4 点退化，应立即改用同分布验证集重新选阈值。

## 优点

- 同时修正指标、阈值、数据划分和后校准基线，而非只提出一个新损失。
- PS 与 IR 训练成本低、模型无关，适合作为任何校准研究的强基线。
- LaECE0/LaACE0 让分数语义具体化为 IoU，便于下游风险系统解释。

## 局限

- 类别级校准需要足够的同分布验证检测，稀有类别可能难以稳定拟合。
- 后校准无法修复框的位置与召回，只重新映射分数并选择工作点。
- 大幅域偏移时，基于同分布验证集学习的映射仍可能失效，论文在 Obj45K 上也观察到 LaECE0 略差。

## 评分

- **问题定义：高**：清楚拆解检测校准评估中的多重混淆。
- **实验覆盖：高**：跨数据集、检测范式、校准方法与域偏移验证充分。
- **工程价值：高**：PS/IR 和 LRP 阈值流程可直接落地。
- **综合评价：强烈推荐**：是构建 YOLO 置信度校准基准时应优先采用的协议论文。
