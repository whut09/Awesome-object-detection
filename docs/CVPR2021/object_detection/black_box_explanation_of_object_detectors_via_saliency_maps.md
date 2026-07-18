---
title: "Black-Box Explanation of Object Detectors via Saliency Maps"
description: "CVPR 2021 论文详解：D-RISE 用随机遮罩和检测向量相似度解释任意黑盒检测器的分类、定位与 objectness。"
tags: ["CVPR 2021", "目标检测", "可解释性", "显著图", "黑盒解释"]
---

# Black-Box Explanation of Object Detectors via Saliency Maps

- **论文页面**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Petsiuk_Black-Box_Explanation_of_Object_Detectors_via_Saliency_Maps_CVPR_2021_paper.html)
- **官方 PDF**：[Paper PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Petsiuk_Black-Box_Explanation_of_Object_Detectors_via_Saliency_Maps_CVPR_2021_paper.pdf)
- **官方代码**：论文仅在脚注列出实验所用 YOLOv3 与 Faster R-CNN 实现，未发现作者声明的 D-RISE 官方仓库。
- **发表会议**：CVPR 2021

## 一句话总结

D-RISE 把待解释框编码成 `位置 + objectness + 类别概率` 的 detection vector，用数千张随机二值 mask 扰动输入，再以“目标框与遮罩后所有 proposal 的最大检测相似度”为权重加权求和 mask，从而无需梯度或模型结构即可解释单个检测、漏检与误定位。

## 研究背景与问题

分类模型通常输出一个类别分数，Grad-CAM、Gradient 或 RISE 可以直接解释这个标量；检测器却同时输出多个 proposal，每个 proposal 又含框坐标、类别分布和可能的 objectness。遮掉一片图像后，原 proposal 可能移动、消失或由另一个 proposal 接替，因此“固定读取某个类别 logit”无法说明指定检测框为何出现。

D-RISE 的关键问题是同时解决“比较什么”和“到哪里比较”：比较对象必须覆盖定位与分类；比较位置不能绑定某个 anchor/index，而应在遮罩后全部候选中寻找与目标 detection 最相似的一项。由此，方法既能解释 YOLOv3，也能解释 Faster R-CNN，甚至能对模型根本没有输出的 ground-truth 框生成诊断图。

## 方法总览

输入为图像 `I`、黑盒检测器 `f` 和一个待解释 detection `d_t`。D-RISE 采样 `N` 张低分辨率二值 mask，双线性上采样并随机裁剪到图像尺寸，得到 `M_i⊙I`；每张遮罩图只需调用一次检测器，输出 proposal 集 `D_i`。

目标 detection 与每个 proposal 都表示成 `[L,O,P]`：`L` 是边界框，`O` 是 objectness，`P` 是类别概率向量。对每张 mask，计算 `d_t` 与 `D_i` 中所有 proposal 的相似度并取最大值作为权重 `w_i`；最终显著图为 `H=Σ_i w_iM_i`。同一批遮罩推理可同时解释图中的多个目标框。

## 方法详解

### 1. RISE Mask 生成

先在 `h×w` 网格上以概率 `p` 独立采样 0/1，随后上采样到略大于输入的尺寸，再随机平移裁剪成 `H×W`，避免固定网格边界。论文多数实验使用 `N=5000、p=0.5、16×16`；精细可视化与平均显著图使用 `30×30`。参数是计算量与空间分辨率的折中。

### 2. Detection Similarity

相似度由三个乘法因子组成：框位置用 `IoU(L_t,L_j)`；类别用目标与 proposal 类别向量的 cosine similarity；YOLOv3 等显式输出 objectness 的模型再乘 `O_j`，Faster R-CNN 则省略该项。乘法实现逻辑 AND：任何一个因素很低，proposal 都不应成为目标检测的替代解释。

### 3. 最大匹配与显著图

对第 `i` 张遮罩，权重是 `max_j s(d_t,d_j)`，而不是固定 proposal 的分数。这样框在遮罩后发生位移时仍能被匹配。权重越高，说明保留该 mask 覆盖区域时目标检测越完整；所有 mask 的加权和即像素重要性。5000 张 mask 在 Tesla V100 上解释一张图的全部检测约需 YOLOv3 70 秒、Faster R-CNN 170 秒。

### 4. 解释检测失败

漏检时直接用 ground-truth 框与 one-hot 类别构造 `d_t`：若显著图仍集中于物体局部，失败可能发生在 NMS 或定位阶段；若相关区域无响应，模型可能根本没学到必要特征。误分类或误定位时分别解释预测框与真值框，再看显著图差异，可定位导致类别混淆或框扩张的上下文。

## 实验与证据

- **数据与模型**：MS-COCO 2017 validation，解释 PyTorch YOLOv3 与 Faster R-CNN；自动评价表针对 YOLOv3 全部检测。基线为 Gradient 和 Grad-CAM，它们只解释 detection vector 中的类别概率。
- **Pointing Game**：按 bbox/mask 真值计算显著峰值命中率，D-RISE 为 0.9656/0.8458，Gradient 为 0.7304/0.5195，Grad-CAM 为 0.5232/0.4209。
- **忠实度指标**：Deletion 越低越好、Insertion 越高越好。D-RISE 为 0.0440/0.5622；Gradient 为 0.0464/0.4561；Grad-CAM 为 0.0762/0.4050。D-RISE 的优势来自同时考虑框、类别和 objectness，而非只读分类分数。
- **Sanity check**：随机化模型参数后，D-RISE 显著图失去可解释结构，说明输出依赖已训练模型而非简单边缘检测。
- **人为偏置实验**：对 COCO 中 fire hydrant/stop sign 的框角加入圆形 marker，并训练 YOLOv3 50 epoch；在无 marker 测试集上两类 mAP 相对下降 10.96%/12.69%。移动 marker 可诱发背景假阳性、hydrant→bottle 误分类和框宽变化，D-RISE 都把 marker 标为原因。
- **用户研究**：比较 COCO mAP 55.3 的 YOLOv3 与 33.1 的 YOLOv3-Tiny，在两者都检测正确的 242 个对象上收集 32 名 Mechanical Turk 用户、每对象 5 次回答；50.2% 认为强模型解释更可信，弱模型为 27.4%。

## 对 YOLO-Agent 的启发

D-RISE 可作为 YOLO-Agent 的黑盒审计器：Agent 无需知道部署模型的梯度或内部层，只要能提交遮罩图并读取框、类别和 objectness，就能检查某次告警依赖目标本体、背景共现还是水印标记。特别适合比较 TensorRT、远程 API 与不同 YOLO 版本的行为一致性。

### 专属 Harness：角标伪相关审计

- **对照组**：A 在原始训练集训练 YOLO；B 在指定两类目标框左上/右上角加入类别固定圆形 marker 后训练；C 与 B 相同但 marker 位置每次随机。三组保持数据划分、epoch、增强和检测阈值一致。
- **观测指标**：无 marker 测试集两类 AP、随机移动 marker 后的假阳性率与框偏移、D-RISE 峰值落在 marker/目标 mask/背景的比例，以及 Deletion/Insertion；同时记录 `N=5000` 的解释时延。
- **通过标准**：B 相对 A 出现可测性能退化时，D-RISE 必须显著提高 marker 区域归因；把 marker 移到背景能复现假阳性或定位变化，删除 D-RISE 高显著 marker 像素后目标 detection similarity 应快速下降；C 的 marker 归因应弱于 B。
- **失败判断**：显著图只覆盖目标类别的所有实例而不能区分指定框；参数随机化后图形基本不变；marker 明显控制预测却未获高归因；或换用等面积无关遮挡得到同样结果，均说明解释器未通过忠实度测试。

## 优点

- 真正黑盒，只依赖输入图像与检测输出，适用于 one-stage、two-stage 和远程模型。
- Detection similarity 同时处理类别、定位与 objectness，并通过最大匹配适应 proposal 身份变化。
- 不只展示漂亮热图，还用自动指标、随机化、人工偏置和用户研究交叉验证用途。

## 局限

- 5000 次遮罩推理非常昂贵，不适合实时解释或大规模逐帧审计。
- 显著图分辨率与 mask 数量强相关，`30×30` 网格在样本不足时会产生 speckle。
- Pointing Game 假设物体内部最重要，但真实检测器可能合理依赖上下文；热图也不能自动给出因果修复方案。

## 评分

- **创新性：8.5/10**：用 detection vector similarity 把 RISE 严格扩展到目标检测。
- **实验充分性：9/10**：定量指标、失败模式、注入偏置和用户研究证据多样。
- **工程可迁移性：8/10**：接口通用，但计算成本限制了使用频率。
- **综合评分：8.6/10**。
