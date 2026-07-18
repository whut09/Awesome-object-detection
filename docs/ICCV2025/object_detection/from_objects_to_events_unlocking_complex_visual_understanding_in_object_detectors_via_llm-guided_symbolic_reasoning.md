---
title: "From Objects to Events: Unlocking Complex Visual Understanding in Object Detectors via LLM-guided Symbolic Reasoning"
description: "详解 SymbolicDet 如何把开放词汇检测结果转化为可解释的事件逻辑表达式。"
tags: ["ICCV 2025", "目标检测", "符号推理", "大语言模型", "事件理解"]
---

# From Objects to Events: Unlocking Complex Visual Understanding in Object Detectors via LLM-guided Symbolic Reasoning

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_From_Objects_to_Events_Unlocking_Complex_Visual_Understanding_in_Object_ICCV_2025_paper.html)  
**代码**：[作者声明的官方代码](https://github.com/p1atdev/SymbolicDet)

## 一句话总结

SymbolicDet 不训练新的事件网络，而把 APE、GLIP 或 YOLO-World 的实体类别、数量、框关系与置信度转换为结构化特征，再由 Symbolic Logic Search（SLS）进化逻辑表达式，并让 LLM 以场景上下文、思维链和历史评分迭代引导搜索，从“检测到什么”推出“正在发生什么”。

## 研究背景与问题

检测器能定位 person、helmet、fishing rod，却不能直接判断工人未戴安全帽或一人使用多根鱼竿。专用事件模型需要大量任务标注，且决策难解释。直接问多模态 LLM 又容易产生不稳定的自然语言结论。论文希望只利用标准 detector 输出，发现可执行、可评分、可人工检查的符号规则，并保持训练无关和 detector 无关。

## 方法总览

开放词汇检测器先根据 LLM 扩展的对象提示生成 `Di={(cj,bj,sj)}`。Structural feature extractor 计算实体数量、空间关系与属性分布，形成 `X`。SLS 在数学算子 `+,-,×,÷,max,min` 和逻辑算子 `∧,∨,¬` 构成的表达式空间中，以分类损失和复杂度惩罚评价候选。每轮最佳表达式进入 LLM structured prompt space，LLM 给出语义建议，新表达式被混入 population 继续 mutation、crossover 和 selection。

## 方法详解

Symbolic Pattern Discovery 通过进化算法搜索 `f*`，目标是在正负事件样本上提高区分能力，同时用 `Ω(f)` 偏好短规则。例如安全帽场景可以比较 person/head 数量与 helmet 数量，而非法多杆垂钓则需要 person、rod 的计数和相对关系。规则最终直接作用于 detector 输出，不修改视觉骨干。

Automated LLM Reasoning 包含三层提示：Scene Context Initialization 提供场景、实体和约束；Chain-of-Thought Guidance 规定分析步骤；Contextual Feedback Integration 带入历史表达式及其分数。每次迭代选择当前最佳 `f*t`，LLM 生成建议 `St`，再由客观数据评分约束建议，避免只凭语言常识拍板。

## 实验与证据

自建 Multi-Event Dataset 超过 110,000 张图，论文重点使用 2,283 张多杆垂钓测试图；Helmet-Mac 有 7,571 张训练图、4,642 张测试图。公共评估包含 ERA 的 BALL、PersonCrowd、Sport 与 UCSD Ped2。主要 detector 为 APE，并比较 GLIP、YOLO-World；APE 阈值设 0.05，另两者 0.1。符号 population 为目标类别数两倍，crossover/mutation 因子 0.5/0.3，最多搜索 5,000 次，LLM 使用 Qwen 系列。

APE 加 SymbolicDet 后，BALL AUROC 从 55.36 升至 94.91，Sport 从 67.13 升至 90.29，Helmet-Mac 从 67.41 升至 83.18，多杆垂钓从 66.82 升至 75.16。YOLO-World 在 PersonCrowd 从 55.00 升至 85.11，说明机制不依赖单一 detector。训练自由版本在 Helmet-Mac/多杆场景为 83.18/75.16，与 LoRA、Prompt tuning 组合结果接近。UCSD Ped2 达 98.7%。正文总结的消融指出，符号模式发现相对手写逻辑提升 18.36%，LLM 同时改善准确率与收敛效率，搜索规模扩大时性能继续提高。

论文在 detector prompt 上也使用 LLM，但目的不同于符号搜索。第一阶段让 LLM 根据事件描述补全直接对象和上下文对象，例如安全帽场景不仅检测 helmet、head、person，也加入 workshop、construction site、scaffolding、hand、gloves。开放词汇检测器因此提供更完整的实体集合；第二阶段的 LLM 才读取当前最佳表达式和评分，提出逻辑结构修改。两次调用应分别缓存和评估，避免把检测漏项与推理错误混为一谈。

SLS 的 population size 设置为目标类别数的两倍，每轮选 top-4 做交叉与变异，最多 5,000 轮。对象检测运行在 RTX 4090，符号回归运行在 Xeon Silver 4214R CPU，说明大量逻辑搜索可以与视觉推理解耦离线进行。论文所谓 training-free 指不更新 detector/LLM 参数，而不是零计算；若事件定义频繁变化，搜索和 LLM 调用成本仍需预算。

数据规模差异很大：Multi-Event 总量超过 110,000，但多杆测试只含 1,098 异常和 1,185 正常；Helmet-Mac 测试为 2,276 违规与 2,366 合规；ERA 的三个事件子集只有数百张。AUROC 对类别平衡较稳健，但规则仍可能学习拍摄背景而非事件关系。跨数据源测试和实体遮挡扰动应作为额外证据。

表 2 的排列需要谨慎解读。APE 原始 Helmet-Mac/多杆为 67.41/66.82，单独 SymbolicDet 为 83.18/75.16；LoRA 加 SymbolicDet 可到 95.67/78.44，Prompt tuning 加 SymbolicDet 为 81.62/76.06。训练自由方法“可比微调”成立于部分配置，但最强 Helmet-Mac 结果仍来自 LoRA 与符号模块叠加，二者不是互斥方案。

可解释性来自最终表达式而非 LLM 文本。图 1 的安全规则组合 Head、Helmet、Person、Workshop、Construction Site、Hand、Gloves、Scaffolding 及比较、与、或运算，并附自然语言解释。真正执行告警的是符号表达式，LLM 解释只用于人类检查。若解释与表达式语义冲突，应以规则在样本上的实际 truth table 为准，并把冲突记录为失败。

架构无关性也有程度差异。APE 在五个事件上通常最高，但 YOLO-World 与 GLIP 加模块后也取得 71.11–90.27 AUROC；原 detector 越弱，符号提升有时越大，例如 YOLO-World PersonCrowd 增加 30.11。规则可以补偿实体级分数聚合不足，却无法恢复完全漏检的对象。因此应同步报告 detector entity recall，防止把感知瓶颈误归因于逻辑搜索。

## 对 YOLO-Agent 的启发

YOLO-Agent 可把检测输出转成可审计事件 DSL，而不是要求 detector 学会所有高阶行为。适合的代理流程是：自动扩展开放词汇提示，缓存实体图，搜索短规则，调用 LLM 解释并提出邻域表达式，最后由验证集 AUROC 决定是否接受。规则复杂度、搜索时间和跨 detector 迁移应与准确率共同优化。

### 论文专属 Harness

- **对照组**：同一 YOLO-World 输出上比较对象分数阈值、人工计数规则、纯遗传 SLS、LLM 直接分类、完整 LLM-guided SLS。
- **观测指标**：事件 AUROC/F1、规则长度、搜索迭代数、LLM 调用数、跨场景与跨 detector 性能、同规则重复运行方差。
- **通过阈值**：完整方法相对人工规则提升至少 8 AUROC，且相对纯 SLS 用少 30% 迭代达到同分数；换 APE/GLIP/YOLO-World 后至少两者仍增益 5 点。
- **失败判断**：规则依赖数据集偶然对象、三次运行方差超过 3 AUROC，或 LLM 建议无法被符号评分稳定复现，则禁止上线事件告警。

## 优点

- 规则显式、可执行、可解释，不需要微调视觉模型。
- 三种开放词汇 detector 和多类事件都获得明显增益。

## 局限

- 搜索仍需要带事件标签的正负样本，并非完全零数据。
- 复杂时序事件被压缩为图像级实体关系，表达能力有限；部分消融仅在补充材料详述。

## 评分

- **问题重要性**：★★★★★
- **方法清晰度**：★★★★☆
- **实验证据**：★★★★☆
- **工程可迁移性**：★★★★☆
- **YOLO-Agent 参考价值**：★★★★★
