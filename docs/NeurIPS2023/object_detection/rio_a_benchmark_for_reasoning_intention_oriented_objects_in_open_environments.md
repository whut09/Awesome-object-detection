---

# RIO: A Benchmark for Reasoning Intention-Oriented Objects in Open Environments
title: "RIO：开放环境中面向意图对象推理的基准"
description: "面向自然语言意图的目标检测与实例分割数据集笔记，涵盖 RIO 数据构建、Detector-Reasoner 与端到端基线、评测结果、消融实验及 YOLO-Agent 验证方案。"
tags:
  - NeurIPS-2023
  - 意图推理
  - 目标检测
  - 实例分割
  - 视觉语言
  - 开放环境
---

## 一句话总结

这项工作的核心不是按类别名找物体，而是输入场景图像与不含目标名的自然句子，如“找一个能让我躺下休息的东西”，输出最符合意图的目标框或实例掩码。作者以 COCO2014 图像及实例标注为基础构建 RIO，数据流依次为 **Step1 Image Selection → Step2 Intention Generation → Step3 Manual Selection → Step4 Sentence Modification**：先保留含三类以上实例的复杂场景，再将图像 Caption 与候选类别交给 GPT-3.5 生成交互描述，由人工判断该类别是否“合理且最佳”，最后删除句中的类别名，把“You can use the sofa to…”改成“Something to…”。

RIO 将标注分为 common 与 uncommon。common 描述来自“在 Caption 所述场景中能用该对象做什么”；uncommon 则先建立可替代类别映射，再询问某对象能否临时代替另一类别，例如让 spoon 代替 knife 切软食物。最终得到 40,214 张图像、130,585 个意图—对象对：训练集为 27,696 张图像、96,415 对，common 测试集为 12,392 张、29,344 对，uncommon 测试集为 4,826 张、4,248 对；每个类别的描述平均出现 158 个动词，约 300 个动词会跨越十个以上类别，因此无法依靠固定“动词—名词”查表完成任务。

作者比较 Detector-Reasoner 与端到端路线。D-R 先以 COCO 类别列表驱动 GroundingDINO，经过分数阈值筛选得到 `[box_i, label_i]`，再把候选标签及意图送入 ChatGPT 选择类别，最后返回该类别的框；端到端基线则包括 MDETR、TOIST、SeqTR、Polyformer。IOOD 使用 AP@0.5 与 Top1-Accuracy，IOIS 使用 mIoU 与 Top1-IoU。关键消融显示，MDETR 将置信度只对齐到“something”时 AP@0.5 为 48.16，而平均分配给整句仅为 27.6；去掉预训练更跌至 2.57，证明名词定位预训练与正确的代词对齐是该任务的关键。

## 研究背景与问题

传统 Affordance Detection 通常把意图压缩为有限动词，Task-Driven Object Detection 也依赖预定义任务。例如“play”既可能对应飞盘，也可能对应钢琴，单个动词不能表达场景、用途和优先级。REC 虽接受自然语言，却主要依赖类别、属性和物体关系，句中通常直接存在目标名；RIO 刻意移除类别名，要求模型从行为目的和环境中反推出对象。

RIO 相比 ADE-Aff、PAD、PADv2、PAD-L、COCO-Tasks，不仅规模更大，而且具有超过 100 种 affordance、69 个对象类别和场景相关的自然句子。它还强调“最佳对象”而非“勉强可用”：同一场景中 chair 与 sofa 都可供阅读，但若 sofa 更舒适，chair 对应的描述会在测试标注中被删除。

## 方法总览

common 数据的生成先过滤 person 与多数 food 类别：前者是意图发起者，后者功能往往集中于“吃”，难以提供足够多样的交互。人工审核分为 Fit-and-Best、Not-the-Best、Unfit；十名标注者各处理约九千条数据，约二十天完成，一致率为 93.7%。训练集保留一定类别级噪声与频率统计，但验证和测试集执行更严格的最佳性筛选。

uncommon 集专门测试少见替代功能，例如叉子或勺子切软食物、杯子临时代替花瓶、长椅充当睡眠表面。这些样本并非随机长尾切分，而是通过类别替代关系和场景 Caption 定向生成，因此更直接检验模型能否从常见功能迁移到罕见意图。

## 方法详解

common IOOD 上，D-R 的 AP@0.5/Top1-Acc 为 38.30/58.21，使用 COCO 微调 GroundingDINO 的 D-R* 提升至 45.15/64.77；MDETR 为 48.61/65.05，TOIST 为 49.05/66.72，SeqTR 的 Top1-Acc 最高，为 67.44。common IOIS 中，Polyformer 的 mIoU 为 48.75，高于 TOIST 的 45.07 与 MDETR 的 44.14。

uncommon 集出现明显退化：D-R* 的 AP@0.5 仅 17.25，MDETR 为 24.20，TOIST 为 21.96；SeqTR 的 Top1-Acc 从 67.44 跌至 23.24。IOIS 中 Polyformer 的 mIoU 从 48.75 降至 26.77，TOIST 从 45.07 降至 19.41。TOIST 的名词—代词蒸馏改善 common，却没有带来更强的罕见功能迁移。

## 实验与证据

MDETR 保持原模型与损失函数，只把真实对象和句中的“something”建立对应。TOIST 先训练含类别名的句子，如“You can use the knife to cut cakes”，再将知识蒸馏到“You can use something to cut cakes”。SeqTR 将框表示为左上角与右下角坐标序列，但只能预测单一目标，因此训练时选择满足意图的最大实例。Polyformer进一步输出多个多边形，可覆盖遮挡后分裂的实例及多个符合意图的对象。

GroundingDINO 的同义词消融揭示了类别接口的脆弱性：使用原始 COCO 类别名时 AP/AP50 为 48.4/64.6，替换为常见同义词后降至 35.6/47.5。端到端视觉—语言对齐避免了“检测错误→类别命名错误→LLM 推理错误”的级联传播，更符合不依赖固定命名体系的目标。

## 对 YOLO-Agent 的启发

若将 RIO 接入 YOLO-Agent，可让 YOLO 负责开放词汇候选框生成，Agent 负责意图解释、场景约束与候选排序，但不能只报告最终选中框。系统应记录候选框、类别词、检测置信度、Agent 排序分数和拒答原因，以区分漏检、错误命名、推理错误及“多个可用对象中优先级错误”。

更可靠的方向是让 Agent 生成候选功能概念而非唯一类别名，并对视觉区域逐一验证“可行性”和“最佳性”。针对 uncommon 样本，还可引入替代用途知识，但必须避免 Agent 根据语言常识选择图中不存在的对象。

## 优点

Harness 至少设置四组：①仅 YOLO 候选框与类别相似度；②YOLO＋LLM 类别选择，复现 D-R；③加入场景 Caption；④加入区域视觉特征与最佳性重排。另设 oracle-box 组，把真实候选框交给 Agent，以隔离检测器上限；设 oracle-reasoner 组，把真实目标类别交给检测器，以测量定位误差。

指标严格沿用 IOOD 的 AP@0.5、Top1-Accuracy，并分别报告 common、uncommon；多实例扩展使用 IOIS 的 mIoU、Top1-IoU。同时统计候选召回率、Agent 在“目标已进入候选集”条件下的选择准确率，以及 common→uncommon 的相对性能下降。

## 局限

具体失败标准为：预测框与任一真实框 IoU 不超过 0.5；目标已在候选集中但 Agent 选择次优类别；只找到部分符合意图的实例；最高分未达到阈值而输出空集；或选择虽可完成动作却不是人工标注的 Fit-and-Best 对象。uncommon Top1-Accuracy 若不高于 SeqTR 的 23.24，或 AP@0.5 不高于 MDETR 的 24.20，则不能声称提升了罕见意图推理。

## 评分

- 论文：[NeurIPS 2023 官方论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html)
- 官方代码：论文正文未提供作者 GitHub URL，无法确认存在官方代码仓库；请以[官方论文页及其补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8644353f7d307baaf29bc1e56fe8e0ec-Abstract-Datasets_and_Benchmarks.html)为准。
- 结论：RIO 的主要价值不在提出新网络，而在把对象识别从“它叫什么”改写为“在当前环境中，什么最适合完成我的意图”，并用 uncommon 集揭示现有模型仍严重依赖常见外观和功能共现。
