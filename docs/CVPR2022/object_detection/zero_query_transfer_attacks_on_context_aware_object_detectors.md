---
title: "Zero-Query Transfer Attacks on Context-Aware Object Detectors"
description: "解读 ZQA 如何用对象共现图规划上下文一致的多目标错误标签，并用 PSPM 在零反馈条件下筛选可迁移攻击方案。"
tags: ["CVPR 2022", "目标检测安全", "黑盒攻击", "上下文建模", "PSPM"]
---

# Zero-Query Transfer Attacks on Context-Aware Object Detectors

**官方论文**：[CVF 论文页](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Zero-Query_Transfer_Attacks_on_Context-Aware_Object_Detectors_CVPR_2022_paper.html) · [官方 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_Zero-Query_Transfer_Attacks_on_Context-Aware_Object_Detectors_CVPR_2022_paper.pdf)  
**官方代码**：论文未声明作者公开实现，CVF 页面和正文也没有仓库地址；本文只按公开算法、数据和表格讨论可复现实验，不把未披露工程细节视为已知。

## 一句话总结

ZQA 不去逐次试探受害检测器，而是先把场景内多个对象一起改写成仍能在共现图中组成团的标签集合，再利用离线 Perturbation Success Probability Matrix 选择最可能迁移成功的映射，最后用一次有界 PGD 提交同时绕过分类与上下文一致性检查。

## 研究背景与问题

普通目标攻击常把某个对象改成任意错误类别；在多对象自然场景中，这可能产生“boat 与 stop-sign 同现”之类训练数据从未出现的组合，因此基于上下文的防御可以直接报警。已有上下文攻击可在每次失败后查询 victim、逐步增加被修改对象，但真实系统可能不给输出，重复通信也会暴露攻击行为。论文研究的是更严格的 zero-query transfer：攻击者只能利用 surrogate 模型和已知数据分布离线规划，最终只有一次提交机会，而且成功必须同时满足目标对象变成指定类别、整组检测标签仍通过防御端的上下文检查。

## 方法总览

系统先从训练集建立类别共现矩阵 `G`，归一化后得到条件共现概率，并把非零或超过阈值 `η` 的边解释为上下文图。给定原场景对象集合 `A` 和一个指定的 victim→target 映射，算法找出所有与 target 共现的候选标签；场景中其余对象也被重新指派到该候选集合，使最终标签 `B` 形成完全连通子图。由于可行方案很多，作者额外预计算 PSPM `M(C, ε, α)`：矩阵元素表示在 surrogate 模型集合 `C`、扰动预算 `ε`、算法 `α` 下，把类别 `i` 改成 `j` 的白盒成功概率。ZQA-PSPM 对每个原对象选择矩阵概率最大的上下文合法目标，再一次性生成整图扰动。

## 方法详解

### 共现约束定义攻击是否“像一个场景”

共现图的节点是 VOC/COCO 类别，边权是训练图像中类别对共同出现的次数；行归一化后可读作观察到类别 `i` 时出现 `j` 的条件概率。论文的严格一致性要求所有输出标签对应节点两两有边，即形成 clique；广义版本先用阈值 `η` 删除弱边。举例来说，原始 `person-crosswalk-stop-sign` 合法，若只把 crosswalk 改成 boat 会因 boat 与 stop-sign 无边而失败；再把 stop-sign 改为 water 后，`person-boat-water` 可重新成为一致组合。

### PSPM 把可行计划按攻击难度重新排序

仅看共现图不知道“horse→bicycle”和“horse→cat”哪个更容易在当前 surrogate 与预算下实现。PSPM 对所有源类—目标类组合实际统计定向扰动成功率，因此把语义可行性与视觉可攻击性分开建模。论文采用逐对象最大化矩阵元素的简单策略，也讨论了最大化平均值或最小值的替代准则。得到标签计划后，以整图零初始化扰动运行 targeted PGD：步长 `λ=2`、最多 50 次迭代，并投影到 `L∞≤ε` 与合法像素范围；与 victim 框 IoU≥0.3 的邻近对象统一映射到 victim 的 target，以免重叠区域的不同梯度互相破坏。

## 实验与证据

- 主要结果从 VOC2007 test 随机取 500 张含 2–6 个对象的图；补充材料使用 COCO2017 val。surrogate/victim 覆盖 Faster R-CNN、RetinaNet、Libra R-CNN、FoveaBox，四者在 VOC 上的 mAP@0.5 为 78.30/78.51/79.01/77.68。
- 评价指标是 context-consistent fooling rate：指定 victim 必须被识别为 target，且全部检测标签必须通过 clique 一致性检查，缺一条件都算失败。
- 以 Faster R-CNN 为白盒、`ε=50` 时，Context-Agnostic、ZQA、ZQA-PSPM 的白盒成功率分别为 34.0%、90.0%、92.6%；迁移到 RetinaNet/Libra/FoveaBox 时，ZQA-PSPM 为 51.2%/61.6%/56.8%，均显著高于 29.0%/30.0%/25.4% 的无上下文方案。
- 同一设置下，ZQA-PSPM 在白盒高于累计到 4 次查询的 Few-Query 91.6%，在三个黑盒上也高于或接近 2–3 次查询方案；论文归纳为白盒可对标 5-query、黑盒可压过最多 3-query。
- PSPM 消融可直接由 ZQA 与 ZQA-PSPM 比较：`ε=50` 的 Faster R-CNN→Libra 迁移从 52.2% 升至 61.6%；当 Libra 作为白盒时，对 Faster R-CNN 的迁移由 47.8% 升至 51.0%。预算降到 `ε=10` 后成功率明显下滑，前一组白盒 ZQA-PSPM 仍为 70.6%，但三个黑盒仅 23.2%/27.4%/28.2%。

## 对 YOLO-Agent 的启发

对 YOLO-Agent，最合理的用途是建立防御回归测试，而非把攻击策略加入生产推理。**Harness** 应在隔离数据上固定一批 2–6 目标场景，以 YOLO 为 victim，并选择 Faster R-CNN 或另一 YOLO 变体作为 surrogate。**对照组**包括单对象 targeted PGD、Context-Agnostic 多对象计划、随机上下文合法 ZQA、PSPM 引导 ZQA，以及能观察 victim 输出的 1/3-query 上界。**指标**：逐个 `ε∈{10,20,30,40,50}` 报告目标转换率、上下文通过率、二者联合成功率、LPIPS/SSIM、跨架构迁移率和防御报警率，并补做“双方共现图相同/不同”“只用共现/加入相对尺度与空间关系”两项压力测试。**失败判断**：PSPM 相对随机合法计划的联合成功率提升不足 3 个百分点、换 surrogate 后优势消失，或简单更新防御图即可把成功率压回单对象攻击水平，均说明该威胁不具稳定迁移性；反之必须把多对象联合异常纳入上线门禁。

## 优点

- 明确区分语义合法性与视觉可扰动性，PSPM 为零查询计划提供了可计算的优先级。
- 同时覆盖两阶段、一阶段和 anchor-free detector，证明现象不局限于单一架构。
- 成功条件包含上下文防御，而不是只报告目标框分类翻转，评价与威胁模型一致。

## 局限

- 假设攻击者知道 victim 的数据分布、标签集合和共现语义；现实中的图分布偏移可能显著降低计划质量。
- clique 只描述“是否共同出现”，没有利用对象尺度、相对位置、背景和关系谓词，语义模型较粗。
- PSPM 必须按数据集、surrogate、攻击算法和每个 `ε` 重新统计，类别数增大后离线成本很高。
- 论文没有作者代码，表中若干实现决策只能依据文字复现，难以核对检测框匹配与计划搜索细节。

## 评分

- **安全研究价值：8.5/10**——把上下文防御的威胁模型推进到无反馈、多目标联合攻击。
- **实验覆盖：8/10**——攻击者/受害者组合和预算较全，但主文数据仍集中在 VOC。
- **YOLO-Agent 防御价值：8.5/10**——适合作为多目标一致性红队 Harness，不能直接等价为部署风险。
- **综合：8.3/10**——概念鲜明且证据有力，真正落地的关键不在 PGD，而在评估攻击者能否获得匹配的上下文与 PSPM。
