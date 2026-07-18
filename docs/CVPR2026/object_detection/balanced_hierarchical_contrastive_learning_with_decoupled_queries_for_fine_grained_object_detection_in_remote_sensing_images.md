---
title: "[论文解读] Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images"
description: "通过分类/定位查询解耦和带类原型均衡的层级对比损失，将遥感细粒度标签树嵌入 DETR 表征空间。"
tags: ["CVPR 2026", "细粒度检测", "遥感", "DETR", "对比学习"]
---

# Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images

**论文**：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Balanced_Hierarchical_Contrastive_Learning_with_Decoupled_Queries_for_Fine-grained_Object_CVPR_2026_paper.html)  
**代码**：[chncdx/BHCL](https://github.com/chncdx/BHCL)

## 一句话总结

该方法把 DETR 的对象查询拆成分类查询与定位查询，只让分类查询接受 Balanced Hierarchical Contrastive Loss（BHCL）；BHCL 用每个树节点的类原型补齐小批次缺失类别，并按类别平均负样本贡献，从而在长尾标签树上学习粗到细语义而不把同父类目标的框位置错误聚拢。

## 研究背景与问题

ShipRSImageNet 与 FAIR1M 等细粒度遥感数据以树状标签组织飞机、舰船等大量子类。普通监督对比学习只在叶节点拉近同类、推远异类，忽略“同属战舰但型号不同”的层级关系；直接在每层计算层级对比损失又会受到两种不平衡影响：叶类样本数长尾，中间节点的子类数量也不均。同时，DETR 的同一对象查询承担分类和框回归，若把共享父类的对象在表征空间强行拉近，可能连空间定位特征也被拉近，妨碍不同实例保持独立位置。

## 方法总览

图像经 Backbone 与 Transformer Encoder 形成 token。解码器维护一一对应的分类查询 `Qcls` 和定位查询 `Qloc`：每层先将两者拼接并做共享 Self-Attention，使同一候选的语义和几何对齐；再拆分成两个流，各自通过独立 Cross-Attention 与 FFN 从编码特征读取任务专属信息。分类分支输出类别，定位分支回归旋转框，匈牙利匹配仍由 focal classification、Rotate IoU 与 L1 代价完成。仅匹配到前景的分类查询经过投影与归一化，在每个解码层计算 BHCL。

## 方法详解

### Decoupled Learning of Object Queries

标准 DETR 的单查询共享分类与回归表示。本文先 `Concat(Qcls,Qloc)`，经共享自注意后再 `Split`，保证两类查询对应同一对象；随后 `Qcls` 与 `Qloc` 分别进入不同的交叉注意和 FFN。这样，分类查询可以按标签树聚类，定位查询仍以类无关方式保持实例空间差异。最终分类头只读取 `Qcls`，旋转框回归头只读取 `Qloc`，BHCL 也不会直接作用于定位流。

### Hierarchical Contrastive Learning

匹配后的分类查询投影到低维单位球面。对标签树的每一层，拥有同一祖先类别的查询构成正样本，不同节点构成负样本；各层损失加权求和，越靠近叶节点权重越高，以强调细粒度区分。论文还把 “Other Aircraft Carrier” 等不确定叶类重新分配给更一般的父节点，使中间节点本身也能接收训练实例，而非把所有 “Other*” 强行视作互斥细类。

### Prototype-Based Class Balancing

BHCL 建立覆盖除根节点外所有类别节点的原型库。即使某稀有类没有出现在当前 mini-batch，其原型也会作为额外实例参与对比。分母不再让高频负类凭实例数量占据更大梯度，而是先在每个类别内部平均，再对类别求和，令各层每个类的贡献更均衡。原型采用 EMA 更新；中间节点原型由该节点及全部后代匹配查询的均值更新。总损失为 BHCL、focal、Rotate IoU 与 L1 的加权和。

这里的“均衡”不是重采样图片，也不是给叶类分类损失简单乘逆频率，而是改变对比损失分母中负类的聚合单位：先按类平均，再让类与类等权。原型的第二个作用是让当前批次未出现的节点仍进入分母，所以小批量训练也能看到完整层级边界。若实现中只为叶节点建立原型，父节点语义和 “Other*” 回退机制都会丢失。

原型更新需避免背景查询污染。只有经匈牙利匹配确认的前景分类查询进入投影空间；背景仍由检测分类损失处理。中间节点原型聚合自身及后代，意味着标签树映射必须在数据预处理阶段固定并可审计。若不同数据集的树深不一致，还应检查层级权重是否过度偏向叶级，否则 BHCL 可能退化为普通细类对比。

## 实验与证据

实验在 ShipRSImageNet、FAIR1M-v1.0、FAIR1M-v2.0 上进行，Backbone 为 ResNet-50，输入统一为 1024×1024；基线包括 RHINO 与 OrientedFormer，比较对象还包括 ReDet、ORCNN、LSKNet、PCLDet、PETDet、SFRNet 和 DRNet。ShipRSImageNet 因测试标注与服务器不可用，在验证集报告旋转 COCO 指标；FAIR1M 使用官方服务器报告 AP50。

最终方法在 ShipRSImageNet 达到 64.3 AP50:95，超过 OrientedFormer 1.1；FAIR1M-v1.0 为 41.66 AP50，超过 OrientedFormer 0.35；FAIR1M-v2.0 为 47.53 AP50，超过 DRNet 0.49。与只在叶级做原型对比的 PCLDet 相比，ShipRSImageNet 高 2.7 AP50:95。

关键消融显示机制具有可迁移性。RHINO 从 59.78 AP50:95 加查询解耦后到 60.99，再加 HCL 到 61.24，换成 BHCL 到 61.41；OrientedFormer 从 63.17 到 63.60、64.12、64.32。FAIR1M-v1.0 上普通 HCL 从解耦后的 41.38 降到 41.14，而 BHCL 升至 41.66，直接说明长尾条件下“层级对比但不均衡”可能有害，类原型与类别均衡是关键。

## 对 YOLO-Agent 的启发

- 在 YOLO 解耦头中，可把分类塔的正样本特征投影后施加 BHCL，框回归塔不接收层级对比梯度；这比对共享 Neck 特征直接做对比更符合论文的冲突分析。
- **Harness 对照组**：原始 YOLO-OBB、共享特征 HCL、分类塔 HCL、分类塔 BHCL；再比较无原型、可学习梯度原型、EMA 原型，以及是否把 “Other*” 回退父类。
- **Harness 指标**：总体 AP50:95、头部/中部/尾部类 AP、层级错误代价、父类正确但叶类错误比例、定位 AP75、类原型覆盖率与每类梯度范数。
- **失败判断**：若分类塔 BHCL 提升叶类 AP 却显著降低 AP75，说明解耦仍不充分；若尾类 AP 未提升或高频类梯度仍主导，说明类别平均与原型补齐未实现；若共享 HCL 与解耦 BHCL无差异，则论文所述分类—定位冲突未在该 YOLO 配置中复现。

## 优点

- 将标签树、长尾均衡和检测任务冲突统一到清晰的查询级方案中。
- 在两个不同 DETR 基线上做组件消融，证明方法不依赖单一架构。
- 对 “Other*” 中间节点给出符合层级语义的处理方式，工程细节有价值。

## 局限

- 需要可靠的标签树；树结构错误会把错误先验写入对比空间。
- 类原型库、每层 BHCL 与双路解码器增加训练和实现复杂度，推理收益主要来自精度而非速度。
- 三个数据集都属于细粒度遥感，向通用检测或非树状多标签任务推广仍待验证。

## 评分

- **创新性**：★★★★☆
- **实验可信度**：★★★★★
- **YOLO-Agent 参考价值**：★★★★☆
