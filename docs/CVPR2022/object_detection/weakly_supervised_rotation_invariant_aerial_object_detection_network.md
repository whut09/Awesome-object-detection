---

# Weakly Supervised Rotation-Invariant Aerial Object Detection Network
title: "Weakly Supervised Rotation-Invariant Aerial Object Detection Network"
description: "RINet 论文精读：通过旋转一致监督与多实例挖掘，缓解航拍图像弱监督目标检测中的旋转敏感和实例遗漏问题。"
tags:
  - 弱监督目标检测
  - 航拍图像
  - 旋转不变学习
  - 多实例学习
  - CVPR2022
---

## 一句话总结

RINet 以 OICR 为基础，同时输入原图 \(I\) 与经 \(T_{rotate}\) 变换后的图像 \(I^{rotate}\)。共享 WSOD 网络和 ROI Pooling 分别产生候选区域特征 \(F_H\)、\(F_{H^{rotate}}\)，再拼接为 \(F_{H^J}=Cat(F_H,F_{H^{rotate}})\)。三组特征依次进入 **detection branch、rotated detection branch、rotation-invariant detection branch**，输出尺寸分别为 \(|H|\times(C+1)\)、\(|H|\times(C+1)\) 和 \(2|H|\times(C+1)\) 的分类概率。原分支选出的伪实例标签按同一仿射关系传播给旋转分支，旋转分支也反向监督原分支；两侧预测经 branch-wise average pooling 耦合，形成旋转一致伪标签并监督联合分支。

为避免 OICR 只选择最高分辨别区域、遗漏同类目标，网络加入 **Multiple Instance Mining（MIM）**。它先按 proposal score 做 K-means，从最高分簇选取 top-ranking proposals；随后分别在原图和旋转图空间构造基于 IoU 的无向无权图 \(G_c^o\) 与 \(G_c^r\)，将二者投影到交互空间并构造 \(G_c^I\)。旋转产生的隐式对应关系作为跨图边，标签在交互图上传播后再投影回原空间，用于监督 rotation-invariant branch，从而联合发现多个方向不同但类别相同的实例。

实验覆盖 NWPU VHR-10.v2 与 DIOR，以 IoU≥0.5 下的 AP/mAP 和 CorLoc 为指标，并与 WSDDN、OICR、PCL、DCL、PCIR、TCANet 等弱监督基线比较。DIOR 消融中，重新配置的 OICR 基线为 **18.7% mAP / 43.3% CorLoc**；仅加入旋转不变学习达到 **26.6% / 48.8%**，即提升 7.9 和 5.5 个百分点；加入普通 proposal clustering 后为 27.1% / 51.4%，最终 MIM 达到 **28.3% / 52.8%**。

- 论文：https://openaccess.thecvf.com/content/CVPR2022/html/Feng_Weakly_Supervised_Rotation-Invariant_Aerial_Object_Detection_Network_CVPR_2022_paper.html
- 官方代码：https://github.com/XiaoxFeng/RINet

## 研究背景与问题

航拍目标的任意朝向会造成同一实例在旋转前后的特征和检测结果不一致。普通 CNN 没有显式旋转约束，弱监督训练又只有图像级类别标签，无法像全监督检测那样同步旋转边界框标注，模型因此容易依赖飞机机翼、球场局部纹理等相对显著的区域，而非学习完整目标。

第二个问题是实例遗漏：一张航拍图中常同时存在大量同类目标，但 OICR 式逐级精炼主要围绕最高分 proposal 及其高 IoU 邻域传播标签，较低分的真实实例可能被当成背景。本文的核心判断是：旋转前后不仅类别不变，实例对应关系也能成为无需额外标注的监督信号；不同旋转感知分支还能互补暴露被单一分支忽略的实例。

## 方法总览

基础检测部分沿用 WSDDN/OICR。WSDDN 的分类流沿类别维做 softmax，检测流沿 proposal 维做 softmax，两者逐元素相乘得到区域分数，再求和形成图像级分类分数，并以多标签交叉熵训练。OICR 增加多级 instance refinement branch，将前一级最高置信区域及其高重叠邻域作为后一级伪标签。

旋转学习包含两层约束。第一层把原图第 \(r\) 个 proposal 的伪标签直接传递给其旋转对应 proposal，以 \(L_{rotate}\) 优化 rotated detection branch。第二层联合原图和旋转图的对应预测，生成长度为 \(2|H|\) 的伪标签集合，以 \(L_{RI}\) 监督拼接特征。该流程逐级执行，使伪标签从预测实例传播到旋转实例及邻近区域，目标是让相同实例在不同方向下保持分类一致。

## 方法详解

NWPU VHR-10.v2 图像大小为 400×400，含 10 类目标；RINet 在测试集获得 **70.4% mAP**。对应的 WSDDN、OICR、PCL、DCL、PCIR、TCANet 分别为 35.1%、34.5%、39.4%、52.1%、55.0%、58.8%，RINet 比最强对手 TCANet 高 11.6 个百分点，并在飞机、船、储油罐、网球场、田径场等类别上取得明显优势。

DIOR 包含 23,463 张 800×800 图像、192,472 个实例和 20 个类别。RINet 获得 **28.3% mAP、52.8% CorLoc**；TCANet 为 25.8% mAP、49.4% CorLoc，PCIR 为 24.9% 和 48.4%。训练仅使用图像级标签，trainval 用于训练，testing 用于检测评估。

## 实验与证据

实现采用 ImageNet 预训练 VGG16，Selective Search 每图生成约 2,000 个 proposals；旋转增强为 90°、180°、270°。优化器为 SGD，初始学习率 0.001，batch size 2，weight decay 0.005，momentum 0.9；NWPU 与 DIOR 分别训练 20K、200K 次，并在 10K、100K 时将学习率缩小十倍。OICR 自适应权重设为 0.1，NMS 阈值为 0.3。

消融设计较清楚，但未分别报告 90°、180°、270° 的独立贡献，也未量化旋转前后框位置与类别分数的一致性。MIM 依赖 Selective Search、K-means 和 proposal IoU 图，若候选框阶段漏掉小目标，后续跨旋转标签传播无法恢复这些实例。

## 对 YOLO-Agent 的启发

若将思想迁移到 YOLO-Agent，可保留单阶段检测器主体，训练时为同一图像生成原始视图和固定角度旋转视图，并记录框坐标的精确仿射映射。Agent 应建立三组对照：仅 YOLO 图像级伪标签基线、基线加旋转一致损失、旋转一致损失加跨视图多实例挖掘；禁止同时改变 backbone、输入尺寸和数据增强，以隔离模块贡献。

评估除 mAP@0.5、CorLoc 外，还应增加旋转一致率：把旋转视图预测逆变换回原坐标，统计匹配框的类别一致率及 IoU。具体失败判据为：最终方案相对“仅旋转一致损失”在 DIOR 风格多实例数据上的 mAP@0.5 提升不足 **1.2 个百分点**，或逆变换后同实例框的平均 IoU 低于 0.5，或小目标召回率下降超过 2 个百分点；任一成立即说明图传播或伪标签扩张没有复现本文收益。

## 优点

主要贡献不是把旋转当作普通数据增强，而是把旋转前后的实例对应关系转化为弱监督约束。三分支结构将“原视图预测—旋转视图预测—联合一致预测”组织成可端到端优化的数据流，同时让两个视图互相提供伪标签。

MIM 则把跨视图一致性用于实例扩张：它不只要求一个目标旋转前后预测相同，还利用两侧候选图的互补性搜索更多同类实例。因此，旋转不变学习负责稳定表征，MIM 负责提高实例覆盖率，两者存在相互促进关系。

## 局限

方法仍会错误利用场景共现关系。例如仅有 bridge 图像级标签时，模型可能定位更显著且经常与桥共现的河流。论文也指出其对小目标和场景歧义目标处理困难。由于监督来自模型自身预测，错误伪标签可能在原图、旋转图和交互图之间被双向传播并进一步强化。

此外，结果主要基于水平框与固定直角旋转，尚不能证明对任意连续角度、密集重叠目标或旋转框检测同样有效。与全监督 Faster R-CNN 在 DIOR 上的 55.5% mAP 相比，28.3% 仍有显著差距。

## 评分

RINet 给出的关键启示是：弱监督条件下缺失的实例标注，可部分由可逆几何变换产生的确定性对应关系替代。有效性来自两个必须同时满足的条件：跨视图 proposal 对应足够可靠，以及两个分支的错误不完全相同、能够形成互补监督。

最值得复现的是 DIOR 四级控制链：18.7/43.3 的基线、26.6/48.8 的旋转不变版本、27.1/51.4 的聚类版本、28.3/52.8 的完整 MIM。若实验无法沿这条链呈现稳定递增，应优先检查旋转 proposal 索引映射、联合特征拼接顺序、伪标签传播方向及图中 IoU 邻接关系，而非直接调节检测阈值。
