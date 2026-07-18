---

# Oriented RepPoints for Aerial Object Detection
title: "Oriented RepPoints for Aerial Object Detection"
description: "以自适应点集、方向转换函数、空间约束和 APAA 动态样本分配实现任意方向航空目标检测的论文笔记"
tags:
  - 航空目标检测
  - 旋转目标检测
  - RepPoints
  - Anchor-Free
  - CVPR2022
---

## 一句话总结

航空图像中的目标通常方向任意、分布密集且背景复杂，直接回归旋转框角度会受到角度周期性、边界不连续和旋转框定义不一致的影响。该方法不预测显式角度，而是把 RepPoints 的细粒度点集表示扩展到任意方向检测，让自适应点主动靠近目标的语义关键位置和几何边界；同时用 **MinAreaRect、NearestGTCorner、ConvexHull** 三种方向转换函数，将点集转换为旋转矩形或多边形。

网络采用 ResNet-FPN 的 P3–P7 特征，检测头保持 RepPoints 的初始化—细化两阶段结构：初始化阶段从特征图位置代表的中心点生成默认 9 个自适应点，细化阶段通过 DCN 使用点偏移采样特征，再输出分类结果和更准确的点集。训练时，点集先经 NearestGTCorner 或 ConvexHull 转成可微方向框，接受分类、GIoU 定位和空间约束监督；随后 **APAA（Adaptive Points Assessment and Assignment）** 根据分类、定位、方向对齐和逐点相关性综合评估候选点集，动态选择正样本。推理时移除 APAA，仅以 MinAreaRect 把最终点集转换为标准旋转矩形。

实验覆盖 DOTA、HRSC2016、UCAS-AOD 和 DIOR-R，并与 RetinaNet-O、S2A-Net、R3Det、RoI Transformer、ReDet、Oriented R-CNN、CFA 等方法比较。DOTA 单尺度测试中，ResNet-50、ResNet-101 和 Swin-T 分别取得 75.97%、76.52% 和 77.63% mAP；方向误差 mAOE 为 5.93，低于 Faster RCNN-O 的 6.01。转换函数消融尤其明确：RepPoints 的训练/推理 min-max 仅为 49.69%，推理改用 MinAreaRect 为 53.21%，训练使用 NearestGTCorner 或 ConvexHull 后分别升至 66.97% 和 68.89%。

- 论文链接：https://openaccess.thecvf.com/content/CVPR2022/html/Li_Oriented_RepPoints_for_Aerial_Object_Detection_CVPR_2022_paper.html
- 官方代码：https://github.com/LiWentomng/OrientedRepPoints

## 研究背景与问题

核心思想是以**点集的空间布局隐式表达方向**。相较于五参数旋转框，点集不需要处理角度首尾相接，也能描述飞机、舰船、桥梁等类别不同的长宽比、姿态和局部几何结构。

三种转换函数承担不同职责。MinAreaRect 寻找覆盖点集的最小面积旋转矩形，但不可微，因此只在后处理使用。NearestGTCorner 为真值框的每个角寻找最近预测点，以四个被选点构成四边形；ConvexHull 则通过 Jarvis March 从点集中构造凸包。后两者可传递定位梯度，其中 ConvexHull 的 DOTA mAP 最高。

总损失由分类损失和两个阶段的空间定位损失组成。分类采用 Focal Loss；定位以转换后的方向多边形计算 GIoU。初始化阶段权重为 0.3，细化阶段权重为 1.0，使网络先形成稳定点集，再重点优化最终几何位置。

## 方法总览

**空间约束**针对被邻近目标或强背景纹理吸引的离群点。若某个预测点落到所属真值旋转框之外，就用该点到真值框几何中心的距离作为惩罚；框内点惩罚为零。它不是简单收缩所有点，而是只纠正越界点，保留点集沿目标边缘展开的能力。

加入空间约束后，DOTA mAP 从 68.89% 升至 70.11%。受益明显的类别包括 Baseball Diamond 76.85→79.99、Bridge 41.72→45.33、Roundabout 67.09→71.39，以及特征较弱的 Helicopter 41.55→51.87，说明该约束主要缓解相似背景和邻接实例造成的归属错误。

## 方法详解

APAA 的质量函数为  
\(Q=Q_{cls}+1.0Q_{loc}+0.3Q_{ori}+0.1Q_{poc}\)。

其中，\(Q_{cls}\) 使用分类损失衡量点特征与类别的匹配程度；\(Q_{loc}\) 使用转换框与真值框之间的定位损失；\(Q_{ori}\) 先对预测框和真值框的四条边等距采样，各取默认 40 个轮廓点，再计算 Chamfer Distance；\(Q_{poc}\) 则提取默认 9 个点的逐点特征，通过各归一化特征与点集平均特征的余弦相似度衡量内部相关性和多样性。

初始化阶段仍使用 RepPoints 的中心点分配器。细化阶段按质量值排序每个目标对应的候选点集，并以 \(k=\sigma N_t\) 选择前 k 个正样本；采样率在 0.2、0.3、0.4、0.5 中以 0.4 最优，得到 75.97% mAP。该模块只参与训练，因此不会增加推理计算量。

## 实验与证据

质量项逐步加入时，mAP 从仅有基础分配的 70.11%，依次提高到 72.34%、74.46%、75.32%，最终四项齐备达到 75.97%，累计增益 5.86。样本分配横向比较中，Max-IoU、ATSS、PAA、CFA 和 APAA 分别为 70.11%、72.87%、74.62%、74.89% 和 75.97%。

与直接角度回归的同结构检测器比较，ResNet-50-FPN 下由 67.50% 提升至 68.89%，增益 1.39；ResNet-101-FPN 下由 68.73% 提升至 70.19%，增益 1.46。这表明收益并非单纯来自更强骨干，而来自点集方向表示本身。

跨数据集结果为：HRSC2016 上 VOC2007 mAP50 为 90.38、VOC2012 mAP50 为 97.26；UCAS-AOD 上汽车 AP 89.51、飞机 AP 90.70、总体 mAP 90.11；DIOR-R 使用 ResNet-50-FPN 获得 66.71% mAP，高于 AOPG 的 64.41%。

## 对 YOLO-Agent 的启发

若将该思路接入 YOLO-Agent，可保留 YOLO 的多尺度主干和分类分支，把水平框或显式角度回归头替换为两阶段 9 点偏移头。第一阶段从网格中心生成点集，第二阶段使用点偏移进行可变形采样；训练端用 ConvexHull 计算方向 GIoU，并加入越界点空间约束和 APAA，部署端仅执行 MinAreaRect。

APAA 适合作为训练期样本路由器，而非推理组件：Agent 可监控 \(Q_{cls}\)、\(Q_{loc}\)、\(Q_{ori}\)、\(Q_{poc}\) 的分布，定位“分类置信度高但方向框质量低”的样本，并调节正样本比例。对于密集舰船、车辆或旋转飞机，应优先检查点集是否跨越相邻实例，而不是只调整角度损失。

## 优点

方法依赖点集能覆盖目标的判别性区域；极小目标在低分辨率特征层上可供采样的位置有限，9 个点可能退化或重叠。NearestGTCorner 借助真值角点构造训练框，训练与推理转换函数并不完全一致；ConvexHull 还能产生非矩形多边形，最终再用 MinAreaRect 规整时可能损失局部几何信息。

空间约束只判断点是否越界，并把越界点拉向几何中心，没有显式约束点沿轮廓均匀分布。在严重遮挡、目标紧邻或中心区域缺少纹理时，点可能集中在少数强响应位置。

## 局限

复现 Harness 应固定 DOTA 的 ResNet-50-FPN、单尺度输入和相同训练日程，建立四组控制：原始 min-max RepPoints；仅推理加入 MinAreaRect；训练使用 ConvexHull 并加入空间约束；最后加入完整 APAA。主要指标记录 mAP、各类别 AP 和 mAOE，并单独观察 BD、BR、RA、HC 四类。

具体失败标准：若转换函数实验不能保持 49.69＜53.21＜66.97＜68.89 的排序，说明点集到方向框的数据流实现有误；若完整 APAA 不高于 CFA 的 74.89 mAP，或方向 mAOE 不低于 Faster RCNN-O 的 6.01，则不能证明样本评估与隐式方向表示复现成功。

## 评分

这项工作的关键贡献不是增加另一种角度编码，而是把“方向”转化为点集几何学习问题。可微方向转换提供定位梯度，空间约束处理越界点，APAA 解决无逐点真值监督时的样本质量选择；三者共同使 anchor-free 点检测器能够稳定处理任意朝向、密集分布和复杂背景中的航空目标。
