---
title: "Toward Open Vocabulary Aerial Object Detection with CLIP-Activated Student-Teacher Learning"
description: "解析 CastDet 的定位教师、RemoteCLIP 外部教师、动态标签队列与混合监督如何发现航拍新类别。"
tags: [ECCV2024, open-vocabulary-detection, aerial-images, CLIP, student-teacher]
---

# Toward Open Vocabulary Aerial Object Detection with CLIP-Activated Student-Teacher Learning

## 论文与代码

- 论文：[ECCV 2024 官方页面](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11741_ECCV_2024_paper.php)
- 代码：[官方 CastDet 仓库](https://github.com/lizzy8587/CastDet)

## 一句话总结

CastDet 用 EMA localization teacher 从无标注航拍图持续提出类别无关框，再让冻结的 RemoteCLIP external teacher 为裁剪区域赋开放词汇标签，并把高可信伪标签存入 Dynamic Label Queue 反复训练 student，使模型不局限于人工标注中的 base classes。

## 研究背景与问题

开放词汇检测通常借助 CLIP 把区域与文本类别对齐，但航拍目标尺寸小、俯视外观与自然图像差异大，RPN 很难先把未标注的新类别框出来；若只在 base-class 框上训练 proposal 网络，训练越久越倾向忽略 novel objects。纯半监督 teacher-student 也只能为标注体系内类别生成伪标签，不能命名机场、风车等训练标注外目标。CastDet 将定位和命名拆给两个教师：一个从 student 的检测知识演化出候选框，另一个固定保留遥感视觉语言知识，避免 proposal 能力与开放词汇语义互相牵制。

## 方法总览

Student 是 Faster R-CNN 式两阶段检测器。Localization Teacher（LT）由 student 参数指数滑动平均得到，接收无标注图的弱增强视图并输出 class-agnostic proposals。External Teacher（ET）使用冻结的 RemoteCLIP：裁剪 LT 候选区域，与类别文本 embedding 计算相似度并产生类别分布。Box Selection 对 proposal foreground score、box jitter stability 与图文分类置信度联合过滤。Dynamic Label Queue（DLQ）保存跨 batch 的高质量框、类别和分数，使稀有 novel 类伪标签不会一次使用后消失。student 同时学习有标注真框和队列伪框，更新后再通过 EMA 改进 LT。

## 方法详解

LT 负责“哪里可能有物体”，不强迫其输出 base/novel 细分类别；ET 负责“这个裁剪像哪个文本类别”，不参与框回归。论文比较三种框筛选信号：RPN foreground confidence、Box Jitter Variance（BJV）和 Region Jitter Variance（RJV）。对候选框坐标做小扰动后，若 LT/ET 输出稳定，说明定位与语义都可靠；再与 CLIP classification score 联合，过滤背景纹理和边界漂移框。补充可视化显示，仅 student 的 RPN 新类别响应会随训练减弱，加入 LT 后仍只能发现标注类，而完整 CastDet 的 novel-category 前景响应持续增强。

DLQ 将当前 batch 产生的伪标签按类别与置信度入队，训练时采样历史样本，缓解航拍图中 novel 类稀少、单批分布剧烈波动。独有闭环是“无标注图弱增强→EMA LT proposals→框扰动稳定性过滤→RemoteCLIP 区域-文本分类→高可信伪标签入 DLQ→student 在强增强图上训练→EMA 更新 LT”。有标注数据走标准检测损失，无标注数据走伪框分类、回归与一致性损失，外部教师全程冻结，因此不会被错误伪标签反向污染。

## 实验与证据

作者构建 VisDroneZSD：base classes 为 aircraft、ship、vehicle、basketball court、tennis court、football field；novel classes 为 windmill、airport、basketball field、ground track field。训练只标注 70% 图像中的 base 类，其余 30% 图像不带标签。CastDet 达到 40.5 mAP，其中 base 39.0、novel 46.3、调和均值 42.3，优于 ViLD、Detic、BARON、OV-DQUO、CoDet 等自然图像开放词汇方法。将 RemoteCLIP 换成普通 CLIP 或改变提示词时性能下降，说明遥感域视觉语言教师是关键而非任意分类器。

消融分别验证 LT、ET、DLQ 与筛框策略。只有 student 或普通半监督 LT 时，新类别能力随训练收缩；加入 ET 后可命名 novel proposals，加入 DLQ 后稀有类别更稳定。BJV/RJV 与 CLIP score 的联合筛选优于只按 RPN score，定性图中机场、篮球场、田径场、风车的伪框更完整。论文还在开放词汇 COCO 上评估，表明框—文本教师分工不是 VisDroneZSD 特例，但其主要贡献仍是航拍新类检测。

## 对 YOLO-Agent 的启发

可让 YOLO-Agent 维护 YOLO student、EMA 定位教师（LT）和冻结 RemoteCLIP 外部教师（ET），再由 Dynamic Label Queue（DLQ）记录伪框迭代、稳定度与文本 margin。围绕 VisDroneZSD，closed-set YOLO、仅 LT、LT+普通 CLIP、LT+RemoteCLIP、加入 BJV/RJV jitter filter、最后接入 DLQ 是必须保留的**对照组**，并在开放词汇 COCO 上做域外复查。评估**指标**同时覆盖 base AP、novel AP、harmonic mean、unknown recall、伪标签 precision/recall、每类队列覆盖率；机场、风车、篮球场、田径场各人工审计 100 个伪框。出现 novel AP 上升却令 base AP 下降超过 3 点、错误标签在 DLQ 存活超过 3 个刷新周期，或低 RemoteCLIP margin 类别仍大量入队，均作为确认偏差的**失败判断**；若真框 oracle+ET 有效而 LT proposal recall 低于 60%，应先修定位教师，不应继续调提示词。

## 优点

- 把开放词汇检测的定位瓶颈与语义瓶颈交给不同教师处理。
- RemoteCLIP 针对遥感域，能覆盖人工标注之外的场景类别。
- DLQ 让罕见新类伪标签跨 batch 留存，适合长尾航拍数据。

## 局限

- 依赖预先给定候选词表，真正未知且未命名的类别仍无法正确分类。
- 队列会积累早期错误，EMA teacher 与 student 可能形成闭环确认偏差。
- 两阶段裁剪加 RemoteCLIP 分类训练成本高，端到端实时部署需另做蒸馏。

## 评分

- 创新性：9/10
- 实证充分性：8/10
- 工程可迁移性：7/10
- 对 YOLO-Agent 价值：9/10
