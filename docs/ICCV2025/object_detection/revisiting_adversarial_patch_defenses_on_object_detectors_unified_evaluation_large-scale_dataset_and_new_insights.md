---
title: "Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights"
description: "APDE 统一评测 13 种补丁攻击、11 个检测器和 11 种防御，并揭示被攻击对象 AP 比补丁掩码精度更能反映真实防御效果。"
tags: ["ICCV 2025", "目标检测", "对抗补丁", "APDE", "鲁棒性评估", "安全"]
---

# Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights

**论文**: [arXiv](https://arxiv.org/abs/2508.00649)  
**代码与数据**: [Gandolfczjh/APDE](https://github.com/Gandolfczjh/APDE)  
**任务**: 目标检测器对抗补丁防御的统一评估与数据集构建

## 一句话总结

该文不是再发明一种防御，而是用 APDE 把隐藏/显现目标、13 种 patch attack、11 个 detector、9 个经验防御和 2 个认证防御放进统一协议，并证明 person AP@0.5、ASR 与效率应优先于单纯 patch mask 精度。

## 研究背景与问题

既有补丁防御论文使用不同攻击参数、贴图位置、受害检测器和指标，横向排名并不可靠。有的方法只报告补丁检测 AP，即使准确找到 patch，也可能误删人体或关键背景，最终检测反而更差；自然式补丁还常被解释为“缺少高频异常”，但这一判断没有在多攻击分布下验证。作者因此把工作分为三层：建立标准数据和攻击协议，统一实际防御指标，再用大规模结果分析失败根因、物理世界和 adaptive attack。

## 方法总览

APDE 从 INRIA-Person 与 MS COCO 出发，使用 13 种攻击在 11 个主流检测器上白盒优化 universal patch，通过旋转、缩放等 applier 贴入图像，得到 94 种补丁、94,000 张 416×416 PNG；按 6:4 划分 56,400 训练和 37,600 测试。基准覆盖 SAC、PAD、Adyolo、NAPGuard、LGS、Zmask、Jedi、DIFFender、NutNet，以及 DetectorGuard、ObjectSeeker 两种 certified defense。评价同时报告 attacked-object AP、ASR、每图时间、SmIoU 和 NmIoU。

## 方法详解

SmIoU 先在全数据累计 mask 的 TP、FP、FN 再求 IoU，因此更受大面积像素影响；NmIoU 逐图求 IoU 后平均，强调样本级稳定性。两者能揭示防御是倾向只覆盖补丁核心，还是把大块正常背景误判为 patch。真正的防御效果用 person AP@0.5 与 ASR 表示，因为补丁定位正确并不保证擦除后仍能识别人。数字域之外，论文打印 30 类补丁，用 iPhone 16 Pro 在不同距离、光照和 -90° 到 90° 角度采集 540 帧；adaptive attack 则假设攻击者知道防御机制，检查梯度、随机性和通用 patch 先验能否被绕过。

数据分析还检验自然式补丁的失败原因。GNAP、DM-NAP 的高频直方图与其他补丁差异并不大，但它们与干净图像及其他攻击的 FID 分布更分散；作者据此认为困难主要来自训练数据分布覆盖不足，而非天然缺乏高频。用 APDE 重训 SAC、Adyolo、NAPGuard 后，再测试训练内攻击和未纳入训练的 AdvCloak、AdvTshirt，用以验证分布多样性能否带来外域防御。

APDE 的攻击目标同时覆盖 hiding 与 appearing，补丁由白盒目标函数和 total variation 正则优化，再按原攻击论文的几何变换贴入干净图像。94 种 patch 不仅包含方形、圆形和矩形，还含狗、卡通图案等自然外观；与 Apricot 的 1,011 张、GAP 的 9,266 张相比，94,000 张规模允许把同一防御的训练和评测分开。作者强调最坏情况来自补丁在受害 detector 上直接优化，因此同一防御必须跨 YOLO、SSD、CenterNet、RetinaNet、两阶段模型和 DDETR 检查，不能只在单一 YOLO 上得出安全结论。

## 实验与证据

数字隐藏攻击中，11 个 detector 的无防御平均 person AP 为 30.74。各防御总体 mean/min AP 中，NutNet 为 76.53/55.79，NAPGuard 为 75.94/46.93，PAD 为 76.12/40.99；但 PAD 每图约 32,100ms，NutNet 71ms，SAC 最快为 44ms。NAPGuard 的 mask 精度最高，却不是最终防御最强，支持“补丁检测精度不能替代被攻击对象 AP”的结论。

物理测试中距离增大使攻击减弱，AP 约从 60% 升到 85%；正面角度 [-30°,30°) 最难防，AP 比其他角度低 30–50%。adaptive attack 下 PAD 因复杂 SAM 语义模型相对稳健，DIFFender 受 diffusion 随机性保护，Zmask/Jedi 依赖更通用的过激活或高熵属性；SAC 等可微或特定纹理先验更易被绕过。三块 patch 的认证评测中，ObjectSeeker 衰减较小但枚举成本随 patch 数增长，经验方法 PAD/NutNet 仍常保持 80% 以上 AP。

APDE 重训带来显著收益：例如 SAC 对 AdvTshirt 从 34.27 提到 64.47，对 AdvCloak 从 4.17 提到 71.29；NAPGuard 对 AdvTshirt 从 50.21 到 70.89。论文汇总新数据可将防御提高 15.09% AP@0.5。补丁尺寸分析还发现大 patch 更容易被 mask 找到，却可能造成更差 person AP，进一步说明定位指标与任务鲁棒性并非同义。

针对具体失败 patch，T-SEA 未防御 AP 为 36.57，PAD、NAPGuard、NutNet 防御后分别为 73.55、79.21、68.03；而自然式 GNAP 未防御已为 73.48，SAC 几乎不变为 73.47，DIFFender 甚至降到 67.51。三块 T-SEA patch 下，YOLOv3 的 PAD/NAPGuard/NutNet AP 为 81.31/83.10/82.09，FRCNN 为 80.94/81.35/81.42；DetectorGuard 与 ObjectSeeker 在三 patch 处未给结果，体现认证范围限制。物理实验还发现光照增强时攻击效果下降，但 PAD 会把人体本身误作补丁，因此数字域高 AP 不保证真实拍摄中仍保持相同排名。

九种经验防御的类别也没有形成简单优劣顺序：分割式、检测式、先验式和生成式方法中都同时存在强弱方案。CenterNet 与 DDETR 在无防御攻击下 AP 仅 14.66 和 4.56，暴露不同检测架构的脆弱度差异；NutNet 在这两者上恢复到 71.04 和 66.56。论文因此建议报告跨 detector 的 mean 与 minimum，minimum 能防止一种方法靠少数容易受保护的模型抬高平均值。

补丁尺寸实验中，大 patch 的 SmIoU 往往更高，但 person AP 更低；PAD 还会把大块背景算入掩码，出现 SmIoU 与 NmIoU 的相反排序。这个现象说明同一个 mIoU 数字也必须结合聚合方式解释。安全报告若只给平均 mask IoU，既可能掩盖少数严重误删样本，也可能偏爱覆盖面积较大的方法，无法替代逐图失败审计。

## 对 YOLO-Agent 的启发

安全 Harness 必须围绕最终检测任务建立。对照组应包括无防御、简单黑块擦除、SAC、NAPGuard、NutNet，以及在 APDE 重训前后版本；受害模型至少覆盖 YOLOv5/YOLOv7、RetinaNet、Faster R-CNN 和 DDETR。观测指标同时记录 clean AP、attacked person AP、ASR、SmIoU/NmIoU、最差攻击 AP、物理角度分桶和每图延迟。通过阈值建议 attacked AP 相对无防御提升 20 点、clean AP 损失小于 1 点、最差攻击 AP 不低于均值 70%，实时方案低于 10ms 额外开销；若 mask IoU 上升但 person AP 不升，adaptive attack 使 AP 回落到无防御的 5 点以内，或把人体大面积抹除，应立即判为失败。

## 优点

- 覆盖攻击、检测器、防御、数字/物理和自适应威胁，评估维度完整。
- 明确区分补丁定位质量与目标检测恢复效果，纠正单指标排名问题。
- 94 种补丁和外域重训实验为“数据分布决定自然式补丁难度”提供直接证据。

## 局限

- 统一协议主要围绕 person detection，其他类别和自动驾驶多目标风险需另建阈值。
- 物理世界仅 540 帧、30 种补丁，环境、打印材质和摄像头覆盖仍有限。
- certified defense 的威胁模型与经验防御不同，统一表格不能等同于同条件安全保证。

## 评分

- **创新性**: ★★★★☆
- **实验充分度**: ★★★★★
- **部署价值**: ★★★★★
- **YOLO-Agent 参考价值**: ★★★★★
