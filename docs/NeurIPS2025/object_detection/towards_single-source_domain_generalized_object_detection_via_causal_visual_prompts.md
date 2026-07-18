---
title: "Towards Single-Source Domain Generalized Object Detection via Causal Visual Prompts"
description: "Cauvis 以 Cross-Attention Prompts 和 Fourier 双分支适配器抑制颜色、背景等伪相关，使冻结 DINOv2 的检测器在未见天气域中学习更稳定的结构特征。"
tags: ["NeurIPS 2025", "目标检测", "域泛化", "Cauvis", "Visual Prompt", "DINOv2"]
---

# Towards Single-Source Domain Generalized Object Detection via Causal Visual Prompts

**会议**：NeurIPS 2025  
**论文**：[NeurIPS 官方页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9718ad22f53ea3566cdd5011844728dc-Abstract-Conference.html)  
**官方代码**：[lichen1015/Cauvis](https://github.com/lichen1015/Cauvis)  
**任务**：单源域泛化目标检测（SDGOD）

## 一句话总结

Cauvis 将可学习视觉提示视作独立于图像域的伪模态信号，通过 **Cross-Attention Prompts（CAP）** 调节冻结 DINOv2 特征，再用包含因果 MLP 分支和 Fourier 高通辅助分支的 **Dual-Branch Adapter**，削弱颜色、天气和背景模式造成的伪相关。

## 研究背景与问题

单源域泛化要求只用 Day-Clear 之类的一个源域训练，却在雾天、雨夜、黄昏等未见域工作。作者用 bus/truck 构造颜色偏置实验：把 truck 为白色的概率从 0.75 增到 0.9 后，偏置测试集上的 truck 精度上升，但无偏测试集 mAP 下跌 21%，说明检测器会把“白色”当成类别捷径，而非依赖轮廓和结构。

普通 visual prompt 能少量改善冻结视觉基础模型，却仍由单一源域监督，可能把提示本身也训练成域相关向量。论文表 1 显示，直接加 prompts 在 SDGOD 的 Night-Clear 只增 2.6 点，而跨域 1-shot 条件可增 4.1 点，暴露了零目标域知识时提示覆盖不足。Cauvis 因此让 prompt 与图像特征通过 cross-attention 交互，并额外从频域分离高频结构与颜色、背景干扰。

## 方法总览

图像进入冻结的 DINOv2-L，若干可学习 prompt 作为伪模态查询与视觉 token 执行 CAP。CAP 输出的 prompt activation 进入 **Causal Branch**，经 MLP 和 sigmoid 映射为因果门控特征；视觉特征同时进入 **Auxiliary Branch**，先由瓶颈 `Wdown/Wup` 压缩恢复，再执行 FFT、高通 mask 和 IFFT，提取轮廓等高频表示。两分支结果以可学习比例和预训练特征融合，送入 DINO 或 Faster R-CNN 检测头。

## 方法详解

论文用注意力矩阵 `A` 的 SVD 解释 CAP：大奇异值方向被假定对应低维因果子空间，小奇异值方向承载易受域变化影响的噪声。训练 prompt 时，cross-attention 期望强化前 `k` 个方向并压低其余奇异值，使提示覆盖潜在混杂因素后，视觉预测近似执行 back-door adjustment。这个论证依赖“因果特征形成低秩主子空间”的假设，但清楚说明了作者为什么选择 cross-attention，而不是简单逐元素相加。

**Causal Branch** 对 `CAP(pi,X)` 使用轻量 MLP 与 sigmoid，保留局部空间和语义响应。**Auxiliary Branch** 的 bottleneck 比例为 `r=D/16`，在频域施加高通 mask；作者把颜色偏移和背景纹理视作域特定干扰，而相对稳定的边缘、形状与几何轮廓被用于辅助主分支。DINOv2 参数全冻结，因此学习集中在 prompts、adapter 和检测相关投影上，避免单源数据彻底改写基础模型表示。

消融还比较 multi-head、SE Block、gate unit、3×3 conv 等更复杂的因果分支，均未超过简单 MLP 版本。Prompt 长度约 100 时已接近最佳：layer-wise/shared 分别约 59.7/59.8 mAP；使用约 1600 个、与视觉序列对齐的 prompt 虽达到最高约 59.9，但相对 100 token 只增不超过 0.2，训练显存和稳定性明显变差，因此作者推荐 100 作为默认长度。

## 实验与证据

SDGOD 包含 Day-Clear（DC，源域）、Day-Foggy（DF）、Dusk-Rainy（DR）、Night-Rainy（NR）、Night-Clear（NC），七类目标被划为 heavy、mid、non-motorized。模型训练 12 epoch，AdamW，DINO batch 16、Faster R-CNN batch 64，硬件为 8 张 RTX 4090；另在 Cityscapes-C 和 BDD100K-C 的 15 种 level-5 corruption 上测 mPC，所有扰动均未用于训练。

目标域总 mAP 上，Cauvis 在 DF、DR、NR、NC 分别达到 **56.5、64.6、47.6、61.2**；对应 UFR 为 39.6、33.2、19.2、40.8。仅 FR+DINOv2 已有 50.6/59.0/43.4/55.4，Cauvis 仍继续提升，说明结果不只是换用更强 backbone。源域 DC 达 73.7，附录还报告去掉整个 Cauvis prompt 模块、只保留冻结骨干时目标域明显下降。

Cityscapes-C 上 Faster R-CNN 为 15.4 mPC，OA-DG 为 21.8，FR+DINOv2 为 29.3，Cauvis 达 **35.6**；其中 Snow 相对 OA-DG 提升 23.7 点。BDD100K-C 从 FR+DINOv2 的 41.3 提到 43.6，增益覆盖噪声、模糊、天气和数字扰动，而非单一 corruption。

完整消融平均 mAP 为 60.7；去掉 Dual-Branch Adapter 降至 59.8，去掉 CAP 降至 57.4，去掉 visual prompts 为 54.3，去掉 DINOv2 后仅 34.2。辅助分支去掉 FFT 后平均值从 60.7 降为 59.8，去掉 mask 为 60.5。将 CAP 换成逐元素加法带来约 2.4 点损失，支持交互式提示而非静态偏置的设计。

不同视觉基础模型上的参数高效微调比较进一步约束了结论：在 EVA02-L、SAM-H、DINOv2-L 三种骨干上，Cauvis 的平均 mAP 分别为 52.0、53.8、60.7，均超过对应的 VPT-Deep、EVP、AdaptFormer、SPT-Deep 和 Rein。以 DINOv2-L 为例，Rein 为 59.0、SPT-Deep 为 59.7，而 Cauvis 为 60.7。收益不是简单来自增加可训练 token，但不同骨干的绝对差距也说明基础表示质量仍决定可达到的上限。

类别级结果显示恶劣域中的提升并非只来自车辆：Dusk-Rainy 总分达到 64.6，Night-Rainy 达 47.6，行人、骑手和非机动车分组也同步改善。因此 Fourier 分支更像跨类别的结构稳健化，而不是只修复白色 bus/truck 的专用规则。

## 对 YOLO-Agent 的启发

YOLO-Agent 在做单域部署优化时，可先主动构造颜色—类别、位置—类别偏置测试，判断模型是否使用捷径，再决定是否引入 prompt/adapter。频域分支不应被泛化成固定“高频一定更好”；代理需按具体域偏移验证不同频带与类别的关系，并监控源域精度，避免只为极端天气牺牲正常场景。还可把 prompt 长度设成资源约束变量，因为论文显示极长序列的边际收益很小。

### Harness

对照组使用冻结的同一视觉骨干加线性检测适配器；实验组依次为 visual prompt、CAP、CAP+causal branch、完整 Cauvis。训练只看源域，测试必须包含至少四个未见天气域和一套合成 corruption。观测源域 mAP、各目标域 mAP、15 类 corruption 的 mPC、颜色置换一致性、轮廓遮挡敏感度及可训练参数量。通过阈值为目标域平均 mAP 提升≥2.0、最差域提升≥1.0，且源域下降≤0.5；若收益只集中在单一天气、颜色置换后预测仍大幅翻转，或参数增加超过 10% 却不足 1 点提升，则判为未真正削弱伪相关。

## 优点

- 先用受控颜色偏置实验展示问题，再提出针对性结构。
- CAP、因果分支和 Fourier 分支均有独立消融证据。
- 冻结 DINOv2，参数高效且便于迁移到不同视觉基础模型。
- 同时覆盖真实天气域、Cityscapes-C 和 BDD100K-C 扰动测试。

## 局限

- “大奇异值等于因果特征”的理论假设缺少直接语义可视化验证。
- 性能的大部分基础来自 DINOv2，去掉它平均 mAP 下降 23.2 点。
- 训练和主实验成本较高，8 张 RTX 4090 不利于轻量复现。
- 高频结构并非所有域偏移都稳定，纹理型类别可能受到错误抑制。

## 评分

- **创新性**：8/10
- **实验充分度**：9/10
- **工程可迁移性**：7/10
- **对 YOLO-Agent 的价值**：8/10
