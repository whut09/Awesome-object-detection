---
title: "ForeSight: Multi-View Streaming Joint Object Detection and Trajectory Forecasting"
description: "详解 ForeSight 的联合流式记忆、预测感知检测与无跟踪轨迹预测。"
tags: ["ICCV 2025", "3D 目标检测", "轨迹预测", "多视角", "自动驾驶"]
---

# ForeSight: Multi-View Streaming Joint Object Detection and Trajectory Forecasting

**论文**：[官方论文页面](https://openaccess.thecvf.com/content/ICCV2025/html/Papais_ForeSight_Multi-View_Streaming_Joint_Object_Detection_and_Trajectory_Forecasting_ICCV_2025_paper.html)  
**代码**：[官方项目代码仓库](https://github.com/sandropapais/ForeSight)（论文项目页标注代码将发布）

## 一句话总结

ForeSight 用 Joint Streaming Memory Queue 连接 Forecast-Aware Detection Transformer 与 Joint Streaming Forecast Transformer：历史检测和多假设预测 query 共同初始化当前 3D 检测，当前检测再生成未来轨迹 query，预测结果反向进入下一帧检测，从而在不显式跟踪关联的情况下联合提升 nuScenes 多视角检测与轨迹预测。

## 研究背景与问题

传统自动驾驶把检测、跟踪、预测串行连接，预测只能消费上游结果，不能反哺被遮挡目标的检测；错误 track association 又会污染轨迹。StreamPETR 类稀疏 query detector 只传播过去检测，忽略多模态未来运动；多数预测器还在每帧重新计算轨迹。论文希望让过去检测、过去预测、当前图像和可选 HD map 在同一流式 query 空间交互。

## 方法总览

多相机图像经 backbone 与 3D positional embedding 形成图像 token `FI`；HD map 被表示为 lane graph，经 GRU 和 GAT 编码为 `FM`。检测侧把 `N` 个随机 3D anchor query 与 memory 中 `K` 个 temporal query 拼接，经 self-attention、历史 query cross-attention、图像 cross-attention 输出 3D 框。每个框与 `M` 个轨迹 anchor、`Tf` 个未来步组合成 forecast query，依次关注自身、历史预测、邻近 map token 和检测 query，解码未来 waypoint，再把 top-K 结果写回 FIFO memory。

## 方法详解

Joint memory `qmem` 保存 top-K 对象在 `Th` 个历史帧和 `Tf` 个未来步的检测—预测 query，并用 ego pose 变换把 3D 位置更新到当前坐标系。与只取上一帧 top-K 不同，ForeSight 可从多帧预测到当前时刻的候选中初始化 query，因此短时遮挡后仍能保留对象假设。

Forecast anchors 一部分由真值轨迹 k-means 获得，另一部分直接复用上一时刻最高置信轨迹，提高时间稳定性。预测器不依赖真实 track history，而从当前检测锚点启动；训练使用 focal classification、L1 box、ADE、FDE 和轨迹模式分类损失，并做检测与预测 query denoising。

## 实验与证据

nuScenes 含 1,000 个场景；作者由 track identity 构建 12 帧、6 秒预测标签。配置使用 `N=300` 检测 query、`K=128` temporal query、`Th=4`、`Tf=12`、`M=6`，总计 30,816 个 forecast query；检测/预测 transformer 分别 6/3 层，嵌入 256 维，训练 20 epochs。评估 mAP、NDS、mATE 等检测指标，以及 minADE、minFDE、MR、EPA。

ResNet50 下 ForeSight 为 0.466 mAP、0.560 NDS、23.5 FPS，比同帧同 query 的 StreamPETR 0.445 mAP 高 2.1 点。端到端表中 R50 版本为 0.499 EPA、0.466 mAP、0.709 minADE；R101 版本达到 0.549 EPA、0.502 mAP、0.689 minADE，比 UniAD 的 EPA 0.456 高 9.3 点。ViT-L 版本 mAP 0.543。

关键消融从单帧 PETR 0.361 mAP 开始：tracking-based forecast 的 minADE 为 2.83，detection-based forecast 为 1.05，改善 1.78 m；四帧直接检测 query 传播达 0.440 mAP；加入 forecast-based 反向传播后为 0.463 mAP、0.735 minADE；再加 HD map 为 0.466、0.709，地图使 minADE 降 0.026。

图像编码遵循 PETR 思路，将每个相机 backbone 特征与 3D positional embedding 融合成 `FI`；地图则把车道折线构成有向图，先用 GRU 编码连接序列，再用 GAT 聚合邻域，形成 `FM`。检测不直接读取地图，只有预测 query 关注最近的 16 个 map node；地图通过更准确的 forecast memory 间接改善下一帧检测，因此其收益较小但数据流清晰。

检测 query 包含 `N=300` 个随机 3D anchor 和 `K=128` 个 temporal query。二者先 self-attention，再关注四帧历史 memory 与当前多视角图像，解码 428 个九维框信息。预测侧把每个检测锚与 6 个模式、12 个未来时刻组合，形成 30,816 个 query。只有 top-K 高置信检测—预测对进入 FIFO，且位置通过前后 ego pose 矩阵转换到当前参考系。

无跟踪预测并非完全没有关联，而是把关联隐含在持续 query 和记忆选择中。tracking-based baseline 使用真实轨迹历史训练，面对检测产生的噪声 track 时 minADE 2.83；从当前 detection anchor 直接预测降到 1.05。该结果说明监督输入与部署输入不一致比显式 ID 本身更危险，但 memory 中错误高置信 query 仍可能长期传播，需要设置年龄和置信衰减。

检测表还报告定位误差和速度。ResNet50 ForeSight 的 mATE 0.614，略好于 StreamPETR 0.640；mAVE 0.370 明显好于 0.443，符合预测反馈增强运动建模的预期，FPS 则从 31.7 降至 23.5。R101、V2-99、ViT-L 的 mAP 分别为 0.502、0.489、0.543，均超过对应 StreamPETR，说明双向传播不只在单一骨干有效。

多任务权重设置为 `λdet=1`、`λforecast=2`，AdamW 训练 20 epoch，batch size 16，base learning rate `4e-4`，另有 cosine decay、500 次 warmup 和 `1e-2` weight decay。预测任务权重更高并未牺牲检测，反而通过反向传播提高 mAP。迁移到不同数据时应扫描任务权重，并监测 forecast loss 是否让类别置信度校准变差。

EPA 同时受检测 false positive、匹配命中和轨迹误差影响，比只在真值 agent 上计算 minADE 更接近端到端部署。作者用当前检测与真值框中心距离不超过 1 米做关联，MR 阈值为 2 米，并只取六种预测模式中最接近真值的一条。不同论文若使用不同感知范围、类别数、预测时长，EPA 和 minADE 仍不可机械横比，表 2 因此专门标注 3 秒/4 秒等变体。

## 对 YOLO-Agent 的启发

YOLO-Agent 若处理视频或多相机 3D，不应把轨迹模块仅当后处理。可把未来位置假设压成 memory token，作为下一帧 query 初始化和遮挡恢复先验；同时用 detection-based forecasting 避免硬跟踪 ID。需要严格限制 top-K、历史长度和模式数，否则 forecast query 数量会迅速膨胀。

### 论文专属 Harness

- **对照组**：单帧 detector、四帧直接检测 query 传播、tracking-based forecast、detection-based forecast、forecast 反向传播、完整模型+HD map。
- **观测指标**：mAP/NDS、遮挡目标 Recall、minADE/minFDE/MR/EPA、ID 关联错误敏感度、FPS、memory 占用与 forecast query 数。
- **通过阈值**：预测反哺相对直接传播提升至少 1.5 mAP，detection-based forecast 相对 tracking-based 降低 minADE 至少 0.5 m，FPS 保持基线 70% 以上。
- **失败判断**：遮挡恢复未提升、EPA 增益来自放宽匹配而非真实命中，或 memory 增长使 FPS 低于 15，则缩短 `Th/Tf` 或取消双向传播。

流式评估不能随机打乱帧。memory 必须在场景起始清空，按时间顺序更新 ego pose，并在掉帧时调整时间间隔；否则历史 query 会泄漏到其他场景或使用错误运动补偿。首帧没有 temporal query，论文用额外 detection query 替代，故冷启动和稳定阶段应分别报告。长序列还要监测 top-K 类别分布，避免静态车辆长期占满 memory 排挤新行人。

项目页同时提供可视化，代码仓库入口由作者页面指向；复现前应确认发布状态与配置是否覆盖论文表格版本。

## 优点

- 检测与预测形成真正双向数据流，消除显式跟踪瓶颈。
- 多 backbone、检测与端到端预测指标以及逐组件消融均较完整。

## 局限

- 30,816 个 forecast query 带来较高内存与实现复杂度。
- nuScenes 单基准且部分比较配置不同，EPA 横向公平性仍受感知范围和预测时长影响。

## 评分

- **问题重要性**：★★★★★
- **方法清晰度**：★★★★☆
- **实验证据**：★★★★★
- **工程可迁移性**：★★★☆☆
- **YOLO-Agent 参考价值**：★★★★★
