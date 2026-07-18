# YOLO-World: Real-Time Open-Vocabulary Object Detection

> This note is a structured first-pass summary generated from the project catalog. Please refine it with paper-specific quantitative details through contribution.

## 论文信息 / Paper Information

- **Venue:** CVPR 2024
- **Paper:** [https://arxiv.org/abs/2401.17270](https://arxiv.org/abs/2401.17270)
- **Code:** [official code](https://github.com/AILab-CVC/YOLO-World)
- **Institution:** Tencent AI Lab / SCUT
- **Category:** Open-Vocabulary and Grounded Detection
- **Detector family:** `yolo`
- **Applicability:** `direct_adapter_candidate`

## 摘要 / Abstract

Adds prompt-then-detect open-vocabulary capability to a real-time YOLO framework.

## 背景与问题 / Background and Problem

This work is cataloged under **Open-Vocabulary and Grounded Detection** for the `yolo` detector family. Its recorded task scope is:

- `object_detection`
- `real_time_detection`
- `open_vocabulary_detection`

The catalog does not currently store a full verbatim abstract or a complete problem statement for every imported paper. The official paper link above is the source of truth for the detailed motivation, assumptions, and benchmark protocol.

## 方法概览 / Method Overview

The catalog identifies the following components for follow-up reading:

- `text_encoder`
- `repvl_pan`
- `prompt_then_detect`

For a faithful implementation, first establish the paper's original baseline, then isolate these components one at a time. Do not attribute the full paper gain to a single component without reproducing the ablations.

## 实验与证据 / Experiments and Evidence

- **Reported evidence:** See the official paper for datasets, metrics, baselines, ablations, and statistical significance.
- **Reproduction status:** No independent reproduction result is recorded in this catalog.
- **Code availability:** https://github.com/AILab-CVC/YOLO-World

## 方法亮点 / Strengths

- The paper is indexed as a `direct_adapter_candidate` for detector research and harness design.
- The method can be compared against the project's existing detector-family and task-family groupings.
- The note keeps the paper link, code link, and structured implementation hints together for later contribution.

## 局限与风险 / Limitations and Risks

- This first-pass note does not claim paper-specific numbers beyond the official source.
- Imported records may not contain complete authorship, affiliation, dataset, or ablation metadata.
- Reproduction should preserve the paper's data split, training budget, augmentation, and evaluation code before comparing results.

## YOLO-Agent Harness Hints / 对 YOLO-Agent 的启发

- Cache text embeddings for fixed vocabularies and measure novel-class AP separately.

When testing this paper in YOLO-Agent, log the baseline, changed components, training budget, latency, AP metrics, and failure slices separately. Treat the paper as a hypothesis until the local reproduction confirms the claimed effect.

## 贡献清单 / Contribution Checklist

- [ ] Add the authors and affiliations.
- [ ] Replace the catalog summary with a paper-specific abstract summary.
- [ ] Record datasets, metrics, baselines, and key ablations.
- [ ] Add reproduction results and hardware/training details.
- [ ] Update limitations after independent verification.
