# STDLens: Model Hijacking-Resilient Federated Learning for Object Detection

> This note is a structured first-pass summary generated from the project catalog. Please refine it with paper-specific quantitative details through contribution.

## 论文信息 / Paper Information

- **Venue:** CVPR 2023
- **Paper:** [https://openaccess.thecvf.com/content/CVPR2023/html/Chow_STDLens_Model_Hijacking-Resilient_Federated_Learning_for_Object_Detection_CVPR_2023_paper.html](https://openaccess.thecvf.com/content/CVPR2023/html/Chow_STDLens_Model_Hijacking-Resilient_Federated_Learning_for_Object_Detection_CVPR_2023_paper.html)
- **Code:** Not recorded in the catalog.
- **Institution:** See paper
- **Category:** General Object Detection
- **Detector family:** `other`
- **Applicability:** `recipe_idea_only`

## 摘要 / Abstract

CVPR 2023 paper on STDLens: Model Hijacking-Resilient Federated Learning for Object Detection.

## 背景与问题 / Background and Problem

This work is cataloged under **General Object Detection** for the `other` detector family. Its recorded task scope is:

- `object_detection`

The catalog does not currently store a full verbatim abstract or a complete problem statement for every imported paper. The official paper link above is the source of truth for the detailed motivation, assumptions, and benchmark protocol.

## 方法概览 / Method Overview

The catalog identifies the following components for follow-up reading:

- `object_detection`

For a faithful implementation, first establish the paper's original baseline, then isolate these components one at a time. Do not attribute the full paper gain to a single component without reproducing the ablations.

## 实验与证据 / Experiments and Evidence

- **Reported evidence:** See the official paper for datasets, metrics, baselines, ablations, and statistical significance.
- **Reproduction status:** No independent reproduction result is recorded in this catalog.
- **Code availability:** No official code link is recorded.

## 方法亮点 / Strengths

- The paper is indexed as a `recipe_idea_only` for detector research and harness design.
- The method can be compared against the project's existing detector-family and task-family groupings.
- The note keeps the paper link, code link, and structured implementation hints together for later contribution.

## 局限与风险 / Limitations and Risks

- This first-pass note does not claim paper-specific numbers beyond the official source.
- Imported records may not contain complete authorship, affiliation, dataset, or ablation metadata.
- Reproduction should preserve the paper's data split, training budget, augmentation, and evaluation code before comparing results.

## YOLO-Agent Harness Hints / 对 YOLO-Agent 的启发

- Reproduce the official baseline and isolate the claimed component before integrating it.

When testing this paper in YOLO-Agent, log the baseline, changed components, training budget, latency, AP metrics, and failure slices separately. Treat the paper as a hypothesis until the local reproduction confirms the claimed effect.

## 贡献清单 / Contribution Checklist

- [ ] Add the authors and affiliations.
- [ ] Replace the catalog summary with a paper-specific abstract summary.
- [ ] Record datasets, metrics, baselines, and key ablations.
- [ ] Add reproduction results and hardware/training details.
- [ ] Update limitations after independent verification.
