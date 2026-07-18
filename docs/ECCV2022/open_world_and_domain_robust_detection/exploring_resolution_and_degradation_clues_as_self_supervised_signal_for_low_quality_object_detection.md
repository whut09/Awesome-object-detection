# Exploring Resolution and Degradation Clues as Self-Supervised Signal for Low Quality Object Detection

> This note is a structured first-pass summary generated from the project catalog. Please refine it with paper-specific quantitative details through contribution.

## 论文信息 / Paper Information

- **Venue:** ECCV 2022
- **Paper:** [https://arxiv.org/abs/2208.03062](https://arxiv.org/abs/2208.03062)
- **Code:** [official code](https://github.com/Chokyeol/AERIS)
- **Institution:** See paper
- **Category:** Open-World and Domain-Robust Detection
- **Detector family:** `other`
- **Applicability:** `recipe_idea_only`

## 摘要 / Abstract

Uses resolution and degradation prediction as self-supervised signals for low-quality detection.

## 背景与问题 / Background and Problem

This work is cataloged under **Open-World and Domain-Robust Detection** for the `other` detector family. Its recorded task scope is:

- `object_detection`
- `low_quality_detection`

The catalog does not currently store a full verbatim abstract or a complete problem statement for every imported paper. The official paper link above is the source of truth for the detailed motivation, assumptions, and benchmark protocol.

## 方法概览 / Method Overview

The catalog identifies the following components for follow-up reading:

- `resolution_clue`
- `degradation_clue`
- `self_supervision`

For a faithful implementation, first establish the paper's original baseline, then isolate these components one at a time. Do not attribute the full paper gain to a single component without reproducing the ablations.

## 实验与证据 / Experiments and Evidence

- **Reported evidence:** See the official paper for datasets, metrics, baselines, ablations, and statistical significance.
- **Reproduction status:** No independent reproduction result is recorded in this catalog.
- **Code availability:** https://github.com/Chokyeol/AERIS

## 方法亮点 / Strengths

- The paper is indexed as a `recipe_idea_only` for detector research and harness design.
- The method can be compared against the project's existing detector-family and task-family groupings.
- The note keeps the paper link, code link, and structured implementation hints together for later contribution.

## 局限与风险 / Limitations and Risks

- This first-pass note does not claim paper-specific numbers beyond the official source.
- Imported records may not contain complete authorship, affiliation, dataset, or ablation metadata.
- Reproduction should preserve the paper's data split, training budget, augmentation, and evaluation code before comparing results.

## YOLO-Agent Harness Hints / 对 YOLO-Agent 的启发

- Benchmark each degradation type separately instead of averaging all low-quality cases.

When testing this paper in YOLO-Agent, log the baseline, changed components, training budget, latency, AP metrics, and failure slices separately. Treat the paper as a hypothesis until the local reproduction confirms the claimed effect.

## 贡献清单 / Contribution Checklist

- [ ] Add the authors and affiliations.
- [ ] Replace the catalog summary with a paper-specific abstract summary.
- [ ] Record datasets, metrics, baselines, and key ablations.
- [ ] Add reproduction results and hardware/training details.
- [ ] Update limitations after independent verification.
