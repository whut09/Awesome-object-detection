#!/usr/bin/env python3
"""Render one structured Markdown note for every catalog paper."""

from __future__ import annotations

import argparse
from pathlib import Path

from catalog import load_catalog, note_relative_path


ROOT = Path(__file__).resolve().parents[1]


def bullet_values(values: list[str]) -> str:
    return "\n".join(f"- `{value}`" for value in values) or "- None recorded in the catalog."


def render_note(paper: dict) -> str:
    code_link = paper.get("official_code_url") or "Not recorded in the catalog."
    hints = paper.get("harness_hints", [])
    hint_text = "\n".join(f"- {hint}" for hint in hints) or "- Review the paper and reproduce its main comparison before adoption."
    summary = paper.get("summary", "No abstract-level summary is recorded yet.")
    return f"""# {paper['title']}

> This note is a structured first-pass summary generated from the project catalog. Please refine it with paper-specific quantitative details through contribution.

## 论文信息 / Paper Information

- **Venue:** {paper['publication']} {paper['year']}
- **Paper:** [{paper['paper_url']}]({paper['paper_url']})
- **Code:** {f"[official code]({code_link})" if code_link.startswith('http') else code_link}
- **Institution:** {paper.get('institution') or 'Not recorded'}
- **Category:** {paper['category']}
- **Detector family:** `{paper['detector_family']}`
- **Applicability:** `{paper['applicability']}`

## 摘要 / Abstract

{summary}

## 背景与问题 / Background and Problem

This work is cataloged under **{paper['category']}** for the `{paper['detector_family']}` detector family. Its recorded task scope is:

{bullet_values(paper.get('task_families', []))}

The catalog does not currently store a full verbatim abstract or a complete problem statement for every imported paper. The official paper link above is the source of truth for the detailed motivation, assumptions, and benchmark protocol.

## 方法概览 / Method Overview

The catalog identifies the following components for follow-up reading:

{bullet_values(paper.get('component_ids', []))}

For a faithful implementation, first establish the paper's original baseline, then isolate these components one at a time. Do not attribute the full paper gain to a single component without reproducing the ablations.

## 实验与证据 / Experiments and Evidence

- **Reported evidence:** See the official paper for datasets, metrics, baselines, ablations, and statistical significance.
- **Reproduction status:** No independent reproduction result is recorded in this catalog.
- **Code availability:** {code_link if code_link.startswith('http') else 'No official code link is recorded.'}

## 方法亮点 / Strengths

- The paper is indexed as a `{paper['applicability']}` for detector research and harness design.
- The method can be compared against the project's existing detector-family and task-family groupings.
- The note keeps the paper link, code link, and structured implementation hints together for later contribution.

## 局限与风险 / Limitations and Risks

- This first-pass note does not claim paper-specific numbers beyond the official source.
- Imported records may not contain complete authorship, affiliation, dataset, or ablation metadata.
- Reproduction should preserve the paper's data split, training budget, augmentation, and evaluation code before comparing results.

## YOLO-Agent Harness Hints / 对 YOLO-Agent 的启发

{hint_text}

When testing this paper in YOLO-Agent, log the baseline, changed components, training budget, latency, AP metrics, and failure slices separately. Treat the paper as a hypothesis until the local reproduction confirms the claimed effect.

## 贡献清单 / Contribution Checklist

- [ ] Add the authors and affiliations.
- [ ] Replace the catalog summary with a paper-specific abstract summary.
- [ ] Record datasets, metrics, baselines, and key ablations.
- [ ] Add reproduction results and hardware/training details.
- [ ] Update limitations after independent verification.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "papers.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    papers = load_catalog(args.catalog)
    paths: set[Path] = set()
    for paper in papers:
        relative = note_relative_path(paper)
        if relative in paths:
            raise ValueError(f"note path collision: {relative}")
        paths.add(relative)
        destination = args.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_note(paper), encoding="utf-8")
    print(f"rendered notes={len(papers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
