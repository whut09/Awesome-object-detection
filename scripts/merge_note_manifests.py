#!/usr/bin/env python3
"""Validate generated note manifests and merge note paths into the catalog."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.catalog import DEFAULT_CATALOG, NOTE_CATEGORIES, atomic_write_text, load_catalog


DEFAULT_MANIFEST_DIR = ROOT / "data" / "note_manifests"
REQUIRED_HEADING_GROUPS = (
    ("## 一句话总结",),
    ("## 研究背景与问题", "## 背景与问题", "## 研究背景与动机"),
    ("## 方法总览", "## 整体框架", "## 核心思路"),
    ("## 方法详解",),
    ("## 实验与证据", "## 实验与消融"),
    ("## 对 YOLO-Agent 的启发",),
    ("## 优点",),
    ("## 局限",),
    ("## 评分",),
)
BANNED_PLACEHOLDER_PHRASES = (
    "The catalog does not currently store",
    "Review the official paper and reproduce",
    "The official paper link above is the source of truth",
    "这项工作的价值不是再堆一个模块",
    "工程上还存在一个常被忽略的问题",
    "论文组件可能只在某个骨干",
    "构造比普通硬标签或逐点回归更有信息量的监督",
    "论文实验节列出的任务数据",
    "论文选定的同任务检测基线",
    "实际模块名",
    "采用三臂对照",
    "技术路径可以拆成明确的数据流和可开关组件",
)


def load_manifests(directory: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"manifest must contain an array: {path}")
        for entry in data:
            if set(entry) != {"paper_id", "note_path"}:
                raise ValueError(f"invalid manifest entry in {path}: {entry}")
            entries.append(entry)
    return entries


def validate_note(path: Path, title: str) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing note file: {path}"]
    text = path.read_text(encoding="utf-8")
    if len(text) < 1200:
        errors.append(f"note is too short: {path}")
    if f"# {title}" not in text:
        errors.append(f"title mismatch: {path}")
    for alternatives in REQUIRED_HEADING_GROUPS:
        if not any(heading in text for heading in alternatives):
            errors.append(f"missing heading {' or '.join(alternatives)}: {path}")
    for phrase in BANNED_PLACEHOLDER_PHRASES:
        if phrase in text:
            errors.append(f"placeholder text found in {path}: {phrase}")
    repeated_character = re.search(r"([\u4e00-\u9fff])\1{3,}", text)
    if repeated_character:
        errors.append(f"garbled repeated characters found in {path}: {repeated_character.group(0)[:20]}")
    return errors


def repeated_sentences(paths: list[Path]) -> list[str]:
    occurrences: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for sentence in re.split(r"[。！？\n]+", text):
            normalized = re.sub(r"[`*_#>|\[\]()]", "", sentence)
            normalized = re.sub(r"\s+", "", normalized).strip("-:：;；,.，")
            if len(normalized) < 28 or normalized.startswith(("title:", "description:", "tags:")):
                continue
            if path not in occurrences[normalized]:
                occurrences[normalized].append(path)
    errors: list[str] = []
    for sentence, sentence_paths in occurrences.items():
        if len(sentence_paths) >= 3:
            listed = ", ".join(str(path) for path in sentence_paths[:4])
            errors.append(f"repeated boilerplate in {len(sentence_paths)} notes: {sentence[:80]} [{listed}]")
    return errors


def catalog_json(papers: list[dict]) -> str:
    rows = [json.dumps(paper, ensure_ascii=False, separators=(",", ":")) for paper in papers]
    return "[\n  " + ",\n  ".join(rows) + "\n]\n"


def merge(manifest_dir: Path, catalog_path: Path, check_only: bool) -> tuple[int, list[str]]:
    papers = load_catalog(catalog_path)
    by_id = {paper["paper_id"]: paper for paper in papers}
    entries = load_manifests(manifest_dir)
    errors: list[str] = []
    seen: set[str] = set()
    note_files: list[Path] = []
    for entry in entries:
        paper_id = entry["paper_id"]
        if paper_id in seen:
            errors.append(f"duplicate manifest paper_id: {paper_id}")
            continue
        seen.add(paper_id)
        paper = by_id.get(paper_id)
        if paper is None:
            errors.append(f"unknown paper_id: {paper_id}")
            continue
        if paper["category"] not in NOTE_CATEGORIES:
            errors.append(f"note category is not enabled: {paper_id} {paper['category']}")
        note_path = entry["note_path"]
        if not note_path.startswith("docs/") or not note_path.endswith(".md"):
            errors.append(f"invalid note_path: {paper_id} {note_path}")
            continue
        note_file = ROOT / note_path
        errors.extend(validate_note(note_file, paper["title"]))
        note_files.append(note_file)
        paper["note_path"] = note_path
    if not errors:
        errors.extend(repeated_sentences(note_files))
    if not errors and not check_only:
        atomic_write_text(catalog_path, catalog_json(papers))
    return len(entries), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    count, errors = merge(args.manifest_dir, args.catalog, args.check)
    for error in errors:
        print(f"error: {error}")
    if errors:
        return 1
    print(f"valid note manifests={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
