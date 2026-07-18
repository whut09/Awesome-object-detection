#!/usr/bin/env python3
"""Validate, render, export, and sync the paper catalog."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "papers.json"
README_START = "<!-- PAPER_TABLES:START -->"
README_END = "<!-- PAPER_TABLES:END -->"
REQUIRED_FIELDS = {
    "paper_id", "title", "year", "publication", "category", "paper_url",
    "institution", "summary", "task_families", "detector_family",
    "component_ids", "applicability", "harness_hints",
}
ALLOWED_APPLICABILITY = {
    "direct_adapter_candidate", "recipe_idea_only", "separate_detector_family",
    "incompatible", "insufficient_information",
}
NOTE_CATEGORIES = {
    "Assignment, Loss, and Training",
    "General Object Detection",
    "Small, Aerial, and Oriented Detection",
    "YOLO and Real-Time Detection",
}


def load_catalog(path: Path = DEFAULT_CATALOG) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("catalog root must be a JSON array")
    return data


def validate_catalog(papers: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    for index, paper in enumerate(papers):
        label = paper.get("paper_id", f"row:{index}")
        missing = sorted(REQUIRED_FIELDS - set(paper))
        if missing:
            errors.append(f"{label}: missing {', '.join(missing)}")
        paper_id = str(paper.get("paper_id", "")).strip().lower()
        title = " ".join(str(paper.get("title", "")).lower().split())
        if paper_id in seen_ids:
            errors.append(f"{label}: duplicate paper_id")
        if title in seen_titles:
            errors.append(f"{label}: duplicate title")
        seen_ids.add(paper_id)
        seen_titles.add(title)
        year = paper.get("year")
        if not isinstance(year, int) or not 2021 <= year <= 2026:
            errors.append(f"{label}: year must be in 2021..2026")
        if paper.get("applicability") not in ALLOWED_APPLICABILITY:
            errors.append(f"{label}: invalid applicability")
        for key in ("paper_url", "official_code_url"):
            value = paper.get(key)
            if value and urlparse(value).scheme not in {"http", "https"}:
                errors.append(f"{label}: {key} must be an HTTP(S) URL")
        note_path = paper.get("note_path")
        if note_path:
            if paper.get("category") not in NOTE_CATEGORIES:
                errors.append(f"{label}: notes are not enabled for category {paper.get('category')}")
            note_file = ROOT / str(note_path)
            if note_file.suffix.lower() != ".md":
                errors.append(f"{label}: note_path must point to a Markdown file")
            elif not note_file.is_file():
                errors.append(f"{label}: note_path does not exist: {note_path}")
        for key in ("task_families", "component_ids", "harness_hints"):
            if not isinstance(paper.get(key), list):
                errors.append(f"{label}: {key} must be a list")
    return errors


def markdown_tables(papers: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper in papers:
        grouped[paper["category"]].append(paper)
    sections: list[str] = []
    for category in sorted(grouped):
        sections.extend([
            f"## {category}",
            "| Year | Pub | Title | Links | Main Institution |",
            "|:---:|:---:|:---|:---:|:---:|",
        ])
        for paper in sorted(grouped[category], key=lambda item: (-item["year"], item["title"].lower())):
            links = f"[[paper]({paper['paper_url']})]"
            if paper.get("official_code_url"):
                links += f" [[code]({paper['official_code_url']})]"
            if paper.get("note_path"):
                links += f" [[note]({paper['note_path']})]"
            sections.append(
                f"| {paper['year']} | {paper['publication']} | **{paper['title']}** | {links} | {paper['institution']} |"
            )
        sections.append("")
    return "\n".join(sections).rstrip()


def render_readme(papers: list[dict[str, Any]], readme_path: Path) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise ValueError("README paper table markers are missing")
    before, rest = text.split(README_START, 1)
    _, after = rest.split(README_END, 1)
    rendered = f"{before}{README_START}\n{markdown_tables(papers)}\n{README_END}{after}"
    atomic_write_text(readme_path, rendered)


def yolo_agent_record(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "research.v1",
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "abstract": paper["summary"],
        "year": paper["year"],
        "authors": paper.get("authors", []),
        "task_families": paper["task_families"],
        "detector_family": paper["detector_family"],
        "source_url": paper["paper_url"],
        "paper_url": paper["paper_url"],
        "official_code_url": paper.get("official_code_url"),
        "datasets": paper.get("datasets", []),
        "benchmarks": [],
        "training_budget": {},
        "claimed_effects": [],
        "component_ids": paper["component_ids"],
        "applicability": paper["applicability"],
        "source": "awesome_object_detection",
        "ingestion_version": "awesome_object_detection.v1",
    }


def harness_record(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "year": paper["year"],
        "detector_family": paper["detector_family"],
        "task_families": paper["task_families"],
        "component_ids": paper["component_ids"],
        "applicability": paper["applicability"],
        "hints": paper["harness_hints"],
        "evidence_level": "paper_prior",
        "requires_local_reproduction": True,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, payload)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{number}: {error}") from error
    return rows


def sync_to_yolo_agent(papers: list[dict[str, Any]], yolo_root: Path) -> Path:
    registry_path = yolo_root.resolve() / "research" / "papers.jsonl"
    existing = {row["paper_id"]: row for row in read_jsonl(registry_path)}
    for paper in papers:
        existing[paper["paper_id"]] = yolo_agent_record(paper)
    if registry_path.exists():
        shutil.copy2(registry_path, registry_path.with_suffix(".jsonl.bak"))
    write_jsonl(registry_path, [existing[key] for key in sorted(existing)])
    return registry_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    render = commands.add_parser("render")
    render.add_argument("--readme", type=Path, default=ROOT / "README.md")
    export = commands.add_parser("export-yolo")
    export.add_argument("--out", type=Path, default=ROOT / "export/yolo_agent/papers.jsonl")
    hints = commands.add_parser("export-hints")
    hints.add_argument("--out", type=Path, default=ROOT / "export/yolo_agent/harness_hints.jsonl")
    sync = commands.add_parser("sync-yolo")
    sync.add_argument("--yolo-root", type=Path, required=True)
    commands.add_parser("stats")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    papers = load_catalog(args.catalog)
    errors = validate_catalog(papers)
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    if args.command == "validate":
        print(f"valid papers={len(papers)}")
    elif args.command == "render":
        render_readme(papers, args.readme)
        print(f"rendered {args.readme}")
    elif args.command == "export-yolo":
        write_jsonl(args.out, [yolo_agent_record(paper) for paper in papers])
        print(f"exported papers={len(papers)} path={args.out}")
    elif args.command == "export-hints":
        write_jsonl(args.out, [harness_record(paper) for paper in papers])
        print(f"exported hints={len(papers)} path={args.out}")
    elif args.command == "sync-yolo":
        print(f"synced papers={len(papers)} path={sync_to_yolo_agent(papers, args.yolo_root)}")
    elif args.command == "stats":
        print(f"papers={len(papers)}")
        print("years=" + json.dumps(Counter(paper["year"] for paper in papers), sort_keys=True))
        print("categories=" + json.dumps(Counter(paper["category"] for paper in papers), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
