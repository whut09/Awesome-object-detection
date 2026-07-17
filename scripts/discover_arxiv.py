#!/usr/bin/env python3
"""Discover recent arXiv candidates without changing the curated catalog."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path


ATOM = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_QUERIES = [
    "object detection", "real-time object detection", "small object detection",
    "oriented object detection", "open-vocabulary object detection",
    "open-world object detection",
]


def fetch(query: str, start_year: int, max_results: int) -> list[dict[str, object]]:
    search = f'all:"{query}" AND submittedDate:[{start_year}01010000 TO {date.today():%Y%m%d}2359]'
    params = urllib.parse.urlencode({
        "search_query": search, "start": 0, "max_results": max_results,
        "sortBy": "submittedDate", "sortOrder": "descending",
    })
    request = urllib.request.Request(
        f"https://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": "awesome-object-detection/0.1 (paper catalog)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())
    rows = []
    for entry in root.findall("atom:entry", ATOM):
        arxiv_id = entry.findtext("atom:id", default="", namespaces=ATOM).rsplit("/", 1)[-1]
        stable_id = arxiv_id.split("v", 1)[0]
        rows.append({
            "paper_id": f"arxiv:{stable_id}",
            "title": " ".join(entry.findtext("atom:title", default="", namespaces=ATOM).split()),
            "published_at": entry.findtext("atom:published", default="", namespaces=ATOM),
            "updated_at": entry.findtext("atom:updated", default="", namespaces=ATOM),
            "paper_url": f"https://arxiv.org/abs/{stable_id}",
            "summary": " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM).split()),
            "authors": [
                author.findtext("atom:name", default="", namespaces=ATOM)
                for author in entry.findall("atom:author", ATOM)
            ],
            "matched_query": query,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--out", type=Path, default=Path("data/candidates/arxiv.jsonl"))
    args = parser.parse_args()
    deduplicated: dict[str, dict[str, object]] = {}
    for query in args.queries or DEFAULT_QUERIES:
        for row in fetch(query, args.start_year, args.max_results):
            existing = deduplicated.get(str(row["paper_id"]))
            if existing is None:
                deduplicated[str(row["paper_id"])] = row
            else:
                matched = set(str(existing["matched_query"]).split(" | "))
                matched.add(query)
                existing["matched_query"] = " | ".join(sorted(matched))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in deduplicated.values()),
        encoding="utf-8",
    )
    print(f"discovered candidates={len(deduplicated)} path={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
