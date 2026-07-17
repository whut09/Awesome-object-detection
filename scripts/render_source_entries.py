#!/usr/bin/env python3
"""Render raw entries transcribed from the supplied screenshots."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "wechat_screenshot_entries.json"
OUTPUT = ROOT / "docs" / "wechat-screenshot-entries.md"


def main() -> int:
    entries = json.loads(SOURCE.read_text(encoding="utf-8"))
    lines = [
        "# WeChat Screenshot Source Entries",
        "",
        "This file preserves the entries transcribed from the supplied screenshots.",
        "The source list is multi-label, so some entries are not object-detection papers.",
        "They are intentionally kept here before paper-level verification and deduplication.",
        "",
        "| Rank | Title | Venue | Year |",
        "|:---:|:---|:---:|:---:|",
    ]
    for entry in sorted(entries, key=lambda item: item["rank"], reverse=True):
        title = entry["title"].replace("|", "\\|")
        lines.append(f"| {entry['rank']} | {title} | {entry['venue']} | {entry['year']} |")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rendered entries={len(entries)} path={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
