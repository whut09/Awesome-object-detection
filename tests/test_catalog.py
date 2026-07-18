import json
import tempfile
import unittest
from pathlib import Path

from scripts.catalog import (
    DEFAULT_CATALOG,
    harness_record,
    load_catalog,
    markdown_tables,
    sync_to_yolo_agent,
    validate_catalog,
    yolo_agent_record,
)
from scripts.merge_note_manifests import validate_note


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.papers = load_catalog(DEFAULT_CATALOG)

    def test_catalog_is_valid(self) -> None:
        self.assertEqual([], validate_catalog(self.papers))
        self.assertGreaterEqual(len(self.papers), 25)

    def test_yolo_export_uses_supported_schema(self) -> None:
        record = yolo_agent_record(self.papers[0])
        self.assertEqual("research.v1", record["schema_version"])
        self.assertEqual("paper_prior", harness_record(self.papers[0])["evidence_level"])
        self.assertNotIn("harness_hints", record)

    def test_completed_notes_are_linked(self) -> None:
        rendered = markdown_tables(self.papers)
        for paper in self.papers:
            if paper.get("note_path"):
                self.assertIn(f"[[note]({paper['note_path']})]", rendered)

    def test_notes_only_exist_in_yolo_agent_categories(self) -> None:
        allowed = {
            "Assignment, Loss, and Training",
            "General Object Detection",
            "Small, Aerial, and Oriented Detection",
            "YOLO and Real-Time Detection",
        }
        self.assertTrue(all(paper["category"] in allowed for paper in self.papers if paper.get("note_path")))

    def test_existing_notes_pass_content_validation(self) -> None:
        for paper in self.papers:
            if paper.get("note_path"):
                self.assertEqual([], validate_note(Path(paper["note_path"]), paper["title"]))

    def test_note_validation_rejects_catalog_placeholders(self) -> None:
        headings = [
            "## 一句话总结",
            "## 研究背景与问题",
            "## 方法总览",
            "## 方法详解",
            "## 实验与证据",
            "## 对 YOLO-Agent 的启发",
            "## 优点",
            "## 局限",
            "## 评分",
        ]
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "note.md"
            note.write_text(
                "# Example Paper\n\n"
                + "\n\n".join(f"{heading}\n论文实验节列出的任务数据" for heading in headings)
                + "\n"
                + "补充内容" * 300,
                encoding="utf-8",
            )
            errors = validate_note(note, "Example Paper")
            self.assertTrue(any("placeholder text found" in error for error in errors))

    def test_note_validation_rejects_garbled_repeated_characters(self) -> None:
        headings = [alternative[0] for alternative in (
            ("## 一句话总结",),
            ("## 研究背景与问题",),
            ("## 方法总览",),
            ("## 方法详解",),
            ("## 实验与证据",),
            ("## 对 YOLO-Agent 的启发",),
            ("## 优点",),
            ("## 局限",),
            ("## 评分",),
        )]
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "note.md"
            note.write_text(
                "# Example Paper\n\n"
                + "\n\n".join(f"{heading}\n有效内容" for heading in headings)
                + "\n纹纹纹纹\n"
                + "补充内容" * 300,
                encoding="utf-8",
            )
            errors = validate_note(note, "Example Paper")
            self.assertTrue(any("garbled repeated characters" in error for error in errors))

    def test_sync_merges_and_preserves_existing_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "research/papers.jsonl"
            registry.parent.mkdir(parents=True)
            registry.write_text(json.dumps({"paper_id": "manual:keep", "title": "Keep"}) + "\n", encoding="utf-8")
            sync_to_yolo_agent(self.papers[:2], root)
            records = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(3, len(records))
            self.assertTrue(registry.with_suffix(".jsonl.bak").is_file())
            self.assertIn("manual:keep", {record["paper_id"] for record in records})


if __name__ == "__main__":
    unittest.main()
