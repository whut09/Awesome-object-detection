import json
import tempfile
import unittest
from pathlib import Path

from scripts.catalog import (
    DEFAULT_CATALOG,
    harness_record,
    load_catalog,
    sync_to_yolo_agent,
    validate_catalog,
    yolo_agent_record,
)


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
