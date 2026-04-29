from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "stage3_analyze_failure_taxonomy.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("stage3_analyze_failure_taxonomy", MODULE_PATH)
stage3_failure_taxonomy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stage3_failure_taxonomy)


def row(failure_type: str, subtype: str) -> dict:
    return {
        "dataset": "nq",
        "id": f"id-{failure_type}-{subtype}",
        "question": "question",
        "golden_answers": "[\"gold\"]",
        "prediction": "pred",
        "failure_type": failure_type,
        "subtype": subtype,
        "queries": "[\"query\"]",
    }


class Stage3FailureTaxonomyTest(unittest.TestCase):
    def test_load_jsonl_preserves_unicode_line_separator_inside_string(self) -> None:
        row_data = {
            "dataset": "hotpotqa",
            "id": "dev-line-separator",
            "prediction": "first\u2028second",
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(
                json.dumps(row_data, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            loaded = stage3_failure_taxonomy.load_jsonl(path)

        self.assertEqual(loaded, [row_data])

    def test_markdown_report_uses_dynamic_next_training_changes(self) -> None:
        rows = [
            row("correct", "correct_or_unclassified"),
            row("retrieval_miss", "missing_multihop_bridge_query"),
            row("retrieval_miss", "direct_topk_miss"),
            row("evidence_use_error", "distractor_span_selection"),
            row("no_answer_tag", "unproductive_repeat_search_loop"),
        ]
        summary = {"metadata": {}, "strict_em": 1, "total": 5}

        report = stage3_failure_taxonomy.markdown_report(
            summary, rows, Path("failure_taxonomy.csv")
        )

        self.assertIn("# Failure Taxonomy", report)
        self.assertNotIn("Stage3 Step10", report)
        self.assertNotIn("21/50", report)
        self.assertIn("1/5 (20.0%) runs lack a closed final answer", report)
        self.assertIn("2/5 (40.0%) runs are retrieval_miss", report)
        self.assertIn("missing_multihop_bridge_query=1", report)


if __name__ == "__main__":
    unittest.main()
