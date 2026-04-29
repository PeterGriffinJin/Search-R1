import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import stage2_eval_small


class Stage2EvalSmallTests(unittest.TestCase):
    def test_run_eval_does_not_rewrite_full_outputs_after_each_result(self):
        examples = [
            {
                "dataset": "nq",
                "id": "test-1",
                "question": "question 1",
                "golden_answers": ["answer 1"],
            },
            {
                "dataset": "nq",
                "id": "test-2",
                "question": "question 2",
                "golden_answers": ["answer 2"],
            },
        ]

        def fake_run_one(example, tokenizer, model, device, stopping_criteria, args):
            return {
                **example,
                "prediction": example["golden_answers"][0],
                "exact_match": True,
                "failure_type": "correct",
                "answer_tag_closed": True,
                "num_searches": 1,
                "searches": [
                    {
                        "turn": 1,
                        "query": example["question"],
                        "gold_in_top_docs": True,
                        "docs": [],
                    }
                ],
                "latency_sec": 0.1,
                "transcript": "",
            }

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out_dir=Path(tmp))
            with patch.object(stage2_eval_small, "run_one", fake_run_one):
                with patch.object(stage2_eval_small, "write_outputs") as write_outputs:
                    results = stage2_eval_small.run_eval(
                        examples,
                        tokenizer=None,
                        model=None,
                        device=None,
                        stopping_criteria=None,
                        args=args,
                        metadata={"run": "test"},
                    )

            jsonl_path = Path(tmp) / "results.jsonl"
            rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(write_outputs.call_count, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
