import unittest
import importlib.util
from pathlib import Path


def load_stage2_metrics():
    path = Path(__file__).resolve().parents[1] / "scripts" / "stage2_metrics.py"
    spec = importlib.util.spec_from_file_location("stage2_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage2_metrics = load_stage2_metrics()
classify_query_quality = stage2_metrics.classify_query_quality
max_answer_metrics = stage2_metrics.max_answer_metrics
summarize_results = stage2_metrics.summarize_results


class Stage2MetricsTest(unittest.TestCase):
    def test_max_answer_metrics_counts_strict_contains_and_f1(self):
        metrics = max_answer_metrics("3 February 1962", ["February 3, 1962", "1962"])

        self.assertFalse(metrics["strict_em"])
        self.assertTrue(metrics["relaxed_match"])
        self.assertTrue(metrics["prediction_contains_gold"])
        self.assertEqual(metrics["token_f1"], 1.0)

    def test_relaxed_match_does_not_count_yes_inside_longer_answer(self):
        metrics = max_answer_metrics("not yes", ["yes"])

        self.assertFalse(metrics["strict_em"])
        self.assertFalse(metrics["relaxed_match"])

    def test_classify_query_quality_flags_common_bad_queries(self):
        quality = classify_query_quality(
            [
                {"query": "query"},
                {"query": "<answer> leaked tag </answer>"},
                {"query": "who voiced thanos in guardians of the galaxy"},
            ]
        )

        self.assertEqual(quality["num_queries"], 3)
        self.assertEqual(quality["bad_query_count"], 2)
        self.assertEqual(quality["bad_query_reasons"], ["too_short", "tag_leak"])

    def test_summarize_results_adds_relaxed_metrics_by_dataset(self):
        results = [
            {
                "dataset": "nq",
                "prediction": "3 February 1962",
                "golden_answers": ["February 3, 1962"],
                "exact_match": False,
                "failure_type": "evidence_use_error",
                "searches": [{"query": "us trade cuba", "gold_in_top_docs": True}],
                "num_searches": 1,
                "latency_sec": 1.0,
            },
            {
                "dataset": "hotpotqa",
                "prediction": "wrong",
                "golden_answers": ["right"],
                "exact_match": False,
                "failure_type": "retrieval_miss",
                "searches": [{"query": "query", "gold_in_top_docs": False}],
                "num_searches": 1,
                "latency_sec": 2.0,
            },
        ]

        summary, enriched = summarize_results(results, {"model_id": "dummy"})

        self.assertEqual(summary["strict_em"], 0)
        self.assertEqual(summary["relaxed_em"], 1)
        self.assertEqual(summary["by_dataset"]["nq"]["relaxed_em"], 1)
        self.assertEqual(summary["by_failure_type"]["retrieval_miss"], 1)
        self.assertTrue(enriched[0]["answer_metrics"]["relaxed_match"])
        self.assertEqual(enriched[1]["query_quality"]["bad_query_count"], 1)

    def test_summarize_results_adds_multisearch_closure_metrics(self):
        results = [
            {
                "dataset": "hotpotqa",
                "prediction": "right",
                "golden_answers": ["right"],
                "exact_match": True,
                "failure_type": "correct",
                "answer_tag_closed": True,
                "searches": [
                    {"query": "first", "gold_in_top_docs": False},
                    {"query": "second", "gold_in_top_docs": True},
                ],
                "num_searches": 2,
                "latency_sec": 1.0,
            },
            {
                "dataset": "hotpotqa",
                "prediction": "",
                "golden_answers": ["right"],
                "exact_match": False,
                "failure_type": "no_answer_tag",
                "answer_tag_closed": False,
                "searches": [
                    {"query": "first", "gold_in_top_docs": False},
                    {"query": "second", "gold_in_top_docs": False},
                    {"query": "third", "gold_in_top_docs": False},
                ],
                "num_searches": 3,
                "latency_sec": 2.0,
            },
            {
                "dataset": "nq",
                "prediction": "wrong",
                "golden_answers": ["right"],
                "exact_match": False,
                "failure_type": "retrieval_miss",
                "answer_tag_closed": True,
                "searches": [{"query": "one", "gold_in_top_docs": False}],
                "num_searches": 1,
                "latency_sec": 1.0,
            },
        ]

        summary, _ = summarize_results(results, {})

        behavior = summary["search_behavior"]
        self.assertEqual(behavior["multi_search"], 2)
        self.assertEqual(behavior["closed_after_multi_search"], 1)
        self.assertEqual(behavior["closed_answer_after_2plus_searches"], 1)
        self.assertEqual(behavior["strict_after_multi_search"], 1)
        self.assertEqual(behavior["multi_search_rate"], 2 / 3)
        self.assertEqual(behavior["closed_after_multi_search_rate"], 0.5)
        self.assertEqual(behavior["closed_answer_after_2plus_search_rate"], 0.5)
        self.assertEqual(behavior["strict_after_multi_search_rate"], 0.5)
        self.assertEqual(summary["by_dataset"]["hotpotqa"]["multi_search"], 2)
        self.assertEqual(summary["by_dataset"]["hotpotqa"]["avg_num_searches"], 2.5)
        self.assertEqual(summary["by_dataset"]["nq"]["multi_search"], 0)


if __name__ == "__main__":
    unittest.main()
