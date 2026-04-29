import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import stage72_eval_tool_calls
from scripts.stage72_eval_tool_calls import (
    build_retrieve_tool_message,
    extract_final_answer,
    parse_tool_arguments,
)


class ToolCallEvalTests(unittest.TestCase):
    def test_parse_tool_arguments_accepts_json_string(self):
        args = parse_tool_arguments('{"query": "who wrote Hamlet", "topk": 5}')

        self.assertEqual(args, {"query": "who wrote Hamlet", "topk": 5})

    def test_build_retrieve_tool_message_serializes_docs(self):
        searches = []
        tool_call = {
            "id": "call_1",
            "function": {
                "name": "retrieve",
                "arguments": json.dumps({"query": "who wrote Hamlet", "topk": 3}),
            },
        }
        docs = [
            {
                "rank": 1,
                "id": "doc-1",
                "title": "Hamlet",
                "score": 0.9,
                "contents": "Hamlet\nHamlet was written by William Shakespeare.",
            }
        ]

        message = build_retrieve_tool_message(
            tool_call,
            "who wrote Hamlet",
            docs,
            searches,
        )

        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "call_1")
        self.assertIn("William Shakespeare", message["content"])
        self.assertEqual(
            searches,
            [
                {
                    "turn": 1,
                    "query": "who wrote Hamlet",
                    "gold_in_top_docs": False,
                    "docs": docs,
                }
            ],
        )

    def test_chat_completion_omits_sampling_and_output_length_controls(self):
        captured = {}

        def fake_post_json(url, payload, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        args = SimpleNamespace(
            model_id="Qwen/Qwen3.6-35B-A3B",
            api_base="http://127.0.0.1:8080/v1",
            generation_timeout=30,
        )

        with patch.object(stage72_eval_tool_calls, "post_json", fake_post_json):
            message = stage72_eval_tool_calls.chat_completion(
                [{"role": "user", "content": "Question?"}],
                args,
                tool_choice="auto",
            )

            self.assertEqual(message, {"role": "assistant", "content": "ok"})
            self.assertEqual(
                captured["url"],
                "http://127.0.0.1:8080/v1/chat/completions",
            )
            self.assertNotIn("temperature", captured["payload"])
            self.assertNotIn("max_tokens", captured["payload"])
            self.assertEqual(captured["payload"]["tool_choice"], "auto")
            self.assertEqual(captured["payload"]["tools"], [stage72_eval_tool_calls.RETRIEVE_TOOL])

            stage72_eval_tool_calls.chat_completion(
                [{"role": "user", "content": "Final?"}],
                args,
                tool_choice="none",
            )
            self.assertNotIn("temperature", captured["payload"])
            self.assertNotIn("max_tokens", captured["payload"])
            self.assertEqual(captured["payload"]["tool_choice"], "none")
            self.assertEqual(captured["payload"]["tools"], [stage72_eval_tool_calls.RETRIEVE_TOOL])

    def test_extract_final_answer_strips_qwen_thinking_fallback(self):
        answer, closed = extract_final_answer(
            "<think>\nreasoning\n</think>\n\nWilhelm Röntgen"
        )

        self.assertEqual(answer, "Wilhelm Röntgen")
        self.assertTrue(closed)

    def test_run_one_executes_tool_calls_then_final_answer_once(self):
        calls = []

        def fake_chat_completion(messages, args, *, tool_choice):
            calls.append(tool_choice)
            if tool_choice == "auto":
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "retrieve",
                                "arguments": json.dumps(
                                    {"query": "who wrote Hamlet", "topk": 3}
                                ),
                            },
                        }
                    ],
                }
            return {"role": "assistant", "content": "William Shakespeare"}

        def fake_retrieve_docs(query, topk, args):
            self.assertEqual(query, "who wrote Hamlet")
            self.assertEqual(topk, 3)
            return (
                "",
                [
                    {
                        "rank": 1,
                        "id": "doc-1",
                        "title": "Hamlet",
                        "score": 0.9,
                        "contents": "Hamlet was written by William Shakespeare.",
                    }
                ],
            )

        example = {
            "dataset": "nq",
            "id": "test-1",
            "question": "who wrote Hamlet",
            "golden_answers": ["William Shakespeare"],
        }
        args = SimpleNamespace(topk=3)

        with patch.object(stage72_eval_tool_calls, "chat_completion", fake_chat_completion):
            with patch.object(stage72_eval_tool_calls, "retrieve_docs", fake_retrieve_docs):
                result = stage72_eval_tool_calls.run_one(example, args)

        self.assertEqual(calls, ["auto", "none"])
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["num_tool_calls"], 1)
        self.assertEqual(result["num_searches"], 1)
        self.assertIn('"role": "tool"', result["transcript"])

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

        def fake_run_one(example, args):
            return {
                **example,
                "prediction": example["golden_answers"][0],
                "exact_match": True,
                "failure_type": "correct",
                "outcome_type": "correct",
                "num_tool_calls": 1,
                "num_searches": 1,
                "searches": [],
                "latency_sec": 0.1,
                "transcript": "[]",
            }

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out_dir=Path(tmp), workers=1)
            with patch.object(stage72_eval_tool_calls, "run_one", fake_run_one):
                with patch.object(stage72_eval_tool_calls, "write_outputs") as write_outputs:
                    stage72_eval_tool_calls.run_eval(examples, args, {"run": "test"})

        self.assertEqual(write_outputs.call_count, 1)


if __name__ == "__main__":
    unittest.main()
