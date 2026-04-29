from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

try:
    from scripts.stage2_eval_small import (
        answer_in_docs,
        classify_failure,
        exact_match,
        format_passages,
        load_examples,
        load_examples_file,
        write_examples_file,
        write_outputs,
    )
    from scripts.stage2_metrics import (
        classify_query_quality,
        max_answer_metrics,
        summarize_results,
    )
except ImportError:
    from stage2_eval_small import (
        answer_in_docs,
        classify_failure,
        exact_match,
        format_passages,
        load_examples,
        load_examples_file,
        write_examples_file,
        write_outputs,
    )
    from stage2_metrics import (
        classify_query_quality,
        max_answer_metrics,
        summarize_results,
    )


SYSTEM_PROMPT = """Answer the user's question using the retrieve tool when external evidence may help.
You may include multiple retrieve tool calls in one assistant message.
When asked for the final answer, provide only the concise answer span without analysis, citations, or explanations."""

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Search a Wikipedia corpus and return the top passages for a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query for the missing evidence.",
                },
                "topk": {
                    "type": "integer",
                    "description": "Number of passages to retrieve.",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass
class ToolEvalState:
    example: dict
    messages: list[dict[str, Any]]
    start_time: float
    searches: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    final_answer: str = ""
    answer_tag_closed: bool = False


def sample_key(example: dict) -> str:
    return f"{example.get('dataset')}:{example.get('id')}"


def load_existing_results(out_dir: Path) -> dict[str, dict]:
    results_path = out_dir / "results.jsonl"
    if not results_path.exists():
        return {}
    rows = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return {sample_key(row): row for row in rows}


def parse_tool_arguments(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        data = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def post_json(url: str, payload: dict, timeout: int) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(
            f"POST {url} failed: HTTP {response.status_code}: {response.text[:1000]}"
        )
    return response.json()


def normalize_assistant_message(message: dict) -> dict:
    normalized = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if message.get("tool_calls"):
        normalized["tool_calls"] = message["tool_calls"]
    return normalized


def chat_completion(
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    tool_choice: str | None,
) -> dict:
    payload: dict[str, Any] = {
        "model": args.model_id,
        "messages": messages,
    }
    if tool_choice:
        payload["tools"] = [RETRIEVE_TOOL]
        payload["tool_choice"] = tool_choice

    data = post_json(
        f"{args.api_base.rstrip('/')}/chat/completions",
        payload,
        timeout=args.generation_timeout,
    )
    return data["choices"][0]["message"]


def retrieve_docs(query: str, topk: int, args: argparse.Namespace) -> tuple[str, list[dict]]:
    topk = max(1, min(int(topk or args.topk), 10))
    data = post_json(
        args.retriever_url,
        {"queries": [query], "topk": topk, "return_scores": True},
        timeout=args.retrieval_timeout,
    )
    return format_passages(data["result"][0])


def format_docs_for_tool(query: str, docs: list[dict]) -> str:
    rows = [f"Query: {query}", "Retrieved passages:"]
    for doc in docs:
        title = doc.get("title", "")
        contents = doc.get("contents", "")
        score = doc.get("score")
        rows.append(f"Doc {doc.get('rank')}(Title: {title}, Score: {score}) {contents}")
    return "\n".join(rows)


def build_retrieve_tool_message(
    tool_call: dict,
    query: str,
    docs: list[dict],
    searches: list[dict],
    golden_answers: list[str] | None = None,
) -> dict:
    searches.append(
        {
            "turn": len(searches) + 1,
            "query": query,
            "gold_in_top_docs": answer_in_docs(golden_answers or [], docs)
            if golden_answers
            else False,
            "docs": docs,
        }
    )
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id", f"call_{len(searches)}"),
        "content": format_docs_for_tool(query, docs),
    }


def extract_final_answer(content: str) -> tuple[str, bool]:
    text = (content or "").strip()
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip(), True
    fallback = re.findall(r"<answer>(.*)", text, flags=re.DOTALL)
    if fallback:
        return fallback[-1].strip(), False
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text, bool(text)


def initial_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question.strip()}"},
    ]


def finalize_state(state: ToolEvalState) -> dict:
    example = state.example
    is_em = exact_match(state.final_answer, example["golden_answers"])
    outcome_type = classify_failure(is_em, state.answer_tag_closed, state.searches)
    answer_metrics = max_answer_metrics(state.final_answer, example["golden_answers"])
    query_quality = classify_query_quality(state.searches)
    return {
        **example,
        "prediction": state.final_answer,
        "exact_match": is_em,
        "relaxed_match": answer_metrics["relaxed_match"],
        "token_f1": answer_metrics["token_f1"],
        "answer_metrics": answer_metrics,
        "query_quality": query_quality,
        "failure_type": outcome_type,
        "outcome_type": outcome_type,
        "answer_tag_closed": state.answer_tag_closed,
        "num_searches": len(state.searches),
        "num_tool_calls": len(state.tool_calls),
        "tool_calls": state.tool_calls,
        "searches": state.searches,
        "latency_sec": round(time.time() - state.start_time, 3),
        "transcript": json.dumps(state.messages, ensure_ascii=False),
    }


def finalize_error(example: dict, exc: BaseException) -> dict:
    answer_metrics = max_answer_metrics("", example["golden_answers"])
    return {
        **example,
        "prediction": "",
        "exact_match": False,
        "relaxed_match": answer_metrics["relaxed_match"],
        "token_f1": answer_metrics["token_f1"],
        "answer_metrics": answer_metrics,
        "query_quality": {
            "num_queries": 0,
            "bad_query_count": 0,
            "bad_query_reasons": [],
            "queries": [],
        },
        "failure_type": "request_error",
        "outcome_type": "request_error",
        "answer_tag_closed": False,
        "num_searches": 0,
        "num_tool_calls": 0,
        "tool_calls": [],
        "searches": [],
        "latency_sec": 0,
        "request_error": repr(exc),
        "transcript": "[]",
    }


def write_resume_outputs(
    examples: list[dict],
    completed: dict[str, dict],
    out_dir: Path,
    metadata: dict,
) -> None:
    ordered = [
        completed[sample_key(example)]
        for example in examples
        if sample_key(example) in completed
    ]
    write_outputs(ordered, out_dir, metadata)


def ordered_completed_results(
    examples: list[dict], completed: dict[str, dict]
) -> list[dict]:
    return [
        completed[sample_key(example)]
        for example in examples
        if sample_key(example) in completed
    ]


def append_result_output(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def write_summary_output(
    examples: list[dict],
    completed: dict[str, dict],
    out_dir: Path,
    metadata: dict,
) -> None:
    ordered = ordered_completed_results(examples, completed)
    summary, _ = summarize_results(ordered, metadata)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_progress(result: dict, completed_count: int, total: int) -> None:
    print(
        f"[{completed_count}/{total}] {result['dataset']} {result['id']} "
        f"em={result['exact_match']} tools={result['num_tool_calls']} "
        f"outcome={result['outcome_type']} pred={result['prediction']!r}",
        flush=True,
    )


def run_one(example: dict, args: argparse.Namespace) -> dict:
    state = ToolEvalState(
        example=example,
        messages=initial_messages(example["question"]),
        start_time=time.time(),
    )

    message = chat_completion(state.messages, args, tool_choice="auto")
    assistant_message = normalize_assistant_message(message)
    state.messages.append(assistant_message)
    tool_calls = assistant_message.get("tool_calls") or []

    if not tool_calls:
        answer, closed = extract_final_answer(assistant_message.get("content", ""))
        state.final_answer = answer
        state.answer_tag_closed = closed
        return finalize_state(state)

    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        name = function.get("name")
        arguments = parse_tool_arguments(function.get("arguments"))
        state.tool_calls.append(
            {
                "id": tool_call.get("id"),
                "name": name,
                "arguments": arguments,
            }
        )
        if name != "retrieve":
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", "unknown_tool_call"),
                    "content": f"Unsupported tool: {name}",
                }
            )
            continue

        query = str(arguments.get("query", "")).strip()
        if not query:
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", "empty_query_call"),
                    "content": "Retriever error: missing non-empty query.",
                }
            )
            continue

        topk = arguments.get("topk", args.topk)
        _, docs = retrieve_docs(query, topk, args)
        state.messages.append(
            build_retrieve_tool_message(
                tool_call,
                query,
                docs,
                state.searches,
                example["golden_answers"],
            )
        )

    state.messages.append(
        {
            "role": "user",
            "content": "Use the retrieved evidence and provide the final concise answer now. Do not call tools.",
        }
    )
    final_message = chat_completion(state.messages, args, tool_choice="none")
    assistant_message = normalize_assistant_message(final_message)
    state.messages.append(assistant_message)
    answer, closed = extract_final_answer(assistant_message.get("content", ""))
    state.final_answer = answer
    state.answer_tag_closed = closed
    return finalize_state(state)


def run_eval(
    examples: list[dict],
    args: argparse.Namespace,
    metadata: dict,
    finalize_outputs: bool = True,
) -> list[dict]:
    completed = load_existing_results(args.out_dir)
    if completed:
        print(f"resuming with {len(completed)} completed examples", flush=True)
        write_summary_output(examples, completed, args.out_dir, metadata)

    pending = [
        example for example in examples if sample_key(example) not in completed
    ]
    if args.workers <= 1:
        for example in pending:
            result = run_one(example, args)
            completed[sample_key(example)] = result
            print_progress(result, len(completed), len(examples))
            append_result_output(result, args.out_dir)
            write_summary_output(examples, completed, args.out_dir, metadata)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one, example, args): example for example in pending
            }
            for future in as_completed(futures):
                example = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = finalize_error(example, exc)
                completed[sample_key(example)] = result
                print_progress(result, len(completed), len(examples))
                append_result_output(result, args.out_dir)
                write_summary_output(examples, completed, args.out_dir, metadata)

    if finalize_outputs:
        write_resume_outputs(examples, completed, args.out_dir, metadata)
    return [completed[sample_key(example)] for example in examples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--examples-file", type=Path)
    parser.add_argument("--write-examples-file", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--generation-timeout", type=int, default=600)
    parser.add_argument("--retrieval-timeout", type=int, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.examples_file:
        examples = load_examples_file(args.examples_file)
    else:
        examples = load_examples(args.per_dataset, args.seed)
    if args.limit:
        examples = examples[: args.limit]
    if args.write_examples_file:
        write_examples_file(examples, args.write_examples_file)

    metadata = {
        "model_id": args.model_id,
        "backend": "openai-compatible-vllm-chat-tools",
        "api_base": args.api_base,
        "retriever_url": args.retriever_url,
        "tool_name": "retrieve",
        "topk": args.topk,
        "per_dataset": args.per_dataset,
        "seed": args.seed,
        "examples_file": str(args.examples_file) if args.examples_file else None,
        "write_examples_file": str(args.write_examples_file)
        if args.write_examples_file
        else None,
        "limit": args.limit,
        "workers": args.workers,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    run_eval(examples, args, metadata, finalize_outputs=False)
    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    existing = load_existing_results(args.out_dir)
    write_resume_outputs(examples, existing, args.out_dir, metadata)
    print(f"wrote {args.out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
