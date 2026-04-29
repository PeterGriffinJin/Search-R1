from __future__ import annotations

import argparse
import csv
import json
import re
import string
import time
from pathlib import Path

import requests
import torch
import transformers
from datasets import load_dataset

try:
    from scripts.stage2_metrics import (
        classify_query_quality,
        max_answer_metrics,
        summarize_results,
    )
except ImportError:
    from stage2_metrics import classify_query_quality, max_answer_metrics, summarize_results


SEARCH_STOPS = [
    "</search>",
    " </search>",
    "</search>\n",
    " </search>\n",
    "</search>\n\n",
    " </search>\n\n",
]


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for target_id, target_len in zip(self.target_ids, self.target_lengths):
            target = torch.as_tensor(target_id, device=input_ids.device)
            if torch.equal(input_ids[0, -target_len:], target):
                return True
        return False


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(answer) for answer in answers)


def answer_in_docs(answers: list[str], docs: list[dict]) -> bool:
    normalized_docs = " ".join(
        normalize_answer(doc.get("contents", "")) for doc in docs
    )
    return any(normalize_answer(answer) in normalized_docs for answer in answers)


def extract_latest_search(text: str) -> str | None:
    matches = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_answer(text: str) -> tuple[str, bool]:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip(), True
    fallback = re.findall(r"<answer>(.*)", text, flags=re.DOTALL)
    if fallback:
        return fallback[-1].strip(), False
    return "", False


def build_prompt(question: str) -> str:
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    return f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


def format_passages(retrieval_result):
    formatted = ""
    docs = []
    for idx, doc_item in enumerate(retrieval_result):
        document = doc_item["document"]
        content = document["contents"]
        title = content.split("\n")[0].strip('"')
        text = "\n".join(content.split("\n")[1:])
        score = doc_item.get("score")
        docs.append(
            {
                "rank": idx + 1,
                "id": document.get("id"),
                "title": title,
                "score": score,
                "contents": content,
                "snippet": text[:500],
            }
        )
        formatted += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return formatted, docs


def search(query: str, retriever_url: str, topk: int) -> tuple[str, list[dict]]:
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    response = requests.post(retriever_url, json=payload, timeout=180)
    response.raise_for_status()
    results = response.json()["result"][0]
    return format_passages(results)


def classify_failure(is_em: bool, answer_tag_closed: bool, searches: list[dict]) -> str:
    if is_em:
        return "correct"
    if not answer_tag_closed:
        return "no_answer_tag"
    if not searches:
        return "no_search"
    if any(turn["gold_in_top_docs"] for turn in searches):
        return "evidence_use_error"
    return "retrieval_miss"


def load_examples(per_dataset: int, seed: int) -> list[dict]:
    specs = [("nq", "test"), ("hotpotqa", "dev")]
    examples = []
    for dataset_name, split in specs:
        dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset_name, split=split)
        dataset = dataset.shuffle(seed=seed).select(range(per_dataset))
        for row in dataset:
            examples.append(
                {
                    "dataset": dataset_name,
                    "id": row.get("id"),
                    "question": row["question"],
                    "golden_answers": list(row["golden_answers"]),
                }
            )
    return examples


def load_examples_file(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def write_examples_file(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def run_one(example, tokenizer, model, device, stopping_criteria, args) -> dict:
    prompt = build_prompt(example["question"])
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

    searches = []
    transcript_parts = [prompt]
    final_answer = ""
    answer_tag_closed = False
    start_time = time.time()

    for turn in range(args.max_turns + 1):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)
        do_sample = args.temperature > 0
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
        outputs = model.generate(**generation_kwargs)
        output_text = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        transcript_parts.append(output_text)

        answer_text, closed = extract_answer(output_text)
        if answer_text:
            final_answer = answer_text
            answer_tag_closed = closed
            break

        query = extract_latest_search(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if not query:
            break

        search_results, docs = search(query, args.retriever_url, args.topk)
        gold_in_top_docs = answer_in_docs(example["golden_answers"], docs)
        searches.append(
            {
                "turn": turn + 1,
                "query": query,
                "gold_in_top_docs": gold_in_top_docs,
                "docs": docs,
            }
        )
        search_text = (
            f"\n\n{output_text}<information>{search_results}</information>\n\n"
        )
        transcript_parts.append(search_text)
        prompt += search_text

    is_em = exact_match(final_answer, example["golden_answers"])
    failure_type = classify_failure(is_em, answer_tag_closed, searches)
    answer_metrics = max_answer_metrics(final_answer, example["golden_answers"])
    query_quality = classify_query_quality(searches)
    return {
        **example,
        "prediction": final_answer,
        "exact_match": is_em,
        "relaxed_match": answer_metrics["relaxed_match"],
        "token_f1": answer_metrics["token_f1"],
        "answer_metrics": answer_metrics,
        "query_quality": query_quality,
        "failure_type": failure_type,
        "answer_tag_closed": answer_tag_closed,
        "num_searches": len(searches),
        "searches": searches,
        "latency_sec": round(time.time() - start_time, 3),
        "transcript": "".join(transcript_parts),
    }


def write_outputs(
    results: list[dict], out_dir: Path, metadata: dict | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    csv_path = out_dir / "results.csv"
    summary_path = out_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "id",
                "question",
                "golden_answers",
                "prediction",
                "exact_match",
                "relaxed_match",
                "token_f1",
                "failure_type",
                "num_searches",
                "bad_query_count",
                "bad_query_reasons",
                "search_queries",
                "top3_docs",
                "latency_sec",
            ],
        )
        writer.writeheader()
        for result in results:
            answer_metrics = result.get("answer_metrics") or max_answer_metrics(
                result.get("prediction", ""), result.get("golden_answers", [])
            )
            query_quality = result.get("query_quality") or classify_query_quality(
                result.get("searches", [])
            )
            search_queries = [search["query"] for search in result["searches"]]
            top3_docs = [
                [
                    {
                        "rank": doc["rank"],
                        "id": doc["id"],
                        "title": doc["title"],
                        "score": doc["score"],
                    }
                    for doc in search["docs"][:3]
                ]
                for search in result["searches"]
            ]
            writer.writerow(
                {
                    "dataset": result["dataset"],
                    "id": result["id"],
                    "question": result["question"],
                    "golden_answers": json.dumps(
                        result["golden_answers"], ensure_ascii=False
                    ),
                    "prediction": result["prediction"],
                    "exact_match": result["exact_match"],
                    "relaxed_match": answer_metrics["relaxed_match"],
                    "token_f1": answer_metrics["token_f1"],
                    "failure_type": result["failure_type"],
                    "num_searches": result["num_searches"],
                    "bad_query_count": query_quality["bad_query_count"],
                    "bad_query_reasons": json.dumps(
                        query_quality["bad_query_reasons"], ensure_ascii=False
                    ),
                    "search_queries": json.dumps(search_queries, ensure_ascii=False),
                    "top3_docs": json.dumps(top3_docs, ensure_ascii=False),
                    "latency_sec": result["latency_sec"],
                }
            )

    summary, _ = summarize_results(results, metadata)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def reset_incremental_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("results.jsonl", "results.csv"):
        path = out_dir / filename
        if path.exists():
            path.unlink()


def append_result_output(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def write_summary_output(
    results: list[dict], out_dir: Path, metadata: dict | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, _ = summarize_results(results, metadata)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_eval(
    examples: list[dict],
    tokenizer,
    model,
    device,
    stopping_criteria,
    args: argparse.Namespace,
    metadata: dict,
    finalize_outputs: bool = True,
) -> list[dict]:
    results = []
    reset_incremental_outputs(args.out_dir)
    for idx, example in enumerate(examples, start=1):
        print(
            f"[{idx}/{len(examples)}] {example['dataset']} {example['id']} "
            f"{example['question']}"
        )
        result = run_one(example, tokenizer, model, device, stopping_criteria, args)
        print(
            f"  em={result['exact_match']} searches={result['num_searches']} "
            f"failure={result['failure_type']} pred={result['prediction']!r}"
        )
        results.append(result)
        append_result_output(result, args.out_dir)
        write_summary_output(results, args.out_dir, metadata)

    if finalize_outputs:
        write_outputs(results, args.out_dir, metadata)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
    )
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--per-dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--examples-file", type=Path, default=None)
    parser.add_argument("--write-examples-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_runs/stage2_small"))
    args = parser.parse_args()

    if args.examples_file:
        examples = load_examples_file(args.examples_file)
    else:
        examples = load_examples(args.per_dataset, args.seed)
    if args.write_examples_file:
        write_examples_file(examples, args.write_examples_file)

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(SEARCH_STOPS, tokenizer)]
    )

    metadata = {
        "model_id": args.model_id,
        "retriever_url": args.retriever_url,
        "topk": args.topk,
        "per_dataset": args.per_dataset,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "examples_file": str(args.examples_file) if args.examples_file else None,
        "write_examples_file": str(args.write_examples_file)
        if args.write_examples_file
        else None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    results = run_eval(
        examples,
        tokenizer,
        model,
        device,
        stopping_criteria,
        args,
        metadata,
        finalize_outputs=False,
    )
    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    write_outputs(results, args.out_dir, metadata)
    print(f"wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
