from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

try:
    from scripts.stage2_metrics import max_answer_metrics, normalize_answer
except ImportError:
    from stage2_metrics import max_answer_metrics, normalize_answer


MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


TRAINING_ACTIONS = {
    "answer_span_boundary_or_format_mismatch": (
        "Reward minimal answer spans matching the requested answer type. Keep relaxed "
        "date/year normalization, but penalize vague nearby spans when evidence contains "
        "the canonical answer."
    ),
    "high_rank_distractor_copied": (
        "Add hard-negative evidence reading data: top-1/top-2 documents can be related but "
        "wrong, so reward support in the exact answer sentence rather than title prominence."
    ),
    "predicate_or_role_confusion": (
        "Add slot-filling examples and reward role-correct extraction, e.g. actor vs role, "
        "campus vs neighboring place, assassin vs victim."
    ),
    "incomplete_comparison_reasoning": (
        "For comparison questions, reward extracting both candidate attributes before "
        "choosing the final entity."
    ),
    "relation_unfaithful_query_drift": (
        "Make retrieval rewards relation-aware and penalize query drift that invents a new "
        "constraint, such as changing 'orbit what' into 'orbit the moon'."
    ),
    "no_answer_despite_gold_evidence": (
        "Add strong finalization reward: when retrieved evidence contains the gold/alias "
        "span, reward exactly one closed <answer> and penalize max-turn termination."
    ),
    "no_answer_loop_bad_or_irrelevant_retrieval": (
        "Reward query revision when top docs are unhelpful, and penalize duplicate searches "
        "with the same query/results."
    ),
    "missing_multihop_bridge_query": (
        "Reward decomposed Hotpot-style search plans: first identify the bridge entity, then "
        "query the bridge property before answering."
    ),
    "ambiguous_wrong_domain_query": (
        "Add hard-negative retrieval data where lexical matches are in the wrong domain, and "
        "reward domain words plus expected answer type in the query."
    ),
    "comparison_or_answer_type_drift": (
        "Reward split queries for A-or-B comparisons and answer-type consistency, e.g. who "
        "questions should not return dates."
    ),
    "no_search_answered_from_parametric_memory": (
        "Add search-decision reward for examples requiring external evidence, and penalize "
        "unsupported parametric answers when the prompt expects tool use."
    ),
    "answerable_repeat_search_loop": (
        "Add terminal-answer reward: once top docs contain a gold/alias span, reward closing "
        "<answer> and penalize repeating the same query."
    ),
    "unproductive_repeat_search_loop": (
        "Add query-diversity and reformulation reward: repeated low-yield queries should be "
        "penalized unless they retrieve new evidence."
    ),
    "placeholder_query_loop_no_final": (
        "Add hard query-format penalty and filter placeholder-query rollouts from positive "
        "updates."
    ),
    "unclosed_answer_tag": (
        "Add format-completion reward for balanced <answer>...</answer> tags and reserve "
        "enough generation budget after opening the final tag."
    ),
    "no_answer_other": (
        "Add a final-answer fallback reward: after search, the policy should produce one "
        "closed answer instead of ending without a tag."
    ),
    "numeric_or_date_specificity": (
        "Add exact numeric/date reward and build date QA contrastive examples where a full "
        "date, year-only answer, and distractor date appear in the same evidence."
    ),
    "near_alias_or_granularity": (
        "Add alias/span-normalization data and reward shortest correct canonical spans, not "
        "nearby broader/narrower descriptions."
    ),
    "query_intent_drift_inside_evidence_case": (
        "Add query-fidelity reward: generated search queries must preserve the original "
        "relation/slot, especially who/what/where questions."
    ),
    "comparison_or_multihop_logic_error": (
        "Add multi-hop decomposition data: require retrieving/comparing both entities and "
        "reward answers supported by both hops."
    ),
    "distractor_span_selection": (
        "Add evidence-span selection reward using top-doc snippets: penalize copying salient "
        "but role-mismatched entities from retrieved docs."
    ),
    "evidence_reading_other": (
        "Add supervised answer-extraction rollouts over retrieved top-3 docs before further "
        "GRPO, then use strict EM/F1 as outcome reward."
    ),
    "hotpot_single_query_topk_miss": (
        "Add Hotpot-style bridge-query reward: after the first search, reward a second query "
        "that targets the missing bridge/final entity."
    ),
    "direct_topk_miss": (
        "Add retrieval auxiliary reward for gold-in-top-k and train paraphrase/entity-expanded "
        "queries."
    ),
    "partial_relation_wrong_answer": (
        "Add question-type reward: the answer type should match the requested slot, e.g. who "
        "questions should not return dates."
    ),
    "retrieval_other": (
        "Add hard-negative retrieval data from the observed top-3 distractors and reward "
        "queries that move gold evidence into top-k."
    ),
}


MANUAL_SUBTYPES = {
    # Evidence-use errors, manually reviewed 2026-04-25 on the fixed 50-sample set.
    "nq:test_1278:evidence_use_error": "answer_span_boundary_or_format_mismatch",
    "nq:test_603:evidence_use_error": "answer_span_boundary_or_format_mismatch",
    "hotpotqa:dev_5402:evidence_use_error": "answer_span_boundary_or_format_mismatch",
    "nq:test_3486:evidence_use_error": "high_rank_distractor_copied",
    "nq:test_3302:evidence_use_error": "high_rank_distractor_copied",
    "nq:test_3037:evidence_use_error": "high_rank_distractor_copied",
    "nq:test_1996:evidence_use_error": "predicate_or_role_confusion",
    "hotpotqa:dev_5697:evidence_use_error": "predicate_or_role_confusion",
    "hotpotqa:dev_2906:evidence_use_error": "predicate_or_role_confusion",
    "hotpotqa:dev_131:evidence_use_error": "incomplete_comparison_reasoning",
    "hotpotqa:dev_4746:evidence_use_error": "relation_unfaithful_query_drift",
    # Retrieval / no-answer errors, manually reviewed 2026-04-25 on the fixed set.
    "nq:test_1549:no_answer_tag": "no_answer_despite_gold_evidence",
    "nq:test_836:no_answer_tag": "no_answer_despite_gold_evidence",
    "nq:test_2055:no_answer_tag": "no_answer_despite_gold_evidence",
    "nq:test_3303:no_answer_tag": "no_answer_despite_gold_evidence",
    "nq:test_113:no_answer_tag": "no_answer_despite_gold_evidence",
    "nq:test_1485:no_answer_tag": "no_answer_despite_gold_evidence",
    "nq:test_2315:no_answer_tag": "no_answer_despite_gold_evidence",
    "hotpotqa:dev_63:no_answer_tag": "no_answer_despite_gold_evidence",
    "hotpotqa:dev_5763:no_answer_tag": "no_answer_despite_gold_evidence",
    "hotpotqa:dev_5839:no_answer_tag": "no_answer_despite_gold_evidence",
    "nq:test_1358:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "nq:test_523:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "nq:test_658:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "hotpotqa:dev_5274:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "hotpotqa:dev_3478:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "hotpotqa:dev_6736:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "hotpotqa:dev_1740:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "hotpotqa:dev_1395:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "hotpotqa:dev_302:no_answer_tag": "no_answer_loop_bad_or_irrelevant_retrieval",
    "nq:test_2777:no_answer_tag": "unclosed_answer_tag",
    "nq:test_3453:no_answer_tag": "unclosed_answer_tag",
    "hotpotqa:dev_2052:retrieval_miss": "missing_multihop_bridge_query",
    "hotpotqa:dev_2876:retrieval_miss": "missing_multihop_bridge_query",
    "hotpotqa:dev_7153:retrieval_miss": "missing_multihop_bridge_query",
    "hotpotqa:dev_236:retrieval_miss": "missing_multihop_bridge_query",
    "hotpotqa:dev_3466:retrieval_miss": "missing_multihop_bridge_query",
    "hotpotqa:dev_2275:retrieval_miss": "missing_multihop_bridge_query",
    "nq:test_205:retrieval_miss": "ambiguous_wrong_domain_query",
    "nq:test_1231:retrieval_miss": "ambiguous_wrong_domain_query",
    "hotpotqa:dev_3361:retrieval_miss": "comparison_or_answer_type_drift",
    "nq:test_2793:retrieval_miss": "comparison_or_answer_type_drift",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_summary(path: Path) -> dict:
    summary_path = path.parent / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def compact_query(query: str) -> str:
    return " ".join(str(query).split())


def queries(result: dict) -> list[str]:
    return [compact_query(search.get("query", "")) for search in result.get("searches", [])]


def repeated_query(result: dict) -> bool:
    qs = [query.lower() for query in queries(result) if query]
    return len(qs) >= 2 and len(set(qs)) <= max(1, len(qs) // 2)


def top3_titles(result: dict) -> list[list[str]]:
    rows = []
    for search in result.get("searches", []):
        rows.append([doc.get("title", "") for doc in search.get("docs", [])[:3]])
    return rows


def top_docs_text(result: dict) -> str:
    parts = []
    for search in result.get("searches", []):
        for doc in search.get("docs", [])[:3]:
            parts.append(str(doc.get("title", "")))
            parts.append(str(doc.get("contents", "")))
    return "\n".join(parts)


def normalize_for_substring(text: str) -> str:
    return normalize_answer(text).replace(" ", "")


def prediction_in_docs(result: dict) -> bool:
    prediction = str(result.get("prediction", "")).strip()
    if not prediction:
        return False
    needle = normalize_for_substring(prediction)
    if len(needle) < 3:
        return False
    return needle in normalize_for_substring(top_docs_text(result))


def answer_metric(result: dict) -> dict:
    existing = result.get("answer_metrics")
    if existing:
        return existing
    return max_answer_metrics(result.get("prediction", ""), result.get("golden_answers", []))


def gold_in_top_docs(result: dict) -> bool:
    return bool(
        result.get("gold_in_any_top_docs")
        or any(search.get("gold_in_top_docs") for search in result.get("searches", []))
    )


def bad_query_count(result: dict) -> int:
    quality = result.get("query_quality") or {}
    return int(quality.get("bad_query_count") or 0)


def contains_numeric_or_date(*texts: str) -> bool:
    for text in texts:
        lowered = str(text).lower()
        if re.search(r"\b\d{1,4}\b", lowered):
            return True
        if any(month in lowered for month in MONTHS):
            return True
    return False


def raw_answer_substring_overlap(result: dict) -> bool:
    prediction = str(result.get("prediction", "")).lower()
    if not prediction:
        return False
    for answer in result.get("golden_answers", []):
        answer_text = str(answer).lower()
        if len(answer_text) >= 4 and answer_text in prediction:
            return True
        if len(prediction) >= 4 and prediction in answer_text:
            return True
    return False


def likely_multihop_or_comparison(result: dict) -> bool:
    question = str(result.get("question", "")).lower()
    if result.get("dataset") == "hotpotqa":
        return True
    return " or " in question or " both " in question


def likely_query_intent_drift(result: dict) -> bool:
    question = str(result.get("question", "")).lower()
    joined_queries = " ".join(queries(result)).lower()
    if "orbit what" in question and "orbit the moon" in joined_queries:
        return True
    if "real name" in question and normalize_answer(result.get("prediction", "")) in {
        "daya jethalal gada"
    }:
        return True
    return False


def classify_no_answer(result: dict) -> tuple[str, list[str]]:
    transcript = str(result.get("transcript", ""))
    num_searches = int(result.get("num_searches") or len(result.get("searches", [])))
    tags = []
    if gold_in_top_docs(result):
        tags.append("gold_seen")
    if repeated_query(result):
        tags.append("repeated_query")
    if bad_query_count(result):
        tags.append("bad_query")
    if num_searches >= 5:
        tags.append("max_turn_exhausted")

    last_answer_open = transcript.rfind("<answer>")
    last_answer_close = transcript.rfind("</answer>")
    if last_answer_open > last_answer_close:
        return "unclosed_answer_tag", tags
    if num_searches and bad_query_count(result) >= num_searches:
        return "placeholder_query_loop_no_final", tags
    if num_searches >= 5 and repeated_query(result) and gold_in_top_docs(result):
        return "answerable_repeat_search_loop", tags
    if num_searches >= 5 and repeated_query(result):
        return "unproductive_repeat_search_loop", tags
    return "no_answer_other", tags


def classify_evidence_use(result: dict) -> tuple[str, list[str]]:
    metric = answer_metric(result)
    prediction = str(result.get("prediction", ""))
    best_gold = str(metric.get("best_gold") or (result.get("golden_answers") or [""])[0])
    tags = []
    if gold_in_top_docs(result):
        tags.append("gold_seen")
    if prediction_in_docs(result):
        tags.append("prediction_in_top_docs")

    if contains_numeric_or_date(prediction, best_gold):
        return "numeric_or_date_specificity", tags
    if (
        float(metric.get("token_f1") or 0.0) > 0.0
        or metric.get("relaxed_match")
        or raw_answer_substring_overlap(result)
    ):
        return "near_alias_or_granularity", tags
    if likely_query_intent_drift(result):
        return "query_intent_drift_inside_evidence_case", tags
    if likely_multihop_or_comparison(result):
        return "comparison_or_multihop_logic_error", tags
    if prediction_in_docs(result):
        return "distractor_span_selection", tags
    return "evidence_reading_other", tags


def classify_retrieval(result: dict) -> tuple[str, list[str]]:
    metric = answer_metric(result)
    tags = []
    if bad_query_count(result):
        tags.append("bad_query")
    if repeated_query(result):
        tags.append("repeated_query")
    if int(result.get("num_searches") or len(result.get("searches", []))) <= 1:
        tags.append("single_search")

    if result.get("dataset") == "hotpotqa" and "single_search" in tags:
        return "hotpot_single_query_topk_miss", tags
    if float(metric.get("token_f1") or 0.0) > 0.0:
        return "partial_relation_wrong_answer", tags
    if result.get("dataset") == "nq":
        return "direct_topk_miss", tags
    return "retrieval_other", tags


def classify_result(result: dict) -> tuple[str, list[str]]:
    manual_key = f"{result.get('dataset')}:{result.get('id')}:{result.get('failure_type')}"
    if manual_key in MANUAL_SUBTYPES:
        subtype = MANUAL_SUBTYPES[manual_key]
        tags = ["manual_review"]
        if gold_in_top_docs(result):
            tags.append("gold_seen")
        if repeated_query(result):
            tags.append("repeated_query")
        if bad_query_count(result):
            tags.append("bad_query")
        return subtype, tags

    failure_type = result.get("failure_type")
    if failure_type == "no_answer_tag":
        return classify_no_answer(result)
    if failure_type == "evidence_use_error":
        return classify_evidence_use(result)
    if failure_type == "retrieval_miss":
        return classify_retrieval(result)
    if failure_type == "no_search":
        return "no_search_answered_from_parametric_memory", []
    return "correct_or_unclassified", []


def row_for_result(result: dict) -> dict:
    metric = answer_metric(result)
    subtype, tags = classify_result(result)
    return {
        "dataset": result.get("dataset", ""),
        "id": result.get("id", ""),
        "question": result.get("question", ""),
        "golden_answers": json.dumps(result.get("golden_answers", []), ensure_ascii=False),
        "prediction": result.get("prediction", ""),
        "failure_type": result.get("failure_type", ""),
        "subtype": subtype,
        "tags": ",".join(tags),
        "training_action": TRAINING_ACTIONS.get(subtype, "Inspect manually."),
        "strict_em": bool(metric.get("strict_em")),
        "relaxed_match": bool(metric.get("relaxed_match")),
        "token_f1": float(metric.get("token_f1") or 0.0),
        "gold_in_top_docs": gold_in_top_docs(result),
        "num_searches": int(result.get("num_searches") or len(result.get("searches", []))),
        "bad_query_count": bad_query_count(result),
        "queries": json.dumps(queries(result), ensure_ascii=False),
        "top3_titles": json.dumps(top3_titles(result), ensure_ascii=False),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_count(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.1%})" if total else "0/0 (0.0%)"


def representative_rows(rows: list[dict], subtype: str, limit: int = 3) -> list[dict]:
    return [row for row in rows if row["subtype"] == subtype][:limit]


def next_training_changes(
    summary: dict,
    rows: list[dict],
    subtype_counts: Counter,
    failure_counts: Counter,
) -> list[str]:
    total = int(summary.get("total") or len(rows))
    no_answer = int(failure_counts.get("no_answer_tag", 0))
    retrieval = int(failure_counts.get("retrieval_miss", 0))
    evidence = int(failure_counts.get("evidence_use_error", 0))
    top_subtypes = ", ".join(
        f"{subtype}={count}" for subtype, count in subtype_counts.most_common(3)
    )

    changes = []
    if no_answer:
        changes.append(
            "Improve answer closure only where needed: "
            f"{format_count(no_answer, total)} runs lack a closed final answer; "
            "target max-turn loops without increasing search count globally."
        )
    else:
        changes.append(
            "Keep format pressure light: no evaluated runs lack a closed final answer, "
            "so prioritize correctness over more termination reward."
        )

    if retrieval >= evidence:
        changes.append(
            "Prioritize retrieval/query reward: "
            f"{format_count(retrieval, total)} runs are retrieval_miss; reward "
            "gold-in-top-k, query reformulation, and HotpotQA bridge queries."
        )
        changes.append(
            "Then improve evidence extraction: "
            f"{format_count(evidence, total)} runs are evidence_use_error; reward "
            "exact canonical spans and penalize salient distractors."
        )
    else:
        changes.append(
            "Prioritize evidence extraction: "
            f"{format_count(evidence, total)} runs are evidence_use_error; reward "
            "exact canonical spans and penalize salient distractors."
        )
        changes.append(
            "Then improve retrieval/query behavior: "
            f"{format_count(retrieval, total)} runs are retrieval_miss; reward "
            "gold-in-top-k, query reformulation, and HotpotQA bridge queries."
        )

    if top_subtypes:
        changes.append(f"Build data/rewards around the top observed subtypes: {top_subtypes}.")
    changes.append(
        "Keep the same 50-sample manifest for every checkpoint, then add a second "
        "held-out 100-sample set only after step10-style gains repeat."
    )
    return changes


def markdown_report(summary: dict, rows: list[dict], output_csv: Path) -> str:
    failed_rows = [row for row in rows if row["failure_type"] != "correct"]
    total = len(rows)
    failures = len(failed_rows)
    subtype_counts = Counter(row["subtype"] for row in failed_rows)
    failure_counts = Counter(row["failure_type"] for row in rows)
    metadata = summary.get("metadata", {})
    strict_em = int(summary.get("strict_em", summary.get("em", 0)))
    total_count = int(summary.get("total", total))

    lines = [
        "# Failure Taxonomy",
        "",
        f"- Eval time: {metadata.get('started_at', 'unknown')} -> {metadata.get('finished_at', 'unknown')}",
        f"- Model: `{metadata.get('model_id', 'unknown')}`",
        f"- Fixed set: `{metadata.get('examples_file', 'unknown')}`",
        f"- Config: topk={metadata.get('topk', 'unknown')}, max_turns={metadata.get('max_turns', 'unknown')}, temperature={metadata.get('temperature', 'unknown')}",
        f"- Strict EM: {format_count(strict_em, total_count)}",
        f"- Failures analyzed: {format_count(failures, total)}",
        f"- CSV: `{output_csv}`",
        "",
        "## Original Failure Types",
        "",
    ]
    for failure_type, count in sorted(failure_counts.items()):
        lines.append(f"- {failure_type}: {count}")

    lines.extend(["", "## Training-Oriented Subtypes", ""])
    for subtype, count in subtype_counts.most_common():
        lines.append(f"### {subtype}: {count}")
        lines.append("")
        lines.append(f"- Training item: {TRAINING_ACTIONS.get(subtype, 'Inspect manually.')}")
        for row in representative_rows(failed_rows, subtype):
            lines.append(
                "- Example {dataset}:{id}: Q={question} | gold={gold} | pred={pred} | queries={queries}".format(
                    dataset=row["dataset"],
                    id=row["id"],
                    question=row["question"],
                    gold=row["golden_answers"],
                    pred=row["prediction"] or "<empty>",
                    queries=row["queries"],
                )
            )
        lines.append("")

    lines.extend(["## Next Training Changes", ""])
    for index, change in enumerate(
        next_training_changes(summary, rows, subtype_counts, failure_counts), start=1
    ):
        lines.append(f"{index}. {change}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "eval_runs/stage3_compare_50/qwen2_5_3b_continue_step10/results.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    results = load_jsonl(args.results)
    summary = load_summary(args.results)
    rows = [row_for_result(result) for result in results]
    failed_rows = [row for row in rows if row["failure_type"] != "correct"]

    output_dir = args.output_dir or args.results.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "failure_taxonomy.csv"
    json_path = output_dir / "failure_taxonomy.json"
    md_path = output_dir / "failure_taxonomy.md"

    write_csv(csv_path, failed_rows)
    subtype_counts = Counter(row["subtype"] for row in failed_rows)
    failure_counts = Counter(row["failure_type"] for row in rows)
    json_path.write_text(
        json.dumps(
            {
                "results": str(args.results),
                "summary": summary,
                "failure_type_counts": dict(failure_counts),
                "subtype_counts": dict(subtype_counts),
                "rows": failed_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path.write_text(markdown_report(summary, rows, csv_path), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("subtypes:", dict(subtype_counts))


if __name__ == "__main__":
    main()
