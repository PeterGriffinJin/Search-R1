from __future__ import annotations

import csv
import json
import re
import string
from collections import Counter
from pathlib import Path


YES_NO = {"yes", "no"}
BAD_PLACEHOLDER_QUERIES = {
    "query",
    "search",
    "search query",
    "what is the answer",
    "find answer",
}
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


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def answer_tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def strict_exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(answer) for answer in answers)


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = answer_tokens(prediction)
    answer_tokens_ = answer_tokens(answer)
    if not pred_tokens or not answer_tokens_:
        return float(pred_tokens == answer_tokens_)

    common = Counter(pred_tokens) & Counter(answer_tokens_)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens_)
    return 2 * precision * recall / (precision + recall)


def _is_numeric_or_date_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}", token)) or token in MONTHS


def _allow_single_token_containment(tokens: list[str]) -> bool:
    if len(tokens) != 1:
        return True
    token = tokens[0]
    if token in YES_NO:
        return False
    return _is_numeric_or_date_token(token)


def _contains_token_span(container: list[str], contained: list[str]) -> bool:
    if not container or not contained or len(contained) > len(container):
        return False
    if len(contained) == 1 and not _allow_single_token_containment(contained):
        return False
    for start in range(0, len(container) - len(contained) + 1):
        if container[start : start + len(contained)] == contained:
            return True
    return False


def max_answer_metrics(prediction: str, answers: list[str]) -> dict:
    answers = list(answers or [])
    pred_tokens = answer_tokens(prediction)
    strict = strict_exact_match(prediction, answers)

    best_answer = ""
    best_f1 = 0.0
    prediction_contains_gold = False
    gold_contains_prediction = False
    for answer in answers:
        current_f1 = token_f1(prediction, answer)
        if current_f1 >= best_f1:
            best_f1 = current_f1
            best_answer = answer

        gold_tokens = answer_tokens(answer)
        prediction_contains_gold = prediction_contains_gold or _contains_token_span(
            pred_tokens, gold_tokens
        )
        gold_contains_prediction = gold_contains_prediction or _contains_token_span(
            gold_tokens, pred_tokens
        )

    f1_relaxed = best_f1 >= 0.8
    relaxed = strict or prediction_contains_gold or gold_contains_prediction or f1_relaxed
    return {
        "strict_em": strict,
        "relaxed_match": relaxed,
        "token_f1": round(best_f1, 6),
        "best_gold": best_answer,
        "f1_relaxed": f1_relaxed,
        "prediction_contains_gold": prediction_contains_gold,
        "gold_contains_prediction": gold_contains_prediction,
    }


def classify_query_quality(searches: list[dict]) -> dict:
    reason_order = ["empty", "too_short", "too_long", "tag_leak", "prompt_leak"]
    reason_seen = {reason: False for reason in reason_order}
    bad_query_count = 0
    query_rows = []

    for search in searches or []:
        query = str(search.get("query", ""))
        stripped = " ".join(query.split())
        lowered = stripped.lower()
        tokens = lowered.split()
        reasons = []

        if not stripped:
            reasons.append("empty")
        if lowered in BAD_PLACEHOLDER_QUERIES or len(tokens) <= 1:
            reasons.append("too_short")
        if len(tokens) > 24 or len(stripped) > 180:
            reasons.append("too_long")
        if re.search(r"</?[a-zA-Z][^>]*>", stripped) or "doc 1(" in lowered:
            reasons.append("tag_leak")
        if (
            "answer the given question" in lowered
            or "you must conduct reasoning" in lowered
        ):
            reasons.append("prompt_leak")

        if reasons:
            bad_query_count += 1
            for reason in reasons:
                reason_seen[reason] = True

        query_rows.append({"query": query, "reasons": reasons})

    return {
        "num_queries": len(searches or []),
        "bad_query_count": bad_query_count,
        "bad_query_reasons": [
            reason for reason in reason_order if reason_seen[reason]
        ],
        "queries": query_rows,
    }


def gold_in_any_top_docs(result: dict) -> bool:
    return any(search.get("gold_in_top_docs") for search in result.get("searches", []))


def answer_tag_closed(result: dict) -> bool:
    if "answer_tag_closed" in result:
        return bool(result["answer_tag_closed"])
    if result.get("failure_type") == "no_answer_tag":
        return False
    return bool(str(result.get("prediction", "")).strip())


def search_behavior_stats(results: list[dict]) -> dict:
    total = len(results)
    total_searches = sum(int(result.get("num_searches", 0)) for result in results)
    multi_search_rows = [
        result for result in results if int(result.get("num_searches", 0)) >= 2
    ]
    multi_search = len(multi_search_rows)
    closed_after_multi = sum(
        1 for result in multi_search_rows if answer_tag_closed(result)
    )
    strict_after_multi = sum(
        1 for result in multi_search_rows if result["answer_metrics"]["strict_em"]
    )
    no_answer = sum(1 for result in results if result.get("failure_type") == "no_answer_tag")
    return {
        "avg_num_searches": total_searches / total if total else 0.0,
        "multi_search": multi_search,
        "multi_search_rate": multi_search / total if total else 0.0,
        "closed_after_multi_search": closed_after_multi,
        "closed_answer_after_2plus_searches": closed_after_multi,
        "closed_after_multi_search_rate": closed_after_multi / multi_search
        if multi_search
        else 0.0,
        "closed_answer_after_2plus_search_rate": closed_after_multi / multi_search
        if multi_search
        else 0.0,
        "strict_after_multi_search": strict_after_multi,
        "strict_after_multi_search_rate": strict_after_multi / multi_search
        if multi_search
        else 0.0,
        "no_answer_tag": no_answer,
        "no_answer_tag_rate": no_answer / total if total else 0.0,
    }


def enrich_result(result: dict) -> dict:
    enriched = dict(result)
    answer_metrics = max_answer_metrics(
        enriched.get("prediction", ""), enriched.get("golden_answers", [])
    )
    query_quality = classify_query_quality(enriched.get("searches", []))
    enriched["answer_metrics"] = answer_metrics
    enriched["query_quality"] = query_quality
    enriched["gold_in_any_top_docs"] = gold_in_any_top_docs(enriched)
    enriched["relaxed_match"] = answer_metrics["relaxed_match"]
    enriched["token_f1"] = answer_metrics["token_f1"]
    return enriched


def summarize_results(results: list[dict], metadata: dict | None = None) -> tuple[dict, list[dict]]:
    enriched_results = [enrich_result(result) for result in results]
    by_dataset: dict[str, dict] = {}
    by_failure: dict[str, int] = {}
    query_reason_counts: dict[str, int] = {}

    for result in enriched_results:
        dataset = result.get("dataset", "unknown")
        stats = by_dataset.setdefault(
            dataset,
            {
                "total": 0,
                "strict_em": 0,
                "relaxed_em": 0,
                "token_f1_sum": 0.0,
                "search_count_sum": 0,
                "multi_search": 0,
                "closed_after_multi_search": 0,
                "strict_after_multi_search": 0,
                "no_answer_tag": 0,
            },
        )
        stats["total"] += 1
        strict = int(result["answer_metrics"]["strict_em"])
        num_searches = int(result.get("num_searches", 0))
        is_multi_search = num_searches >= 2
        stats["strict_em"] += strict
        stats["relaxed_em"] += int(result["answer_metrics"]["relaxed_match"])
        stats["token_f1_sum"] += result["answer_metrics"]["token_f1"]
        stats["search_count_sum"] += num_searches
        stats["multi_search"] += int(is_multi_search)
        stats["closed_after_multi_search"] += int(
            is_multi_search and answer_tag_closed(result)
        )
        stats["strict_after_multi_search"] += int(is_multi_search and strict)
        stats["no_answer_tag"] += int(result.get("failure_type") == "no_answer_tag")

        failure_type = result.get("failure_type", "unknown")
        by_failure[failure_type] = by_failure.get(failure_type, 0) + 1
        for reason in result["query_quality"]["bad_query_reasons"]:
            query_reason_counts[reason] = query_reason_counts.get(reason, 0) + 1

    for stats in by_dataset.values():
        total = stats["total"]
        stats["strict_em_rate"] = stats["strict_em"] / total if total else 0.0
        stats["relaxed_em_rate"] = stats["relaxed_em"] / total if total else 0.0
        stats["avg_token_f1"] = stats["token_f1_sum"] / total if total else 0.0
        stats["avg_num_searches"] = stats["search_count_sum"] / total if total else 0.0
        stats["multi_search_rate"] = stats["multi_search"] / total if total else 0.0
        stats["closed_after_multi_search_rate"] = (
            stats["closed_after_multi_search"] / stats["multi_search"]
            if stats["multi_search"]
            else 0.0
        )
        stats["strict_after_multi_search_rate"] = (
            stats["strict_after_multi_search"] / stats["multi_search"]
            if stats["multi_search"]
            else 0.0
        )
        stats["no_answer_tag_rate"] = stats["no_answer_tag"] / total if total else 0.0
        del stats["token_f1_sum"]
        del stats["search_count_sum"]

    total = len(enriched_results)
    strict_em = sum(int(r["answer_metrics"]["strict_em"]) for r in enriched_results)
    relaxed_em = sum(int(r["answer_metrics"]["relaxed_match"]) for r in enriched_results)
    total_queries = sum(r["query_quality"]["num_queries"] for r in enriched_results)
    bad_queries = sum(r["query_quality"]["bad_query_count"] for r in enriched_results)
    summary = {
        "metadata": metadata or {},
        "total": total,
        "strict_em": strict_em,
        "strict_em_rate": strict_em / total if total else 0.0,
        "relaxed_em": relaxed_em,
        "relaxed_em_rate": relaxed_em / total if total else 0.0,
        "avg_token_f1": sum(r["answer_metrics"]["token_f1"] for r in enriched_results)
        / total
        if total
        else 0.0,
        "by_dataset": by_dataset,
        "by_failure_type": by_failure,
        "query_quality": {
            "total_queries": total_queries,
            "bad_queries": bad_queries,
            "bad_query_rate": bad_queries / total_queries if total_queries else 0.0,
            "bad_query_reason_counts": query_reason_counts,
        },
        "search_behavior": search_behavior_stats(enriched_results),
        "avg_num_searches": sum(result.get("num_searches", 0) for result in results)
        / total
        if total
        else 0.0,
        "avg_latency_sec": sum(result.get("latency_sec", 0.0) for result in results)
        / total
        if total
        else 0.0,
    }
    return summary, enriched_results


def top_doc_brief(searches: list[dict], limit: int = 3) -> list[list[dict]]:
    rows = []
    for search in searches or []:
        docs = []
        for doc in search.get("docs", [])[:limit]:
            docs.append(
                {
                    "rank": doc.get("rank"),
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "score": doc.get("score"),
                }
            )
        rows.append(docs)
    return rows


def recommended_failure_action(result: dict) -> str:
    failure_type = result.get("failure_type", "")
    if failure_type == "no_answer_tag":
        return "format_policy"
    if failure_type == "no_search":
        return "search_decision"
    if failure_type == "retrieval_miss":
        if result["query_quality"]["bad_query_count"]:
            return "query_generation"
        return "retrieval_or_query"
    if failure_type == "evidence_use_error":
        if result["answer_metrics"]["relaxed_match"]:
            return "metric_alias_or_span"
        return "evidence_reading"
    return "inspect"


def failure_rows(enriched_results: list[dict]) -> list[dict]:
    rows = []
    for result in enriched_results:
        if result["answer_metrics"]["strict_em"]:
            continue
        searches = result.get("searches", [])
        rows.append(
            {
                "dataset": result.get("dataset"),
                "id": result.get("id"),
                "question": result.get("question"),
                "golden_answers": json.dumps(
                    result.get("golden_answers", []), ensure_ascii=False
                ),
                "prediction": result.get("prediction", ""),
                "failure_type": result.get("failure_type"),
                "recommended_action": recommended_failure_action(result),
                "relaxed_match": result["answer_metrics"]["relaxed_match"],
                "token_f1": result["answer_metrics"]["token_f1"],
                "prediction_contains_gold": result["answer_metrics"][
                    "prediction_contains_gold"
                ],
                "gold_contains_prediction": result["answer_metrics"][
                    "gold_contains_prediction"
                ],
                "gold_in_any_top_docs": result.get("gold_in_any_top_docs", False),
                "num_searches": result.get("num_searches", len(searches)),
                "bad_query_count": result["query_quality"]["bad_query_count"],
                "bad_query_reasons": json.dumps(
                    result["query_quality"]["bad_query_reasons"],
                    ensure_ascii=False,
                ),
                "search_queries": json.dumps(
                    [search.get("query", "") for search in searches],
                    ensure_ascii=False,
                ),
                "top3_docs": json.dumps(top_doc_brief(searches), ensure_ascii=False),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
