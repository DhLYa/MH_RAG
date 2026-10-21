"""
Scores the pipeline against the labelled questions in evaluation/questions.json.

Retrieval is scored by where the first chunk from an expected source page lands, both
before reranking (top RETRIEVAL_K by vector similarity) and after (top RERANK_K). This
needs only Voyage calls.

--generate also asks the LLM each question and checks the answer contains the expected
terms, or that it says it doesn't know for out-of-scope questions. That is one LLM call
per question, spaced by --delay to stay under the free-tier rate limit.

    python -m rag.evaluate
    python -m rag.evaluate --generate --limit 10
    python -m rag.evaluate --output evaluation/results/baseline.json
"""
import argparse
import json
import time
from pathlib import Path

from langchain_core.documents import Document
from tqdm import tqdm

from . import config
from .chain import build_answer_chain, cited_indices
from .chunking import chunk_documents
from .ingest import load_documents
from .vectorstore import chunk_id, get_reranker, get_vector_store, stored_ids

REFUSAL_PHRASES = (
    "don't know",
    "do not know",
    "does not contain",
    "doesn't contain",
    "does not mention",
    "doesn't mention",
    "no information",
)


def load_questions(path: Path = config.EVAL_QUESTIONS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def first_hit_rank(docs: list[Document], expected_sources: set[str]) -> int | None:
    """1-based rank of the first chunk from an expected source page, or None if there is none."""
    for rank, doc in enumerate(docs, 1):
        if doc.metadata.get("source") in expected_sources:
            return rank
    return None


def retrieval_metrics(ranks: list[int | None], ks: tuple[int, ...]) -> dict[str, float]:
    """Hit rate at each k (share of questions with a relevant chunk in the top k),
    and mean reciprocal rank, where a miss scores 0."""
    if not ranks:
        return {}

    metrics = {}
    for k in ks:
        hits = 0
        for rank in ranks:
            if rank is not None and rank <= k:
                hits += 1
        metrics[f"hit@{k}"] = hits / len(ranks)

    reciprocal_ranks = [1 / rank if rank is not None else 0.0 for rank in ranks]
    metrics["mrr"] = sum(reciprocal_ranks) / len(ranks)
    return metrics


def normalise(text: str) -> str:
    return text.lower().replace("’", "'")    # curly apostrophes from the LLM


def answer_matches(answer: str, expected_terms: list) -> bool:
    """True if the answer contains every expected term. A term given as a list accepts
    any one of its alternatives, e.g. ["8 million", "eight million"]."""
    answer = normalise(answer)
    for term in expected_terms:
        alternatives = [term] if isinstance(term, str) else term
        if not any(normalise(alt) in answer for alt in alternatives):
            return False
    return True


def is_refusal(answer: str) -> bool:
    answer = normalise(answer)
    return any(phrase in answer for phrase in REFUSAL_PHRASES)


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

def warn_if_store_out_of_sync(store) -> None:
    """Results on a store built from an older corpus or chunking settings are misleading."""
    current = {chunk_id(c) for c in chunk_documents(load_documents())}
    stored = stored_ids(store)
    stale, missing = len(stored - current), len(current - stored)
    if stale or missing:
        print(
            f"WARNING: the vector store is out of sync with the corpus ({stale} stale, "
            f"{missing} missing chunks). Run `python -m rag.pipeline` first; these results "
            "reflect the chunks currently stored.\n"
        )


def evaluate_question(question: dict, store, reranker, answer_chain=None) -> dict:
    text = question["question"]
    expected = set(question["sources"])

    # The same two steps the compression retriever runs, kept separate so both can be scored
    candidates = store.similarity_search(text, k=config.RETRIEVAL_K)
    reranked = reranker.compress_documents(candidates, text) if candidates else []

    result = {
        "id": question["id"],
        "category": question["category"],
        "question": text,
        "expected_sources": sorted(expected),
        "retrieved": [d.metadata.get("source") for d in reranked],
    }
    if expected:
        result["base_rank"] = first_hit_rank(candidates, expected)
        result["reranked_rank"] = first_hit_rank(reranked, expected)

    if answer_chain is not None:
        try:
            answer = answer_chain.invoke({"docs": reranked, "question": text})
        except Exception as e:
            result["answer_error"] = f"{type(e).__name__}: {e}"[:300]
            return result
        result["answer"] = answer
        cited = [reranked[i - 1].metadata.get("source") for i in cited_indices(answer, len(reranked))]
        result["cited_sources"] = cited
        if expected:
            result["answer_correct"] = answer_matches(answer, question["answer_contains"])
            result["citation_correct"] = bool(expected & set(cited))
        else:
            # Nothing in the corpus answers it, so nothing should be cited
            result["answer_correct"] = is_refusal(answer)
            result["citation_correct"] = not cited
    return result


def summarise(results: list[dict]) -> dict:
    """Metrics overall and per question category."""
    groups = {"all": results}
    for r in results:
        groups.setdefault(r["category"], []).append(r)

    summary = {}
    for name, group in groups.items():
        scored = [r for r in group if "base_rank" in r]
        answered = [r for r in group if "answer_correct" in r]
        entry = {"questions": len(group)}
        if scored:
            entry["retrieval_scored"] = len(scored)
            entry["base"] = retrieval_metrics([r["base_rank"] for r in scored], (1, config.RERANK_K, config.RETRIEVAL_K))
            entry["reranked"] = retrieval_metrics([r["reranked_rank"] for r in scored], (1, config.RERANK_K))
        if answered:
            correct = sum(1 for r in answered if r["answer_correct"])
            entry["answer_accuracy"] = correct / len(answered)
            cited_correctly = sum(1 for r in answered if r.get("citation_correct"))
            entry["citation_accuracy"] = cited_correctly / len(answered)
        summary[name] = entry
    return summary


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_retrieval_report(summary: dict, results: list[dict]) -> None:
    k, big_k = config.RERANK_K, config.RETRIEVAL_K
    print("\nRetrieval: hit@k = share of questions with an expected page in the top k\n")
    print(f"{'':16}{'n':>4} | {'base hit@1':>10} {f'hit@{k}':>6} {f'hit@{big_k}':>7} {'MRR':>5}"
          f" | {'reranked hit@1':>14} {f'hit@{k}':>6} {'MRR':>5}")
    for name, entry in summary.items():
        if "base" not in entry:
            continue
        base, reranked = entry["base"], entry["reranked"]
        print(f"{name:16}{entry['retrieval_scored']:>4} | {base['hit@1']:>10.2f} {base[f'hit@{k}']:>6.2f} {base[f'hit@{big_k}']:>7.2f}"
              f" {base['mrr']:>5.2f} | {reranked['hit@1']:>14.2f} {reranked[f'hit@{k}']:>6.2f} {reranked['mrr']:>5.2f}")

    misses = [r for r in results if "reranked_rank" in r and r["reranked_rank"] is None]
    if misses:
        print(f"\nMissed after reranking ({len(misses)}):")
        for r in misses:
            if r["base_rank"] is None:
                where = f"not in the top {big_k} candidates"
            else:
                where = f"candidate rank {r['base_rank']}, dropped by the reranker"
            print(f"  [{r['id']}] {r['question']}")
            print(f"      expected {r['expected_sources']}: {where}")
            print(f"      got {r['retrieved'][:3]}")


def print_generation_report(summary: dict, results: list[dict]) -> None:
    print("\nAnswers: expected terms present, or a refusal for out-of-scope questions")
    print("Citations: an expected page cited, or nothing cited for out-of-scope questions\n")
    print(f"  {'':16}{'answers':>8}{'citations':>11}")
    for name, entry in summary.items():
        if "answer_accuracy" in entry:
            print(f"  {name:16}{entry['answer_accuracy']:>8.2f}{entry['citation_accuracy']:>11.2f}")

    errors = [r for r in results if "answer_error" in r]
    if errors:
        print(f"\n{len(errors)} LLM calls failed and are excluded, e.g. {errors[0]['answer_error'][:150]}")

    wrong = [r for r in results if r.get("answer_correct") is False]
    if wrong:
        print(f"\nIncorrect answers ({len(wrong)}):")
        for r in wrong:
            answer = " ".join(r["answer"].split())
            print(f"  [{r['id']}] {r['question']}")
            print(f"      {answer[:200]}")


def config_snapshot() -> dict:
    """Settings that affect the scores, saved with the results so runs can be compared."""
    names = ["EMBEDDING_MODEL", "RERANKER_MODEL", "LLM_MODEL", "CHUNK_SIZE",
             "CHUNK_OVERLAP", "MIN_CHUNK_CHARS", "RETRIEVAL_K", "RERANK_K"]
    return {name: getattr(config, name) for name in names}


def main(generate: bool, limit: int | None, delay: float, questions_path: Path, output: Path | None):
    questions = load_questions(questions_path)[:limit]

    store = get_vector_store()
    warn_if_store_out_of_sync(store)
    reranker = get_reranker()
    answer_chain = build_answer_chain() if generate else None

    results = []
    for i, question in enumerate(tqdm(questions, unit="question")):
        if generate and i > 0:
            time.sleep(delay)
        results.append(evaluate_question(question, store, reranker, answer_chain))

    summary = summarise(results)
    print_retrieval_report(summary, results)
    if generate:
        print_generation_report(summary, results)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump({"config": config_snapshot(), "summary": summary, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval, and optionally answers, on labelled questions.")
    parser.add_argument("--generate", action="store_true", help="Also score LLM answers (one LLM call per question).")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions.")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between LLM calls (default 4).")
    parser.add_argument("--questions", type=Path, default=config.EVAL_QUESTIONS_FILE, help="Labelled questions file.")
    parser.add_argument("--output", type=Path, help="Save per-question results and metrics as JSON.")
    args = parser.parse_args()
    main(args.generate, args.limit, args.delay, args.questions, args.output)
