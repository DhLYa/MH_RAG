"""
Scores the pipeline against the labelled questions in evaluation/questions.json.

Retrieval is scored by where the first chunk from an expected source page lands, both
among the candidates before reranking and in the top RERANK_K after. Candidates come from
vector search, fused with BM25 keyword search when hybrid; --retrieval picks either, so the
two can be compared. This needs only Voyage calls.

--generate also asks the LLM each question and checks the answer contains the expected
terms, or that it says it doesn't know for out-of-scope questions. That is one LLM call
per question, spaced by --delay to stay under the free-tier rate limit.

--agent evaluates agent mode instead, where the model runs its own searches. It always answers,
using 2-5 LLM calls per question, so the pause between questions grows with the calls made.

    python -m rag.evaluate
    python -m rag.evaluate --generate --limit 10
    python -m rag.evaluate --output evaluation/results/baseline.json
    python -m rag.evaluate --retrieval vector
    python -m rag.evaluate --agent --output evaluation/results/agent.json
"""
import argparse
import json
import time
from pathlib import Path

from langchain_core.documents import Document
from tqdm import tqdm

from . import config
from .agent import Agent
from .chain import build_answer_chain, cited_indices, get_llm
from .chunking import chunk_documents
from .ingest import load_documents
from .vectorstore import (
    build_compression_retrievers,
    chunk_id,
    get_candidate_retriever,
    get_reranker,
    get_vector_store,
    stored_ids,
)

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


def answer_matches(answer: str, expected_terms: list, excluded_terms: list = ()) -> bool:
    """True if the answer contains every expected term and none of the excluded ones.

    A term given as a list accepts any one of its alternatives, e.g. ["8 million", "eight million"].
    Excluded terms catch answers that name the right thing for the wrong reason, e.g. "both" in
    "Both Zinogre and Nargacuga are weak to Thunder" when only Nargacuga is.
    """
    answer = normalise(answer)
    for term in expected_terms:
        alternatives = [term] if isinstance(term, str) else term
        if not any(normalise(alt) in answer for alt in alternatives):
            return False
    for term in excluded_terms:
        if normalise(term) in answer:
            return False
    return True


def is_refusal(answer: str) -> bool:
    answer = normalise(answer)
    return any(phrase in answer for phrase in REFUSAL_PHRASES)


def score_answer(question: dict, answer: str, docs: list[Document]) -> dict:
    """Answer and citation scores for one question, given the answer and the passages it could cite."""
    expected = set(question["sources"])
    cited = [docs[i - 1].metadata.get("source") for i in cited_indices(answer, len(docs))]
    scores = {"answer": answer, "cited_sources": cited}

    if expected:
        excluded = question.get("answer_excludes", [])
        scores["answer_correct"] = answer_matches(answer, question["answer_contains"], excluded)
        scores["citation_correct"] = bool(expected & set(cited))
    else:
        # Nothing in the corpus answers it, so nothing should be cited
        scores["answer_correct"] = is_refusal(answer)
        scores["citation_correct"] = not cited
    return scores


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


def evaluate_question(question: dict, candidate_retriever, reranker, answer_chain=None) -> dict:
    text = question["question"]
    expected = set(question["sources"])

    # The same two steps the compression retriever runs, kept separate so both can be scored
    candidates = candidate_retriever.invoke(text)
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
        result.update(score_answer(question, answer, reranked))
    return result


def evaluate_agent_question(question: dict, agent) -> dict:
    """
    Agent mode chooses its own searches, so there is no single ranked list to score. Retrieval is
    scored as whether an expected page was among all the passages the agent saw; answers and
    citations are scored as in the pipeline. The searches and LLM calls it spent are recorded too.
    """
    text = question["question"]
    expected = set(question["sources"])
    result = {
        "id": question["id"],
        "category": question["category"],
        "question": text,
        "expected_sources": sorted(expected),
    }

    try:
        output = agent.invoke(text)
    except Exception as e:
        result["answer_error"] = f"{type(e).__name__}: {e}"[:300]
        return result

    docs = output["docs"]
    result["retrieved"] = [d.metadata.get("source") for d in docs]
    result["searches"] = output["searches"]
    result["llm_calls"] = output["llm_calls"]
    if expected:
        result["found_expected"] = first_hit_rank(docs, expected) is not None
    result.update(score_answer(question, output["answer"], docs))
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

        # Agent mode only: did it find an expected page, and what did it spend doing so
        searched = [r for r in group if "found_expected" in r]
        if searched:
            entry["expected_page_found"] = sum(1 for r in searched if r["found_expected"]) / len(searched)
        costed = [r for r in group if "llm_calls" in r]
        if costed:
            entry["avg_searches"] = sum(len(r["searches"]) for r in costed) / len(costed)
            entry["avg_llm_calls"] = sum(r["llm_calls"] for r in costed) / len(costed)

        summary[name] = entry
    return summary


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_retrieval_report(summary: dict, results: list[dict]) -> None:
    k, big_k = config.RERANK_K, config.RETRIEVAL_K
    print("\nRetrieval: hit@k = share of questions with an expected page in the top k\n")
    print(f"{'':16}{'n':>4} | {'candidates hit@1':>16} {f'hit@{k}':>6} {f'hit@{big_k}':>7} {'MRR':>5}"
          f" | {'reranked hit@1':>14} {f'hit@{k}':>6} {'MRR':>5}")
    for name, entry in summary.items():
        if "base" not in entry:
            continue
        base, reranked = entry["base"], entry["reranked"]
        print(f"{name:16}{entry['retrieval_scored']:>4} | {base['hit@1']:>16.2f} {base[f'hit@{k}']:>6.2f} {base[f'hit@{big_k}']:>7.2f}"
              f" {base['mrr']:>5.2f} | {reranked['hit@1']:>14.2f} {reranked[f'hit@{k}']:>6.2f} {reranked['mrr']:>5.2f}")

    misses = [r for r in results if "reranked_rank" in r and r["reranked_rank"] is None]
    if misses:
        print(f"\nMissed after reranking ({len(misses)}):")
        for r in misses:
            if r["base_rank"] is None:
                where = "not among the candidates"
            else:
                where = f"candidate rank {r['base_rank']}, dropped by the reranker"
            print(f"  [{r['id']}] {r['question']}")
            print(f"      expected {r['expected_sources']}: {where}")
            print(f"      got {r['retrieved'][:3]}")


def print_agent_report(summary: dict, results: list[dict]) -> None:
    print("\nAgent searches: found = share of questions where an expected page was among the passages seen\n")
    print(f"{'':16}{'n':>4} | {'found':>6} {'searches':>9} {'LLM calls':>10}")
    for name, entry in summary.items():
        if "expected_page_found" not in entry:
            continue
        print(f"{name:16}{entry['questions']:>4} | {entry['expected_page_found']:>6.2f}"
              f" {entry['avg_searches']:>9.1f} {entry['avg_llm_calls']:>10.1f}")

    misses = [r for r in results if r.get("found_expected") is False]
    if misses:
        print(f"\nNo expected page found ({len(misses)}):")
        for r in misses:
            queries = [s["query"] for s in r["searches"]] or ["(no searches)"]
            print(f"  [{r['id']}] {r['question']}")
            print(f"      expected {r['expected_sources']}; searched {queries}")


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


def config_snapshot(hybrid: bool, agent: bool) -> dict:
    """Settings that affect the scores, saved with the results so runs can be compared."""
    names = ["EMBEDDING_MODEL", "RERANKER_MODEL", "LLM_MODEL", "LLM_TEMPERATURE", "CHUNK_SIZE",
             "CHUNK_OVERLAP", "MIN_CHUNK_CHARS", "RETRIEVAL_K", "BM25_K", "RERANK_K"]
    snapshot = {name: getattr(config, name) for name in names}
    snapshot["HYBRID_SEARCH"] = hybrid
    snapshot["MODE"] = "agent" if agent else "pipeline"
    if agent:
        snapshot["AGENT_MAX_SEARCHES"] = config.AGENT_MAX_SEARCHES
    return snapshot


def main(generate: bool, limit: int | None, delay: float, questions_path: Path, output: Path | None,
         hybrid: bool = config.HYBRID_SEARCH, agent: bool = False):
    questions = load_questions(questions_path)[:limit]

    store = get_vector_store()
    warn_if_store_out_of_sync(store)
    reranker = get_reranker()

    if agent:
        # The agent always answers, so this mode always calls the LLM
        agent_runner = Agent(get_llm(), build_compression_retrievers(store, reranker, hybrid=hybrid))

        def run(question):
            return evaluate_agent_question(question, agent_runner)
    else:
        candidate_retriever = get_candidate_retriever(store, hybrid=hybrid)
        answer_chain = build_answer_chain() if generate else None

        def run(question):
            return evaluate_question(question, candidate_retriever, reranker, answer_chain)

    calls_llm = agent or generate
    results = []
    for question in tqdm(questions, unit="question"):
        if results and calls_llm:
            # Space LLM calls out for the free tier; an agent question can make several
            time.sleep(delay * results[-1].get("llm_calls", 1))
        results.append(run(question))

    summary = summarise(results)
    if agent:
        print_agent_report(summary, results)
    else:
        print_retrieval_report(summary, results)
    if calls_llm:
        print_generation_report(summary, results)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump({"config": config_snapshot(hybrid, agent), "summary": summary, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"\nSaved results to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval, and optionally answers, on labelled questions.")
    parser.add_argument("--generate", action="store_true", help="Also score LLM answers (one LLM call per question).")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions.")
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between LLM calls (default 4).")
    parser.add_argument("--questions", type=Path, default=config.EVAL_QUESTIONS_FILE, help="Labelled questions file.")
    parser.add_argument("--output", type=Path, help="Save per-question results and metrics as JSON.")
    parser.add_argument("--retrieval", choices=["vector", "hybrid"],
                        default="hybrid" if config.HYBRID_SEARCH else "vector",
                        help="Candidate retrieval to evaluate (default from config.HYBRID_SEARCH).")
    parser.add_argument("--agent", action="store_true",
                        help="Evaluate agent mode instead of the pipeline (always answers; 2-5 LLM calls per question).")
    args = parser.parse_args()
    main(args.generate, args.limit, args.delay, args.questions, args.output,
         hybrid=args.retrieval == "hybrid", agent=args.agent)
