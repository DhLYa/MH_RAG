import json

import pytest
from langchain_core.documents import Document

from rag.config import EVAL_QUESTIONS_FILE, PAGES_FILES
from rag.evaluate import (
    answer_matches,
    evaluate_question,
    first_hit_rank,
    is_refusal,
    load_questions,
    retrieval_metrics,
    summarise,
)


def doc(source):
    return Document(page_content="text", metadata={"source": source})


# --------------------------------------------------------------------------
# first_hit_rank / retrieval_metrics
# --------------------------------------------------------------------------

def test_first_hit_rank_is_one_based():
    docs = [doc("Rathian"), doc("Rathalos"), doc("Rathalos")]
    assert first_hit_rank(docs, {"Rathalos"}) == 2


def test_first_hit_rank_accepts_any_expected_source():
    docs = [doc("Tigrex"), doc("Stygian Zinogre"), doc("Zinogre")]
    assert first_hit_rank(docs, {"Zinogre", "Stygian Zinogre"}) == 2


def test_first_hit_rank_none_when_missed():
    assert first_hit_rank([doc("Tigrex")], {"Zinogre"}) is None
    assert first_hit_rank([], {"Zinogre"}) is None


def test_retrieval_metrics():
    metrics = retrieval_metrics([1, 3, None, 6], ks=(1, 5))
    assert metrics["hit@1"] == 0.25
    assert metrics["hit@5"] == 0.5
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 6) / 4)


def test_retrieval_metrics_empty():
    assert retrieval_metrics([], ks=(1, 5)) == {}


# --------------------------------------------------------------------------
# answer_matches / is_refusal
# --------------------------------------------------------------------------

def test_answer_matches_needs_every_term():
    assert answer_matches("Weak to water, ice and dragon.", ["Water", "Ice", "Dragon"])
    assert not answer_matches("Weak to water and ice.", ["Water", "Ice", "Dragon"])


def test_answer_matches_accepts_alternatives():
    assert answer_matches("It sold eight million copies.", [["8 million", "eight million"]])
    assert not answer_matches("It sold ten million copies.", [["8 million", "eight million"]])


def test_answer_matches_normalises_curly_apostrophes():
    assert answer_matches("That is Safi’jiiva.", ["Safi'jiiva"])


def test_is_refusal():
    assert is_refusal("I don’t know.")
    assert is_refusal("The context does not contain that information.")
    assert not is_refusal("Rathalos is weak to Dragon.")


# --------------------------------------------------------------------------
# evaluate_question / summarise
# --------------------------------------------------------------------------

class FakeStore:
    def __init__(self, sources):
        self.sources = sources

    def similarity_search(self, query, k):
        return [doc(s) for s in self.sources[:k]]


class FakeReranker:
    """Reverses the candidates and keeps the top 2."""
    def compress_documents(self, documents, query):
        return list(reversed(documents))[:2]


class FakeAnswerChain:
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, inputs):
        return self.answer


def test_evaluate_question_scores_both_stages():
    question = {"id": "q", "category": "mh_fact", "question": "?", "sources": ["Rathalos"],
                "answer_contains": ["Dragon"]}
    store = FakeStore(["Tigrex", "Rathalos", "Rathian"])
    result = evaluate_question(question, store, FakeReranker(), FakeAnswerChain("Dragon."))
    assert result["base_rank"] == 2
    assert result["reranked_rank"] == 2
    assert result["retrieved"] == ["Rathian", "Rathalos"]
    assert result["answer_correct"] is True


def test_out_of_scope_question_is_scored_on_refusal_only():
    question = {"id": "q", "category": "out_of_scope", "question": "?", "sources": [], "answer_contains": []}
    result = evaluate_question(question, FakeStore(["Tigrex"]), FakeReranker(), FakeAnswerChain("I don't know."))
    assert "base_rank" not in result
    assert result["answer_correct"] is True


def test_failed_llm_call_is_recorded_not_scored():
    class FailingChain:
        def invoke(self, inputs):
            raise RuntimeError("429 rate limited")

    question = {"id": "q", "category": "mh_fact", "question": "?", "sources": ["Rathalos"], "answer_contains": []}
    result = evaluate_question(question, FakeStore(["Rathalos"]), FakeReranker(), FailingChain())
    assert "answer_correct" not in result
    assert "429" in result["answer_error"]


def test_summarise_excludes_out_of_scope_from_retrieval():
    results = [
        {"category": "mh_fact", "base_rank": 1, "reranked_rank": 1, "answer_correct": True},
        {"category": "mh_fact", "base_rank": None, "reranked_rank": None, "answer_correct": False},
        {"category": "out_of_scope", "answer_correct": True},
    ]
    summary = summarise(results)
    assert summary["all"]["retrieval_scored"] == 2
    assert summary["all"]["reranked"]["hit@1"] == 0.5
    assert summary["all"]["answer_accuracy"] == pytest.approx(2 / 3)
    assert "base" not in summary["out_of_scope"]


# --------------------------------------------------------------------------
# The question set itself
# --------------------------------------------------------------------------

CATEGORIES = {"mh_fact", "mh_paraphrase", "wiki_fact", "out_of_scope"}


def test_questions_are_well_formed():
    questions = load_questions(EVAL_QUESTIONS_FILE)
    assert len({q["id"] for q in questions}) == len(questions), "ids must be unique"
    for q in questions:
        assert q["category"] in CATEGORIES, q["id"]
        # Out-of-scope questions have no source to find and nothing to say but "I don't know"
        out_of_scope = q["category"] == "out_of_scope"
        assert bool(q["sources"]) != out_of_scope, q["id"]
        assert bool(q["answer_contains"]) != out_of_scope, q["id"]
        for term in q["answer_contains"]:
            assert isinstance(term, str) or (isinstance(term, list) and term), q["id"]


@pytest.mark.skipif(not all(p.exists() for p in PAGES_FILES), reason="corpus not built")
def test_question_labels_match_the_corpus():
    """Every expected source exists, and every expected answer term appears in one of them,
    so a failing question points at the pipeline rather than at a mislabelled question."""
    pages = {}
    for path in PAGES_FILES:
        with open(path, encoding="utf-8") as f:
            pages.update(json.load(f))

    for q in load_questions(EVAL_QUESTIONS_FILE):
        for source in q["sources"]:
            assert source in pages, f"{q['id']}: no page titled {source!r}"
        source_text = " ".join(pages[s] for s in q["sources"]).lower()
        for term in q["answer_contains"]:
            alternatives = [term] if isinstance(term, str) else term
            assert any(alt.lower() in source_text for alt in alternatives), f"{q['id']}: {alternatives}"
