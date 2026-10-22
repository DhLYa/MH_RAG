import json

import pytest
from langchain_core.documents import Document

from rag.config import EVAL_QUESTIONS_FILE, PAGES_FILES
from rag.evaluate import (
    answer_matches,
    evaluate_agent_question,
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


def test_answer_matches_rejects_excluded_terms():
    """Only Nargacuga is weak to Thunder; naming both contains the expected term but is wrong."""
    assert answer_matches("Nargacuga is weak to Thunder.", ["Nargacuga"], ["both"])
    assert not answer_matches("Both Zinogre and Nargacuga are weak to Thunder.", ["Nargacuga"], ["both"])


def test_is_refusal():
    assert is_refusal("I don’t know.")
    assert is_refusal("The context does not contain that information.")
    assert not is_refusal("Rathalos is weak to Dragon.")


# --------------------------------------------------------------------------
# evaluate_question / summarise
# --------------------------------------------------------------------------

class FakeRetriever:
    """Candidate retriever returning one chunk per source, in order."""
    def __init__(self, sources):
        self.sources = sources

    def invoke(self, query):
        return [doc(s) for s in self.sources]


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
    store = FakeRetriever(["Tigrex", "Rathalos", "Rathian"])
    result = evaluate_question(question, store, FakeReranker(), FakeAnswerChain("Dragon [2]."))
    assert result["base_rank"] == 2
    assert result["reranked_rank"] == 2
    assert result["retrieved"] == ["Rathian", "Rathalos"]
    assert result["answer_correct"] is True
    assert result["cited_sources"] == ["Rathalos"]
    assert result["citation_correct"] is True


def test_citing_only_the_wrong_page_is_a_citation_miss():
    question = {"id": "q", "category": "mh_fact", "question": "?", "sources": ["Rathalos"],
                "answer_contains": ["Dragon"]}
    store = FakeRetriever(["Tigrex", "Rathalos", "Rathian"])
    result = evaluate_question(question, store, FakeReranker(), FakeAnswerChain("Dragon [1]."))
    assert result["cited_sources"] == ["Rathian"]
    assert result["answer_correct"] is True
    assert result["citation_correct"] is False


def test_out_of_scope_question_is_scored_on_refusal_only():
    question = {"id": "q", "category": "out_of_scope", "question": "?", "sources": [], "answer_contains": []}
    result = evaluate_question(question, FakeRetriever(["Tigrex"]), FakeReranker(), FakeAnswerChain("I don't know."))
    assert "base_rank" not in result
    assert result["answer_correct"] is True
    assert result["citation_correct"] is True


def test_out_of_scope_answer_citing_a_block_is_a_citation_miss():
    question = {"id": "q", "category": "out_of_scope", "question": "?", "sources": [], "answer_contains": []}
    answer = FakeAnswerChain("I don't know, but see [1].")
    result = evaluate_question(question, FakeRetriever(["Tigrex"]), FakeReranker(), answer)
    assert result["citation_correct"] is False


def test_failed_llm_call_is_recorded_not_scored():
    class FailingChain:
        def invoke(self, inputs):
            raise RuntimeError("429 rate limited")

    question = {"id": "q", "category": "mh_fact", "question": "?", "sources": ["Rathalos"], "answer_contains": []}
    result = evaluate_question(question, FakeRetriever(["Rathalos"]), FakeReranker(), FailingChain())
    assert "answer_correct" not in result
    assert "429" in result["answer_error"]


def test_pipeline_scoring_applies_excluded_terms():
    question = {"id": "q", "category": "cross_page", "question": "?", "sources": ["Nargacuga"],
                "answer_contains": ["Nargacuga"], "answer_excludes": ["both"]}
    answer = FakeAnswerChain("Both Zinogre and Nargacuga are weak to Thunder [1].")
    result = evaluate_question(question, FakeRetriever(["Nargacuga"]), FakeReranker(), answer)
    assert result["answer_correct"] is False


# --------------------------------------------------------------------------
# evaluate_agent_question
# --------------------------------------------------------------------------

class FakeAgent:
    """Answers like rag.agent.Agent, from a fixed set of passages and searches."""
    def __init__(self, answer, sources, searches=None, llm_calls=2):
        self.output = {
            "answer": answer,
            "docs": [doc(s) for s in sources],
            "searches": searches if searches is not None else [{"query": "q", "corpus": None, "passages": [1]}],
            "llm_calls": llm_calls,
        }

    def invoke(self, question):
        return self.output


TIGREX_FIRST = {"id": "q", "category": "cross_page", "question": "Tigrex or Nargacuga first?",
                "sources": ["Tigrex", "Nargacuga"], "answer_contains": ["Tigrex"]}


def test_agent_question_scores_answer_citations_and_cost():
    searches = [{"query": "Tigrex debut", "corpus": None, "passages": [1, 2]},
                {"query": "Nargacuga debut", "corpus": None, "passages": [3]}]
    agent = FakeAgent("Tigrex came first [1], before Nargacuga [3].", ["Tigrex", "Rathalos", "Nargacuga"],
                      searches=searches, llm_calls=3)
    result = evaluate_agent_question(TIGREX_FIRST, agent)

    assert result["found_expected"] is True
    assert result["answer_correct"] is True
    assert result["cited_sources"] == ["Tigrex", "Nargacuga"]
    assert result["citation_correct"] is True
    assert result["searches"] == searches
    assert result["llm_calls"] == 3
    assert "base_rank" not in result, "agent results have no single ranked list to score"


def test_agent_question_records_when_no_expected_page_was_seen():
    result = evaluate_agent_question(TIGREX_FIRST, FakeAgent("I don't know.", ["Rathalos"]))
    assert result["found_expected"] is False
    assert result["answer_correct"] is False


def test_agent_failure_is_recorded_not_scored():
    class FailingAgent:
        def invoke(self, question):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    result = evaluate_agent_question(TIGREX_FIRST, FailingAgent())
    assert "answer_correct" not in result
    assert "429" in result["answer_error"]


def test_summarise_reports_agent_cost():
    results = [
        {"category": "cross_page", "found_expected": True, "searches": [{}, {}], "llm_calls": 3,
         "answer_correct": True, "citation_correct": True},
        {"category": "cross_page", "found_expected": False, "searches": [{}], "llm_calls": 2,
         "answer_correct": False, "citation_correct": False},
    ]
    entry = summarise(results)["cross_page"]
    assert entry["expected_page_found"] == 0.5
    assert entry["avg_searches"] == 1.5
    assert entry["avg_llm_calls"] == 2.5


def test_summarise_excludes_out_of_scope_from_retrieval():
    results = [
        {"category": "mh_fact", "base_rank": 1, "reranked_rank": 1, "answer_correct": True, "citation_correct": True},
        {"category": "mh_fact", "base_rank": None, "reranked_rank": None, "answer_correct": False,
         "citation_correct": False},
        {"category": "out_of_scope", "answer_correct": True, "citation_correct": True},
    ]
    summary = summarise(results)
    assert summary["all"]["retrieval_scored"] == 2
    assert summary["all"]["reranked"]["hit@1"] == 0.5
    assert summary["all"]["answer_accuracy"] == pytest.approx(2 / 3)
    assert summary["all"]["citation_accuracy"] == pytest.approx(2 / 3)
    assert "base" not in summary["out_of_scope"]


# --------------------------------------------------------------------------
# The question set itself
# --------------------------------------------------------------------------

CATEGORIES = {
    "mh_fact", "mh_paraphrase", "wiki_fact",
    "variant",          # look-alike monsters, e.g. Azure Rathalos vs Rathalos
    "cross_page",       # comparisons needing facts from two pages
    "keyword",          # rare exact terms such as Japanese names and quest titles
    "lost_content",     # facts only in chunks the original length filter dropped
    "out_of_scope",
}


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
        for term in q.get("answer_excludes", []):
            assert isinstance(term, str) and term, q["id"]


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
