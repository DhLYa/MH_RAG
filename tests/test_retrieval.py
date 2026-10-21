from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.vectorstore import get_bm25_retriever, get_candidate_retriever, tokenise


def chunk(source, text):
    return Document(page_content=f"Source: {source}\n\n{text}", metadata={"source": source})


CHUNKS = [
    chunk("Rathalos", "The Fire Wyvern Rathalos (Rioreusu) is a Flying Wyvern."),
    chunk("Arkveld", "The Chain Blade Wyvern Arkveld (Arushuberudo) is a Flying Wyvern."),
    chunk("Amatsu", "Amatsu is an Elder Dragon that controls storms."),
]


class FakeVectorRetriever(BaseRetriever):
    """Stands in for Chroma's vector search: always returns the same ranked chunks."""
    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return self.docs


class FakeStore:
    """The two Chroma calls the candidate retriever makes."""
    def __init__(self, chunks, vector_results):
        self.chunks = chunks
        self.vector_results = vector_results

    def get(self, include):
        return {"documents": [c.page_content for c in self.chunks],
                "metadatas": [c.metadata for c in self.chunks]}

    def as_retriever(self, search_kwargs):
        return FakeVectorRetriever(docs=self.vector_results)


def test_tokenise_lowercases_and_drops_punctuation():
    assert tokenise("Rathalos, the Fire Wyvern (Rioreusu).") == ["rathalos", "the", "fire", "wyvern", "rioreusu"]


def test_bm25_finds_a_rare_exact_term():
    retriever = get_bm25_retriever(FakeStore(CHUNKS, []), k=1)
    assert [d.metadata["source"] for d in retriever.invoke("Which monster is Arushuberudo?")] == ["Arkveld"]


def test_vector_only_candidates_skip_bm25():
    store = FakeStore(CHUNKS, vector_results=[CHUNKS[2]])
    candidates = get_candidate_retriever(store, hybrid=False).invoke("Arushuberudo")
    assert [d.metadata["source"] for d in candidates] == ["Amatsu"]


def test_hybrid_candidates_add_keyword_matches_vector_search_missed():
    """The case hybrid search exists for: an exact Japanese name the embedding doesn't place."""
    store = FakeStore(CHUNKS, vector_results=[CHUNKS[2], CHUNKS[0]])
    sources = [d.metadata["source"] for d in get_candidate_retriever(store, hybrid=True).invoke("Arushuberudo")]
    assert "Arkveld" in sources
    assert len(sources) == len(set(sources)), "chunks found by both searches appear once"
