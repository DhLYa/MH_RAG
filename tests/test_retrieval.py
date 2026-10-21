from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag.vectorstore import get_bm25_retriever, get_candidate_retriever, tokenise


def chunk(source, text, corpus="mh_wiki_monsters"):
    return Document(page_content=f"Source: {source}\n\n{text}", metadata={"source": source, "corpus": corpus})


CHUNKS = [
    chunk("Rathalos", "The Fire Wyvern Rathalos (Rioreusu) is a Flying Wyvern."),
    chunk("Arkveld", "The Chain Blade Wyvern Arkveld (Arushuberudo) is a Flying Wyvern."),
    chunk("Amatsu", "Amatsu is an Elder Dragon that controls storms."),
    chunk("Monster_Hunter_Wilds", "Arkveld is the flagship of Wilds.", corpus="wikipedia_pages"),
]


class FakeVectorRetriever(BaseRetriever):
    """Stands in for Chroma's vector search: always returns the same ranked chunks."""
    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return self.docs


class FakeStore:
    """The two Chroma calls the candidate retriever makes, honouring a corpus filter."""
    def __init__(self, chunks, vector_results):
        self.chunks = chunks
        self.vector_results = vector_results
        self.search_kwargs = None

    def get(self, where, include):
        chunks = [c for c in self.chunks if not where or c.metadata["corpus"] == where["corpus"]]
        return {"documents": [c.page_content for c in chunks], "metadatas": [c.metadata for c in chunks]}

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
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


def test_corpus_filter_reaches_vector_search():
    store = FakeStore(CHUNKS, vector_results=[])
    get_candidate_retriever(store, hybrid=False, corpus="wikipedia_pages")
    assert store.search_kwargs["filter"] == {"corpus": "wikipedia_pages"}


def test_no_corpus_means_no_vector_filter():
    store = FakeStore(CHUNKS, vector_results=[])
    get_candidate_retriever(store, hybrid=False)
    assert "filter" not in store.search_kwargs


def test_corpus_filter_limits_the_keyword_index():
    """BM25 is built per corpus, so a filtered search can't surface the other corpus's chunks."""
    retriever = get_bm25_retriever(FakeStore(CHUNKS, []), k=5, corpus="wikipedia_pages")
    assert {d.metadata["corpus"] for d in retriever.invoke("Arkveld")} == {"wikipedia_pages"}
