import hashlib
import re

from langchain_chroma import Chroma
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank

from .config import (
    BM25_K,
    CHROMA_DIR,
    CORPORA,
    EMBEDDING_MODEL,
    HYBRID_SEARCH,
    RERANK_K,
    RERANKER_MODEL,
    RETRIEVAL_K,
)


def get_embedding_model() -> VoyageAIEmbeddings:
    embedding_model = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
    return embedding_model

def get_vector_store() -> Chroma:
    vector_store = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=get_embedding_model()
)
    return vector_store

def chunk_id(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    corpus = doc.metadata.get("corpus", "")
    h = hashlib.sha256(f"{corpus}-{source}-{doc.page_content}".encode()).hexdigest()
    return h

def stored_ids(store: Chroma) -> set[str]:
    """IDs only, without loading every document and its metadata."""
    return set(store.get(include=[])["ids"])

def is_empty(store: Chroma) -> bool:
    return not store.get(include=[], limit=1)["ids"]

def upsert_chunks(store: Chroma, chunks: list[Document]) -> int:
    existing = stored_ids(store)
    seen = set()
    new_chunks, new_ids = [], []

    for chunk in chunks:
        cid = chunk_id(chunk)
        if cid in existing or cid in seen:
            continue
        seen.add(cid)
        new_chunks.append(chunk)
        new_ids.append(cid)

    if new_chunks:
        store.add_documents(documents=new_chunks, ids=new_ids)
    return len(new_chunks)

def remove_stale_chunks(store: Chroma, chunks: list[Document], batch_size: int = 5000) -> int:
    """
    Deletes stored chunks the current corpus and chunking settings no longer produce,
    e.g. after changing CHUNK_SIZE or re-scraping an edited page. Without this they
    stay in the store and keep being retrieved alongside their replacements.
    """
    current = {chunk_id(c) for c in chunks}
    stale = sorted(stored_ids(store) - current)
    # Chroma caps the number of IDs accepted per call
    for i in range(0, len(stale), batch_size):
        store.delete(ids=stale[i:i + batch_size])
    return len(stale)

def get_reranker(k: int = RERANK_K):
    reranker = VoyageAIRerank(model=RERANKER_MODEL, top_k=k)
    return reranker

def get_retriever(store: Chroma, k: int = RETRIEVAL_K, corpus: str | None = None):
    search_kwargs = {"k": k}
    if corpus:
        search_kwargs["filter"] = {"corpus": corpus}
    return store.as_retriever(search_kwargs=search_kwargs)

def tokenise(text: str) -> list[str]:
    """Lower-cased words, so "Rathalos," in a chunk matches "rathalos" in a question."""
    return re.findall(r"\w+", text.lower())

def get_bm25_retriever(store: Chroma, k: int = BM25_K, corpus: str | None = None) -> BM25Retriever:
    """Keyword index over the stored chunks, or one corpus's chunks, built in memory from the store."""
    where = {"corpus": corpus} if corpus else None
    data = store.get(where=where, include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(data["documents"], data["metadatas"])
    ]
    return BM25Retriever.from_documents(docs, k=k, preprocess_func=tokenise)

def get_candidate_retriever(store: Chroma, hybrid: bool = HYBRID_SEARCH, corpus: str | None = None):
    """
    Candidates for the reranker, optionally from one corpus only. With hybrid search, vector
    and BM25 keyword results are merged by reciprocal rank fusion, so a rare exact term such
    as a Japanese name can reach the reranker even when its embedding isn't close to the question's.
    """
    vector = get_retriever(store, corpus=corpus)
    if not hybrid:
        return vector
    keyword = get_bm25_retriever(store, corpus=corpus)
    return EnsembleRetriever(retrievers=[vector, keyword], weights=[0.5, 0.5])

def get_compression_retriever(reranker, retriever):
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever,
    )
    return compression_retriever

def build_compression_retrievers(store: Chroma, reranker, hybrid: bool = HYBRID_SEARCH) -> dict:
    """
    A reranking retriever for each corpus filter: None searches every corpus, and each corpus
    name searches that one. Built once and shared, since each builds its own BM25 index.
    """
    retrievers = {}
    for corpus in (None, *CORPORA):
        candidates = get_candidate_retriever(store, hybrid=hybrid, corpus=corpus)
        retrievers[corpus] = get_compression_retriever(reranker, candidates)
    return retrievers