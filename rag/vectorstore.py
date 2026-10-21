import hashlib
 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
 
from .config import EMBEDDING_MODEL, RERANKER_MODEL, CHROMA_DIR, RETRIEVAL_K, RERANK_K

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

def get_retriever(store: Chroma, k: int = RETRIEVAL_K):
    retriever = store.as_retriever(
    search_kwargs={"k": k}
)
    return retriever

def get_compression_retriever(reranker, retriever):
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever,
    )
    return compression_retriever