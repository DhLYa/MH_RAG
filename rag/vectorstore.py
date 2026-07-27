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

def upsert_chunks(store: Chroma, chunks: list[Document]) -> int:
    existing = set(store.get()["ids"])
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