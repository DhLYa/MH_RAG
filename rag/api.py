from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .chain import build_chain
from .vectorstore import (
    get_compression_retriever,
    get_reranker,
    get_retriever,
    get_vector_store,
)

app = FastAPI(title="Monster Hunter RAG")

_store = get_vector_store()
_retriever = get_retriever(_store)
_reranker = get_reranker()
_chain = build_chain(get_compression_retriever(_reranker, _retriever))


class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return RedirectResponse("/docs")

@app.post("/query")
def query(q: Query):
    result = _chain.invoke(q.question)
    return {
        "answer": result["answer"],
        "sources": [
            {
                "id": i,
                "title": d.metadata.get("source"),
                "corpus": d.metadata.get("corpus"),
                "relevance": d.metadata.get("relevance_score"),
            }
            for i, d in enumerate(result["docs"], 1)
        ],
    }