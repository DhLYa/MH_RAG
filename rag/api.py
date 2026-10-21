from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .chain import build_chain
from .vectorstore import (
    get_compression_retriever,
    get_reranker,
    get_retriever,
    get_vector_store,
    is_empty,
)


def build_rag_chain():
    store = get_vector_store()
    # The API never ingests; on an empty store every query would silently answer "I don't know"
    if is_empty(store):
        raise RuntimeError(
            "Vector store is empty. Run `python -m rag.pipeline` once to ingest the corpus."
        )
    retriever = get_retriever(store)
    reranker = get_reranker()
    return build_chain(get_compression_retriever(reranker, retriever))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once at startup rather than on import, so tests can substitute the chain
    app.state.chain = build_rag_chain()
    yield


app = FastAPI(title="Monster Hunter RAG", lifespan=lifespan)


class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return RedirectResponse("/docs")

@app.post("/query")
def query(q: Query, request: Request):
    result = request.app.state.chain.invoke(q.question)
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
