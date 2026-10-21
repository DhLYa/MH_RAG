import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .chain import build_chain, cited_indices
from .config import CORPORA
from .vectorstore import (
    get_candidate_retriever,
    get_compression_retriever,
    get_reranker,
    get_vector_store,
    is_empty,
)

logger = logging.getLogger(__name__)


def build_rag_chains() -> dict:
    """One chain per corpus filter, keyed by corpus name, plus None for all corpora."""
    store = get_vector_store()
    # The API never ingests; on an empty store every query would silently answer "I don't know"
    if is_empty(store):
        raise RuntimeError(
            "Vector store is empty. Run `python -m rag.pipeline` once to ingest the corpus."
        )
    reranker = get_reranker()
    chains = {}
    for corpus in (None, *CORPORA):
        retriever = get_candidate_retriever(store, corpus=corpus)
        chains[corpus] = build_chain(get_compression_retriever(reranker, retriever))
    return chains


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once at startup rather than on import, so tests can substitute the chains
    app.state.chains = build_rag_chains()
    yield


app = FastAPI(title="Monster Hunter RAG", lifespan=lifespan)


class Query(BaseModel):
    question: str
    # Search one corpus only; omitted or null searches both
    corpus: Literal[CORPORA] | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return RedirectResponse("/docs")

def is_rate_limit(error: Exception) -> bool:
    """Gemini reports 429 / RESOURCE_EXHAUSTED; Voyage raises RateLimitError."""
    text = f"{type(error).__name__} {error}"
    return any(marker in text for marker in ("RateLimit", "ResourceExhausted", "RESOURCE_EXHAUSTED", "429"))

@app.post("/query")
def query(q: Query, request: Request):
    try:
        result = request.app.state.chains[q.corpus].invoke(q.question)
    except Exception as e:
        # The embedding, reranking or generation call failed upstream. Log the details here and
        # tell the client what it can do about it, without passing the provider's error text on.
        logger.exception("Query failed")
        if is_rate_limit(e):
            raise HTTPException(
                status_code=503,
                detail="The model provider is rate limiting requests. Retry in about 30 seconds.",
                headers={"Retry-After": "30"},
            ) from e
        raise HTTPException(
            status_code=503,
            detail="Answering failed because a model provider is unavailable. Try again shortly.",
        ) from e
    cited = cited_indices(result["answer"], len(result["docs"]))
    return {
        "answer": result["answer"],
        "sources": [
            {
                "id": i,
                "title": d.metadata.get("source"),
                "corpus": d.metadata.get("corpus"),
                "relevance": d.metadata.get("relevance_score"),
                "cited": i in cited,
            }
            for i, d in enumerate(result["docs"], 1)
        ],
    }
