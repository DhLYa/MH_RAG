import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .agent import Agent
from .chain import build_chain, cited_indices, get_llm
from .config import CORPORA
from .vectorstore import (
    build_compression_retrievers,
    get_reranker,
    get_vector_store,
    is_empty,
)

logger = logging.getLogger(__name__)


def build_answerers() -> dict:
    """
    What answers a query, by mode and then corpus filter (None for all corpora):
    {"pipeline": {corpus: chain}, "agent": {corpus: agent}}. Both modes share one set of
    retrievers, so each BM25 index is built once.
    """
    store = get_vector_store()
    # The API never ingests; on an empty store every query would silently answer "I don't know"
    if is_empty(store):
        raise RuntimeError(
            "Vector store is empty. Run `python -m rag.pipeline` once to ingest the corpus."
        )
    retrievers = build_compression_retrievers(store, get_reranker())
    llm = get_llm()

    answerers = {"pipeline": {}, "agent": {}}
    for corpus, retriever in retrievers.items():
        answerers["pipeline"][corpus] = build_chain(retriever)
        answerers["agent"][corpus] = Agent(llm, retrievers, corpus=corpus)
    return answerers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once at startup rather than on import, so tests can substitute fakes
    app.state.answerers = build_answerers()
    yield


app = FastAPI(title="Monster Hunter RAG", lifespan=lifespan)


class Query(BaseModel):
    question: str
    # Search one corpus only; omitted or null searches both
    corpus: Literal[CORPORA] | None = None
    # Let the model search for itself, several times if needed; slower and uses 2-5 LLM calls
    agent: bool = False

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
    mode = "agent" if q.agent else "pipeline"
    answerer = request.app.state.answerers[mode][q.corpus]
    try:
        result = answerer.invoke(q.question)
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
    response = {
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
    # The agent's own queries, with the passage numbers each returned
    if q.agent:
        response["searches"] = result["searches"]
    return response
