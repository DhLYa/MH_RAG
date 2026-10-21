import argparse

from langchain_chroma import Chroma

from .chain import ask, build_chain
from .chunking import chunk_documents
from .config import HYBRID_SEARCH, RERANK_K
from .ingest import load_documents
from .vectorstore import (
    get_candidate_retriever,
    get_compression_retriever,
    get_reranker,
    get_vector_store,
    remove_stale_chunks,
    upsert_chunks,
)


def ingest() -> Chroma:
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    chunks = chunk_documents(documents)
    print(f"split documents into {len(chunks)} chunks.")

    store = get_vector_store()
    removed = remove_stale_chunks(store, chunks)
    if removed:
        print(f"Removed {removed} stale chunks.")
    added = upsert_chunks(store, chunks)
    print(f"Added {added} new chunks." if added else "No new chunks to add.")

    return store

def print_retrieval(candidate_retriever, reranker, question):
    """
    Shows the candidate and reranked results side by side without invoking unnecessary LLM calls.
    Reranks the candidates already fetched rather than searching again, so the question
    is embedded once.
    """
    candidates = candidate_retriever.invoke(question)
    reranked = reranker.compress_documents(candidates, question) if candidates else []

    kind = "vector + keyword, fused" if HYBRID_SEARCH else "vector"
    print(f"\nCandidates top {RERANK_K} of {len(candidates)} ({kind}):")
    for i, d in enumerate(candidates[:RERANK_K], 1):
        print(f"  {i}. {d.metadata.get('source')}")

    print(f"\nReranked top {RERANK_K}:")
    for i, d in enumerate(reranked, 1):
        score = d.metadata.get("relevance_score")
        score = f"{score:.4f}" if isinstance(score, (int, float)) else score
        print(f"  {i}. {d.metadata.get('source')}   score={score}")
        print(f"     {d.page_content[:150].strip()}")

def main(retrieval_only: bool = False):
    store = ingest()

    reranker = get_reranker()
    candidate_retriever = get_candidate_retriever(store)

    if retrieval_only:
        chain = None
        mode = "Retrieval only, no unnecessary LLM calls"
    else:
        chain = build_chain(get_compression_retriever(reranker, candidate_retriever))
        mode = "RAG ready"

    print(f"\n{mode}, Ask a question. (empty line to quit).")
    while True:
        try:
            question = input("\n>").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        if retrieval_only:
            print_retrieval(candidate_retriever, reranker, question)
        else:
            print(ask(chain, question))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monster Hunter RAG pipeline.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Inspect retrieved chunks and scores without unnecessary LLM calls.",
    )
    args = parser.parse_args()
    main(retrieval_only=args.retrieval_only)