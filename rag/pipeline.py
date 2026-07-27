import argparse

from langchain_chroma import Chroma

from .chain import ask, build_chain
from .chunking import chunk_documents
from .config import RERANK_K, RETRIEVAL_K
from .ingest import load_documents
from .vectorstore import (
    get_compression_retriever,
    get_reranker,
    get_retriever,
    get_vector_store,
    upsert_chunks,
)


def ingest() -> Chroma:
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    chunks = chunk_documents(documents)
    print(f"split documents into {len(chunks)} chunks.")

    store = get_vector_store()
    added = upsert_chunks(store, chunks)
    print(f"Added {added} new chunks." if added else "No new chunks to add.")

    return store

def print_retrieval(store, compression_retriever, question):
    """
    Shows the base and reranked results side by side without invoking unnessary LLM calls
    """
    scored = store.similarity_search_with_score(question, k=RETRIEVAL_K)
    reranked = compression_retriever.invoke(question)

    print(f"\nBase retriever top {RERANK_K} (distance, lower = closer):")
    for i, (d, distance) in enumerate(scored[:RERANK_K], 1):
        print(f"  {i}. {d.metadata.get('source')}   distance={distance:.4f}")

    print(f"\nReranked top {RERANK_K}:")
    for i, d in enumerate(reranked, 1):
        score = d.metadata.get("relevance_score")
        score = f"{score:.4f}" if isinstance(score, (int, float)) else score
        print(f"  {i}. {d.metadata.get('source')}   score={score}")
        print(f"     {d.page_content[:150].strip()}")

def main(retrieval_only: bool = False):
    store = ingest()

    reranker = get_reranker()
    retriever = get_retriever(store)
    compression_retriever = get_compression_retriever(reranker, retriever)

    if retrieval_only:
        chain = None
        mode = "Retrieval only, no unnecessary LLM calls"
    else:
        chain = build_chain(compression_retriever)
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
            print_retrieval(store, compression_retriever, question)
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