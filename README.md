# Monster Hunter RAG

A retrieval-augmented generation pipeline that answers questions about the Monster Hunter
game series using a corpus built from wikipedia articles converted to JSON files via the Wikipedia API. Questions are answered only from the retrieved source material, so the system cites where each answer came from and says
"I don't know" rather than falling back on the language model's own knowledge.

Built with LangChain, Voyage AI embeddings and reranking, Chroma as a persistent vector
store, and Gemini 3.5-flash for free generation.

## Example

```
> How many copies did Monster Hunter Portable 3rd sell?

Monster Hunter Portable 3rd sold 2.58 million units in Japan within two weeks of its
December 2010 release, and went on to reach 4.8 million units in Japan. As of 2018 it
had sold 4.9 million copies, despite never receiving a Western release.

Sources:
  [1] Monster_Hunter_Portable_3rd    relevance=0.9258
  [2] Monster_Hunter_Freedom_Unite   relevance=0.8398
  [3] Monster_Hunter_Freedom_Unite   relevance=0.7891
  [4] Monster_Hunter                 relevance=0.6562
  [5] Monster_Hunter_Freedom_2       relevance=0.6328
```

<!-- Sources:
  Monster_Hunter_Portable_3rd    relevance 0.8[XX]
  Monster_Hunter                 relevance 0.7[XX] -->

## Pipeline

```
Wikipedia API
    -> ingest       One document per article from JSON files
    -> chunking     recursive character split with structural chunking + character based filtering
    -> vectorstore  Voyage embeddings, Chroma, content-hash deduplication
    -> retrieval    top-20 by vector similarity
    -> reranking    Voyage cross-encoder, top-20 -> top-5
    -> generation   Gemini, answering only from retrieved context
```

## Problems worth describing

### Section headings were causing identical hashing

Wikipedia API text marks sections with a bare heading on its own line rather than with
wiki markup. `RecursiveCharacterTextSplitter` splits on blank lines first, so a heading
landing on a chunk boundary became its own chunk containing nothing but the word
`Gameplay` or `Launch`.

~200 of ~2000 chunks (10%) were degenerate fragments under 80 characters. These carried no
information, consumed embedding tokens, and surfaced in retrieval as false matches
that displaced chunks actually containing answers.

Chunk IDs are SHA-256 hashes of source and content, so two identical headings in the same
article hashed identically and Chroma rejected the batch with a duplicate-ID error. Articles
covering multiple products were what triggered this: `Monster_Hunter_Tri` documents both Tri and
3 Ultimate, `PlayStation` covers several console revisions, each with repeated section
names.

Filtering chunks below 80 characters removed all fragments and eliminated the ID
collisions as a side effect.

## Setup

```bash
pip install -e .
```

Create a `.env` in the repository root containing API_KEY variables:

```
VOYAGE_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

Both have usable free tiers: 
- Voyage provides 200 million free tokens for embedding and re-ranking, much more than required in this project (2.3% used)
- Gemini's 3.5-flash free tier is rate limited, so use `--retrieval-only`
(below) when iterating on retrieval.

## Running

Interactive question answering:

```bash
python -m rag.pipeline
```

Retrieval only, showing which chunks are returned and their re-ranking relevance scores, without calling the
language model:

```bash
python -m rag.pipeline --retrieval-only
```

## Repo layout

```
rag/
    config.py       paths, models, and hyperparameters in one place
    ingest.py       loads the Wikipedia JSON corpus into Documents
    chunking.py     splitting and the minimum-length filter
    vectorstore.py  embeddings, Chroma, deduplication, retriever, reranker
    chain.py        prompt, language model, and the LCEL retrieval chain
    pipeline.py     orchestration and command-line entry point
    api.py          FastAPI serving layer
documents/          the Wikipedia corpus as JSON
notebooks/          exploration and retrieval inspection
tests/              unit tests for the pure-logic components
```

## Data

The corpus is currently 33 English Wikipedia articles covering the Monster Hunter series, its
individual titles and spin-offs, the consoles they released on, and a list of best-selling
video games. Text is fetched via the Wikipedia API rather than scraped or extracted from
PDFs, which avoids the table-mangling and reference-section noise that PDF extraction
introduces. Reference, External links, and See also sections are stripped at fetch time.

`documents/wikipedia_pages.json` maps article titles to their plain text. The vector store
in `chroma_db/` is generated from it and is not committed, since it is reproducible from
the corpus and the ingestion code. A `.gitkeep` file exists in the `chroma_db/` folder to retain the folder structure.

## Design decisions

**Content-hash chunk IDs.** Each chunk's ID is a SHA-256 hash of its source article and
text, so an unchanged chunk (same `CHUNK_SIZE`, `CHUNK_OVERLAP`) always produces the same ID. Re-running ingestion embeds only
chunks not already stored, which means adding one article to the corpus costs only that
article's embeddings rather than a full rebuild(300k tokens using `voyage-4-large` embedding model on current corpus size).

**Persistent vector store.** Chroma writes to disk, so embeddings survive process
restarts.

**Fixed-size chunking over semantic chunking.** Wikipedia articles are already cleanly
sectioned, so semantic chunking's advantage is modest here, while it costs an additional
embedding pass over every sentence to determine boundaries. The splitter is configured to
prefer section and paragraph boundaries before falling back to more arbitrary cuts.

**Retrieval and serving are separated.** `chain.py` contains no ingestion or storage
logic, so the serving layer imports it without triggering corpus loading. The chain is
built once at application startup rather than per request.

**Reranking over a larger candidate set.** The base retriever returns the 20 closest chunks by cosine similarity.
This is specifically so the reranker has enough candidates to work with. Reranking improves
precision within a candidate set but cannot recover a relevant chunk that the initial
retrieval missed entirely.

## Limitations and next steps

**No systematic evaluation yet.** The most useful addition would be a labelled set of
questions paired with expected answers, evaluated with the RAGAS framework to give
separate metrics for retrieval and generation quality. The main blocker is cost: RAGAS
uses an LLM as judge, and the number of calls required sits above Gemini's free-tier
rate limit. Implementing this would be also be time-intensive.

**Editing an article leaves stale chunks.** Content-hash IDs mean modified text produces
new chunks while the old versions remain in the store. The corpus is append-mostly, so
this has not mattered, but a periodic full rebuild would be needed for an evolving corpus.

**No test coverage.** A `test_pipeline.py` covering each component in isolation would
verify that loading, chunking, ID generation, and context formatting behave as expected,
and would catch regressions in the pure-logic parts without requiring API calls.

**Limited corpus scope.** The current sources cover development and commercial history
rather than in-game content. Expanding via the Fandom wiki API would add monsters,
weapons, and quest information, allowing questions about the games themselves rather than
just the metadata surrounding them.

**No caching layer.** Caching both common queries and their generated answers would avoid
repeat API calls, reducing latency and token consumption.

~~**No retrieval-only mode.** A flag to run retrieval without calling the LLM would allow
inspection of which chunks were returned and their relevance scores, making it possible to
diagnose retrieval quality without consuming generation quota.~~

**Chunk filtering can discard real content.** The minimum-length filter removes any chunk
under 80 characters, which catches stranded section headings but will also drop short
passages that carry genuine information. A more targeted rule, matching heading-like
segments rather than filtering purely on length, would avoid false positives.

~~**LLM outputs do not yet carry citations.** Chunk metadata includes the source article and the
reranker's relevance score, but the current chain discards both after formatting the
context.~~
