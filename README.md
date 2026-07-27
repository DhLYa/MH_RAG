# Monster Hunter RAG

A retrieval-augmented generation pipeline that answers questions about the Monster Hunter
game series using a corpus built from wikipedia articles, and content from the [Monster Hunter Wiki](https://monsterhunterwiki.org/wiki/Main_Page).
Questions are answered only from the retrieved source material, so the system cites where each
answer came from and says "I don't know" rather than falling back on the language model's own knowledge.

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

### Empty section headings polluted the corpus
Write-up pending. Summary: bare heading lines became content-free chunks in both
corpora; fixed at scrape time using HTML tag structure rather than text heuristics

### Duplicate chunk IDs rejected on ingestion
Write-up pending. Summary: content-hash IDs collide when a page repeats a
paragraph verbatim; ingestion now deduplicates within the batch as well as
against the store.

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
- Voyage provides 200 million free tokens for embedding and reranking. Building the full vector store costs ~620k tokens with `voyage-4-large`, 0.3% of the allowance, so it can be rebuilt many times over at no cost.
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

The corpus is composed from two sources totalling 502 documents. Pages from both
are selected by category rather than listed by hand, so the corpus is
reproducible from the category names.

The first is 33 English Wikipedia articles covering the Monster Hunter series,
its individual titles and spin-offs, the consoles they released on, and a list
of best-selling video games. This provides meta-information about the series.

The second is 469 pages from monsterhunterwiki.org, providing in-game content.
The two sources answer different question types, and chunks record which corpus
they came from so retrieval results can be attributed and filtered.

Both are fetched via the MediaWiki API. Wikipedia supports plain-text extracts
directly, whereas monsterhunterwiki.org pages are requested as rendered HTML and
reduced to prose by stripping empty headings and non-textual elements. Infobox
tables are read separately into label-value lines before the surrounding tables
are discarded, since they hold attributes worth retaining such as
classification, elements, and elemental weaknesses. Those values are rendered as
icons whose names appear on the parent link rather than in the image, so they
are recovered from the link rather than the image text.

Reference, External links, and See also sections are stripped at fetch time,
along with headings left empty once their tables were removed, MediaWiki
citation errors, and stub pages whose only content was the legend explaining the
stat tables.

`documents/wikipedia_pages.json` and `documents/mh_wiki_monsters.json` map page
titles to their plain text. The vector store in `chroma_db/` is generated from
them and is not committed, since it is reproducible from the corpus and the
ingestion code. A `.gitkeep` file retains the folder structure.

## Design decisions

**Content-hash chunk IDs.** Each chunk's ID is a SHA-256 hash of its source article and
text, so an unchanged chunk (same `CHUNK_SIZE`, `CHUNK_OVERLAP`) always produces the same ID. Re-running ingestion embeds only
chunks not already stored, which means adding one article to the corpus costs only that
article's embeddings rather than a full rebuild.

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

**No test coverage.** A `test_pipeline.py` covering each component in isolation would
verify that loading, chunking, ID generation, and context formatting behave as expected,
and would catch regressions in the pure-logic parts without requiring API calls.

~~**Limited corpus scope.** The current sources cover development and commercial history
rather than in-game content. Expanding via the monster hunter wiki API would add monsters,
weapons, and quest information, allowing questions about the games themselves rather than
just the metadata surrounding them.~~

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
