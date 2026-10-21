# Monster Hunter RAG

A retrieval-augmented generation pipeline that answers questions about the Monster Hunter
game series using a corpus built from wikipedia articles, and content from the [Monster Hunter Wiki](https://monsterhunterwiki.org/wiki/Main_Page).
Questions are answered only from the retrieved source material. Each statement cites the numbered
passage it came from, only the cited sources are listed, and the system says "I don't know" rather
than falling back on the language model's own knowledge.

Built with LangChain, Voyage AI embeddings and reranking, BM25 keyword search, Chroma as a
persistent vector store, and Gemini 3.5 Flash-Lite for free generation with a high rate limit.

## Example

```
>What 4 monsters share a flagship title?

The four flagship monsters featured in Monster Hunter Generations (also known as Monster Hunter X) are Astalos, Gammoth, Mizutsune, and Glavenus [1].

Sources:
  [1] Monster_Hunter_Generations   relevance=0.7930 - wikipedia_pages
```

Five passages were retrieved; only the one the answer cites is listed. The API returns all five,
each marked `cited` true or false.

## Pipeline

```
Wikipedia API + Monster Hunter Wiki API
    -> ingest       One document per article from JSON files
    -> chunking     recursive character split at section, paragraph and sentence ends; heading-only chunks dropped
    -> vectorstore  Voyage embeddings, Chroma, content-hash deduplication
    -> retrieval    top-20 by vector similarity + top-20 by BM25 keywords, fused by reciprocal rank
    -> reranking    Voyage cross-encoder, up to 40 candidates -> top-5
    -> generation   Gemini, answering only from retrieved context with [n] citations
```

## Problems worth describing

### Empty section headings polluted the corpus
Headings were written out as lines of their own, so a section with no prose beneath it left a
bare heading behind. On monsterhunterwiki.org this was common: many sections hold only stat
tables or galleries, which the scraper strips, leaving runs of consecutive headings. The
splitter turned these into chunks with nothing to retrieve, in both corpora.

Filtering on the text alone cannot reliably tell a heading from a short sentence, so the fix
uses the page structure. When scraping the HTML, a heading is held back and written only once
a prose block follows it; if another heading arrives first, it replaces the held one. The
Wikipedia plain-text extracts have no HTML to use, so the chunker drops chunks made only of
headings, as described below.

### Short facts were lost to the length filter
The first fix for bare headings dropped every chunk under 80 characters. Of the 209 chunks it
removed, 190 were headings, but the rest were real sentences, some found nowhere else in the
corpus: "the Gore Magala is genderless", "Bleem! were subsequently forced to shut down in
November 2001". They were short because the splitter, when breaking a long paragraph at a
sentence boundary, attached the ". " separator to the next piece, leaving tails such as
". After Ratha comes to his senses..." as chunks of their own.

The splitter now keeps each separator with the sentence it ends, and the length filter was
replaced by one that recognises headings by shape: short lines with no punctuation. The first
version of that rule would have dropped all 471 monster infobox chunks, since lines like
"Weakest To: Thunder" are just as short and unpunctuated; counting a colon as content fixed it,
and a test now guards against it.

### Duplicate chunk IDs rejected on ingestion
Chunk IDs are a hash of each chunk's corpus, source page and text, so re-running ingestion
skips chunks already stored. When a page repeats a passage verbatim, two chunks produce the
same ID. Both passed the check against the store, since neither was stored yet, and Chroma
then rejected the entire batch for containing a duplicate ID. Ingestion now also tracks the
IDs seen within the batch and keeps only the first copy, which loses nothing because the
text is identical.

## Setup

```bash
conda create -n MH_RAG python=3.11
conda activate MH_RAG
pip install -r requirements.txt
```

For development, `requirements-dev.txt` adds pytest and ruff, which CI runs on every push.

Create a `.env` in the repository root containing API_KEY variables:

```
VOYAGE_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

Both have usable free tiers:
- Voyage provides 200 million free tokens for embedding and reranking. Building the full vector store costs ~620k tokens with `voyage-4-large`, 0.3% of the allowance, so it can be rebuilt many times over at no cost.
- Gemini 3.5 Flash-Lite's free tier is rate limited, so use `--retrieval-only`
(below) when iterating on retrieval.

## Running
All commands run from the repository root.

Interactive question answering: First run embeds the full corpus; later runs reuse the persisted store and starts immediately.

```bash
python -m rag.pipeline
```

Retrieval only, showing which chunks are returned and their re-ranking relevance scores, without calling the
language model:

```bash
python -m rag.pipeline --retrieval-only
```

Evaluation against the labelled questions in `evaluation/questions.json`. Retrieval is scored
by hit rate and MRR before and after reranking, using only Voyage calls; `--generate` also checks
each LLM answer for the expected terms, or for a refusal on out-of-scope questions:

```bash
python -m rag.evaluate
python -m rag.evaluate --generate --output evaluation/results/baseline.json
python -m rag.evaluate --retrieval vector     # compare against vector search alone
```

Tests and lint. The tests use fakes in place of the APIs, so they need no keys:

```bash
python -m pytest tests -q
ruff check rag tests
```

## Repo layout

```
rag/
    config.py       paths, models, and hyperparameters in one place
    ingest.py       loads the JSON corpus into Documents
    chunking.py     splitting and the heading-only filter
    vectorstore.py  embeddings, Chroma, deduplication, retriever, reranker
    chain.py        prompt, language model, and the LCEL retrieval chain
    pipeline.py     orchestration and command-line entry point
    api.py          FastAPI serving layer
    evaluate.py     retrieval and answer scoring on the labelled questions
documents/          the corpus as JSON files
evaluation/         labelled questions with expected source pages and answer terms
notebooks/
    build_corpus.ipynb  fetches both sources and writes the corpus JSON files
    archive/            earlier exploration notebooks, kept for reference
tests/              unit tests, with fakes standing in for the APIs
.github/workflows/  CI: ruff and pytest on every push
```

## Data

The corpus is composed from two sources totalling 505 documents, and is rebuilt
from source by running `notebooks/build_corpus.ipynb`. Monster Hunter Wiki pages
are selected by category; the Wikipedia articles are a fixed list of titles in
the notebook, since the consoles and sales list sit outside the series category.

The first is 33 English Wikipedia articles covering the Monster Hunter series,
its individual titles and spin-offs, the consoles they released on, and a list
of best-selling video games. This provides meta-information about the series.

The second is 472 pages from monsterhunterwiki.org, providing in-game content.
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

**Content-hash chunk IDs.** Each chunk's ID is a SHA-256 hash of its corpus, source article and
text, so an unchanged chunk (same `CHUNK_SIZE`, `CHUNK_OVERLAP`) always produces the same ID. Re-running 
ingestion embeds only chunks not already stored, which means adding one article to the corpus costs only that
article's embeddings rather than a full rebuild.

**Persistent vector store.** Chroma writes to disk, so embeddings survive process
restarts.

**Fixed-size chunking over semantic chunking.** All articles are already cleanly
sectioned, so semantic chunking's advantage is modest here, while it costs an additional
embedding pass over every sentence to determine boundaries. The splitter is configured to
prefer section and paragraph boundaries before falling back to more arbitrary cuts.

**Retrieval and serving are separated.** `chain.py` contains no ingestion or storage
logic, so the serving layer imports it without triggering corpus loading. The chain is
built once at application startup rather than per request.

**Reranking over a larger candidate set.** Retrieval returns up to 40 candidates so the
reranker has enough to work with. Reranking improves precision within a candidate set but
cannot recover a relevant chunk that the initial retrieval missed entirely.

**Hybrid retrieval.** Embeddings place a question near passages with similar meaning, which
fails for rare exact terms: asked which monster is called "Arushuberudo" in Japanese, vector
search ranked the Arkveld page 7th and the reranker dropped it. BM25 keyword search over the
same chunks finds such terms directly, and the two ranked lists are merged by reciprocal rank
fusion before reranking. On the evaluation set this raised reranked hit@1 from 0.96 to 0.98
and keyword questions from 0.80 to 1.00, with no category getting worse. The BM25 index is
built in memory from the stored chunks at startup. `HYBRID_SEARCH` in `config.py` turns it off.

## Limitations and next steps

**Evaluation is keyword-based and the question set is small.** `rag.evaluate` scores
retrieval against labelled source pages and checks answers for expected terms, which avoids
an LLM judge and stays within free tiers. Term matching cannot judge paraphrased answers: two
correct answers ("highly intelligent", "lacks any determinable sex") were scored wrong until
their labels accepted those wordings. An LLM-as-judge framework such as RAGAS would handle
this, at the cost of more calls than Gemini's free tier allows. With 60 questions, one
question is worth almost two points of accuracy, so small differences between runs are noise.

**No caching layer.** Caching both common queries and their generated answers would avoid
repeat API calls, reducing latency and token consumption.

## Completed

**Corpus rebuildable from source.** The rough ingestion code for the Wikipedia and
MonsterHunterWiki.org APIs was consolidated into `notebooks/build_corpus.ipynb`, which
writes both JSON files and reports what changed against the previous version. The
exploration notebooks it replaced are kept in `notebooks/archive/`.

**Sources returned with every answer.** The chain passes the reranked documents through
alongside the answer, so the CLI and API list each source's title, corpus and relevance
score rather than discarding them after formatting the context.

**Retrieval-only mode.** `--retrieval-only` shows the base and reranked results with their
scores without calling the LLM, so retrieval quality can be diagnosed without consuming
generation quota.

**Test coverage and CI.** The tests cover loading, chunking, ID generation, deduplication,
stale-chunk removal, context formatting, the API's responses and the evaluation scoring, all
without API calls. GitHub Actions runs them, with ruff, on every push.

**In-game content.** The original sources covered only development and commercial history.
Monster pages from monsterhunterwiki.org now allow questions about the games themselves
rather than just the metadata surrounding them.
