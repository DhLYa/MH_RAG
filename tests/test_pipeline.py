import json

import pytest
from langchain_core.documents import Document

from rag.chain import ask, build_prompt, format_docs
from rag.chunking import chunk_documents
from rag.config import MIN_CHUNK_CHARS
from rag.ingest import load_documents
from rag.vectorstore import chunk_id, is_empty, remove_stale_chunks, upsert_chunks


def make_doc(text, source="Rathalos", corpus="mh_wiki_monsters"):
    return Document(page_content=text, metadata={"source": source, "corpus": corpus})


# --------------------------------------------------------------------------
# chunk_id
# --------------------------------------------------------------------------

def test_chunk_id_is_stable():
    """Identical inputs must always have the same hash to prevent reingestion of same content."""
    assert chunk_id(make_doc("hello world")) == chunk_id(make_doc("hello world"))


def test_chunk_id_differs_on_content():
    assert chunk_id(make_doc("red shell")) != chunk_id(make_doc("blue shell"))


def test_chunk_id_differs_on_source():
    """Same text on different pages must hash differently."""
    a = make_doc("Gameplay details", source="Rathalos")
    b = make_doc("Gameplay details", source="Rathian")
    assert chunk_id(a) != chunk_id(b)


def test_chunk_id_differs_on_corpus():
    """Same title in 2 different corpora must hash differently."""
    a = make_doc("Shared text", source="Monster Hunter", corpus="wikipedia_pages")
    b = make_doc("Shared text", source="Monster Hunter", corpus="mh_wiki_monsters")
    assert chunk_id(a) != chunk_id(b)


# --------------------------------------------------------------------------
# chunk_documents
# --------------------------------------------------------------------------

def test_no_chunk_below_minimum_length():
    """Bare section headings in wikipedia corpus became content-less chunks, the filter removes them."""
    doc = make_doc("Physiology\n\n" + "The Agnaktor is a magmatic Leviathan. " * 40)
    for chunk in chunk_documents([doc]):
        body = chunk.page_content.split("\n\n", 1)[1]
        assert len(body.strip()) >= MIN_CHUNK_CHARS


def test_bare_heading_does_not_survive_as_a_chunk():
    doc = make_doc("Physiology\n\n" + "The Agnaktor is a magmatic Leviathan. " * 40)
    bodies = [c.page_content.split("\n\n", 1)[1].strip() for c in chunk_documents([doc])]
    assert "Physiology" not in bodies


def test_metadata_survives_chunking():
    """chunk_id and citations both read this metadata. Ingestion would break without it surviving."""
    doc = make_doc("The Agnaktor is a magmatic Leviathan. " * 60, source="Agnaktor")
    chunks = chunk_documents([doc])
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["source"] == "Agnaktor"
        assert chunk.metadata["corpus"] == "mh_wiki_monsters"


def test_every_chunk_carries_the_source_prefix_exactly_once():
    """Chunks split far from page opening lose context on the source title making them
    unmatchable through queries naming them. The prefix mustn't be doubled which happened 
    during development as both chunker and scraper added this"""
    doc = make_doc("Volcanic Leviathans covered in fins. " * 80, source="Agnaktor")
    for chunk in chunk_documents([doc]):
        assert chunk.page_content.startswith("Source: Agnaktor\n\n")
        assert chunk.page_content.count("Source: Agnaktor") == 1


def test_distinct_prose_yields_distinct_ids():
    """Chroma rejects a batch containing duplicate IDs."""
    paragraphs = []
    for i in range(6):
        paragraphs.append(f"Paragraph {i} describes a different aspect of the monster in detail. " * 12)
    doc = make_doc("\n\n".join(paragraphs))
    ids = [chunk_id(c) for c in chunk_documents([doc])]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# load_documents
# --------------------------------------------------------------------------

@pytest.fixture
def corpus_files(tmp_path):
    """Two small corpus files in the same title -> text format as the Wikipedia and Monster Hunter Wiki ones."""
    wiki = tmp_path / "wikipedia_pages.json"
    wiki.write_text(
        json.dumps({"Monster Hunter": "Monster Hunter is a game series. " * 8}),
        encoding="utf-8",
    )
    monsters = tmp_path / "mh_wiki_monsters.json"
    monsters.write_text(
        json.dumps({
            "Rathalos": "Rathalos is a Flying Wyvern. " * 8,
            "Agnaktor": "Agnaktor is a Leviathan. " * 8,
        }),
        encoding="utf-8",
    )
    return [wiki, monsters]


def test_load_documents_returns_one_per_page(corpus_files):
    assert len(load_documents(corpus_files)) == 3


def test_load_documents_sets_source_from_title(corpus_files):
    sources = {d.metadata["source"] for d in load_documents(corpus_files)}
    assert sources == {"Monster Hunter", "Rathalos", "Agnaktor"}


def test_load_documents_sets_corpus_from_filename(corpus_files):
    by_source = {d.metadata["source"]: d.metadata["corpus"] for d in load_documents(corpus_files)}
    assert by_source["Monster Hunter"] == "wikipedia_pages"
    assert by_source["Rathalos"] == "mh_wiki_monsters"
    assert by_source["Agnaktor"] == "mh_wiki_monsters"

# --------------------------------------------------------------------------
# format_docs
# --------------------------------------------------------------------------
 
def test_format_docs_numbers_blocks_from_one():
    """Check if formatted docs start with source [1]."""
    out = format_docs([make_doc("first"), make_doc("second")])
    assert out.startswith("[1]")
    assert "[2]" in out
 
 
def test_format_docs_labels_each_block_with_its_source():
    """Without source label the LLM output has nothing to cite."""
    out = format_docs([make_doc("text", source="Rathalos")])
    assert "Source: Rathalos" in out
 
 
def test_format_docs_separates_blocks_with_a_blank_line():
    """Verify format when printing sources is correct"""
    out = format_docs([make_doc("first"), make_doc("second")])
    assert "\n\n" in out
 
 
def test_format_docs_handles_empty_input():
    """Check if empty input is handled correctly"""
    assert format_docs([]) == ""

# --------------------------------------------------------------------------
# build_prompt
# --------------------------------------------------------------------------
 
def test_prompt_fills_both_placeholders():
    """Verify prompt template is setup correctly to accept context and question."""
    filled = str(build_prompt().invoke({"context": "CONTEXT_HERE", "question": "QUESTION_HERE"}))
    assert "CONTEXT_HERE" in filled
    assert "QUESTION_HERE" in filled
# --------------------------------------------------------------------------
# upsert_chunks / remove_stale_chunks
# --------------------------------------------------------------------------

class FakeStore:
    """Stands in for Chroma: an id -> Document map with the calls the pipeline uses."""
    def __init__(self, docs=()):
        self.docs = {chunk_id(d): d for d in docs}
        self.delete_calls = []

    def get(self, include=None, limit=None):
        ids = list(self.docs)
        return {"ids": ids[:limit] if limit else ids}

    def add_documents(self, documents, ids):
        assert len(ids) == len(set(ids)), "Chroma rejects duplicate IDs in a batch"
        self.docs.update(zip(ids, documents))

    def delete(self, ids):
        self.delete_calls.append(ids)
        for i in ids:
            del self.docs[i]


def test_upsert_skips_stored_and_repeated_chunks():
    stored = make_doc("already embedded")
    store = FakeStore([stored])
    new = make_doc("new paragraph")
    assert upsert_chunks(store, [stored, new, make_doc("new paragraph")]) == 1
    assert set(store.docs) == {chunk_id(stored), chunk_id(new)}


def test_remove_stale_chunks_deletes_only_chunks_no_longer_produced():
    """Changing CHUNK_SIZE or re-scraping leaves old chunks behind that would still be retrieved."""
    kept, stale = make_doc("still current"), make_doc("old chunking")
    store = FakeStore([kept, stale])
    assert remove_stale_chunks(store, [kept]) == 1
    assert set(store.docs) == {chunk_id(kept)}


def test_remove_stale_chunks_is_a_no_op_when_in_sync():
    docs = [make_doc("a"), make_doc("b")]
    store = FakeStore(docs)
    assert remove_stale_chunks(store, docs) == 0
    assert store.delete_calls == []


def test_remove_stale_chunks_deletes_in_batches():
    """Chroma caps the number of IDs per call."""
    store = FakeStore([make_doc(f"chunk {i}") for i in range(5)])
    assert remove_stale_chunks(store, [], batch_size=2) == 5
    assert [len(c) for c in store.delete_calls] == [2, 2, 1]
    assert store.docs == {}


def test_is_empty():
    assert is_empty(FakeStore())
    assert not is_empty(FakeStore([make_doc("x")]))

# --------------------------------------------------------------------------
# ask
# --------------------------------------------------------------------------

class FakeChain:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, question):
        return {"answer": "I don't know.", "docs": self.docs}


def test_ask_handles_no_retrieved_docs():
    out = ask(FakeChain([]), "anything")
    assert out.startswith("I don't know.")
    assert "Sources: none retrieved" in out


def test_ask_handles_missing_relevance_score():
    out = ask(FakeChain([make_doc("text", source="Rathalos")]), "anything")
    assert "[1] Rathalos" in out
    assert "relevance=n/a" in out


def test_ask_formats_relevance_score():
    doc = make_doc("text", source="Rathalos")
    doc.metadata["relevance_score"] = 0.80859
    assert "relevance=0.8086 - mh_wiki_monsters" in ask(FakeChain([doc]), "anything")
