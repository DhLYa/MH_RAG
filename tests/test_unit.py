import json

import pytest
from langchain_core.documents import Document

from rag.chain import build_prompt, format_docs
from rag.chunking import chunk_documents
from rag.config import MIN_CHUNK_CHARS
from rag.ingest import load_documents
from rag.vectorstore import chunk_id


def make_doc(text, source="Rathalos", corpus="mh_wiki_monsters"):
    return Document(page_content=text, metadata={"source": source, "corpus": corpus})


# --------------------------------------------------------------------------
# chunk_id
# --------------------------------------------------------------------------

def test_chunk_id_is_stable():
    """Identical inputs must always has the same to prevent reingestion of same content."""
    assert chunk_id(make_doc("hello world")) == chunk_id(make_doc("hello world"))


def test_chunk_id_differs_on_content():
    assert chunk_id(make_doc("red shell")) != chunk_id(make_doc("blue shell"))


def test_chunk_id_differs_on_source():
    """Same text on different pages must hash differently."""
    a = make_doc("Gameplay details", source="Rathalos")
    b = make_doc("Gameplay details", source="Rathian")
    assert chunk_id(a) != chunk_id(b)


def test_chunk_id_differs_on_corpus():
    """Same title in 2 different corpos must hash differently."""
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
    """Chunks split far from page opening lose context on the source title lreacing them 
    unmatachble through queries naming them. The prefix mustn't be doubled which happened 
    during development as both chunker and scraper added this"""
    doc = make_doc("Volcanic Leviathans covered in fins. " * 80, source="Agnaktor")
    for chunk in chunk_documents([doc]):
        assert chunk.page_content.startswith("Source: Agnaktor\n\n")
        assert chunk.page_content.count("Source: Agnaktor") == 1


def test_distinct_prose_yields_distinct_ids():
    """Chroma rejects a batch containing duplicate IDs."""
    paragraphs = []
    for i in range(6):
        f"Paragraph {i} describes a different aspect of the monster in detail. " * 12
    doc = make_doc("\n\n".join(paragraphs))
    ids = [chunk_id(c) for c in chunk_documents([doc])]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# load_documents
# --------------------------------------------------------------------------

@pytest.fixture
def corpus_files(tmp_path):
    """Two small corpus files in the same title -> text shape as the real ones."""
    """Two small corpus files for testing in the same format as Wikipedia and MonsterHunterWiki"""
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