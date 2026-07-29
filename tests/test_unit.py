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