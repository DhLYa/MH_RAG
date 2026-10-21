import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS

HEADING_MAX_CHARS = 40
# Sentence endings, plus ":" so infobox lines such as "Weakest To: Thunder" count as content
CONTENT_PUNCTUATION = re.compile(r"[.!?:]")

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        # Keep each ". " with the sentence it ends. The default attaches it to the next piece,
        # leaving short tails such as ". After Ratha comes to his senses..." as their own chunks
        keep_separator="end",
    )

def is_heading_only(text: str) -> bool:
    """True for chunks made only of section headings, e.g. "Gameplay" or "Plot\\nRise arc".

    Headings are short lines without punctuation; prose and label-value lines, however
    short, have it. Filtering on length alone also dropped short sentences with real facts.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line or (line.startswith("==") and line.endswith("==")):
            continue
        if len(line) > HEADING_MAX_CHARS or CONTENT_PUNCTUATION.search(line):
            return False
    return True

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = get_splitter()
    chunks = splitter.split_documents(documents)
    kept = []
    for c in chunks:
        text = c.page_content.strip()
        if len(text) >= MIN_CHUNK_CHARS and not is_heading_only(text):
            kept.append(c)
    for c in kept:
        c.page_content = f"Source: {c.metadata['source']}\n\n{c.page_content}"
    return kept
