from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = get_splitter()
    chunks = splitter.split_documents(documents)
    kept = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    for c in kept:
        c.page_content = f"Source: {c.metadata['source']}\n\n{c.page_content}"
    return kept