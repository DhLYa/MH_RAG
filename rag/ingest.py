from collections import defaultdict
from pathlib import Path
import json

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from .config import DOCS_DIR, PAGES_FILES

def load_documents(pages_files: list[Path] = PAGES_FILES) -> list[Document]:
    docs = []
    for path in pages_files:
        corpus = path.stem
        with open(path, "r", encoding="utf-8") as f:
            pages = json.load(f)
        docs.extend(
            Document(page_content=text, metadata={"source": title, "corpus": corpus})
            for title, text in pages.items()
        )
    return docs