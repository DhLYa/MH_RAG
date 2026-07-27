from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGES_FILES = [
    Path("documents/wikipedia_pages.json"),
    Path("documents/mh_wiki_monsters.json"),
]
DOCS_DIR = Path("documents")
CHROMA_DIR = Path("chroma_db")
CHUNK_CACHE_DIR = ""

EMBEDDING_MODEL = "voyage-4-large"
RERANKER_MODEL = "rerank-2.5"
LLM_MODEL = "gemini-3.5-flash"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 80

RETRIEVAL_K = 20
RERANK_K = 5