from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# The repository root, so paths resolve the same wherever the code is run from
ROOT = Path(__file__).resolve().parent.parent

PAGES_FILES = [
    ROOT / "documents" / "wikipedia_pages.json",
    ROOT / "documents" / "mh_wiki_monsters.json",
]
# Each chunk's "corpus" metadata is its file's name, so these are the values retrieval can filter on
CORPORA = tuple(path.stem for path in PAGES_FILES)
CHROMA_DIR = ROOT / "chroma_db"
EVAL_QUESTIONS_FILE = ROOT / "evaluation" / "questions.json"

EMBEDDING_MODEL = "voyage-4-large"
RERANKER_MODEL = "rerank-2.5"
LLM_MODEL = "gemini-3.5-flash-lite"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 10    # floor for stray fragments; headings are filtered by shape, not length

RETRIEVAL_K = 20
HYBRID_SEARCH = True    # fuse BM25 keyword results with vector results before reranking
BM25_K = 20
RERANK_K = 5