
from pathlib import Path

# Base directories – can be overridden via CLI args if needed
BASE_DIR = Path(__file__).resolve().parents[2]  # repo root

DATA_DIR = BASE_DIR / "data" / "docs"
INDEX_DIR = BASE_DIR / "index"

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Search defaults
DEFAULT_TOP_K = 5
EMBED_BATCH_SIZE = 32
SNIPPET_MAX_CHARS = 200
