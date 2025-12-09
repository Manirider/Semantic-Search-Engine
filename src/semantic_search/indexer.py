
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json
import time

import numpy as np
from tqdm import tqdm

from .config import DATA_DIR, INDEX_DIR, EMBEDDINGS_PATH, METADATA_PATH
from .loader import load_text_documents
from .embeddings import embed_texts

def build_index(
    docs_dir: Path = DATA_DIR,
    index_dir: Path = INDEX_DIR,
) -> None:
    """
    Build the semantic search index:
    - load .txt docs
    - create embeddings
    - save embeddings + metadata to disk
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading documents from: {docs_dir}")
    docs = load_text_documents(docs_dir)
    doc_ids = [doc_id for doc_id, _ in docs]
    texts = [text for _, text in docs]

    print(f"[INFO] Loaded {len(docs)} documents")

    start = time.perf_counter()
    print("[INFO] Generating embeddings...")
    embeddings = embed_texts(texts)
    elapsed = time.perf_counter() - start
    print(f"[INFO] Embeddings generated in {elapsed:.2f} seconds")

    # Save embeddings
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"[INFO] Saved embeddings to {EMBEDDINGS_PATH}")

    # Save metadata: mapping index -> doc_id and optional snippet
    metadata: Dict[str, Any] = {
        "docs": [
            {
                "id": doc_id,
                "filename": doc_id,
                "snippet": make_snippet(texts[i]),
            }
            for i, doc_id in enumerate(doc_ids)
        ],
        "num_documents": len(docs),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved metadata to {METADATA_PATH}")
    print("[INFO] Index build COMPLETE.")

def make_snippet(text: str, max_chars: int = 200) -> str:
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
