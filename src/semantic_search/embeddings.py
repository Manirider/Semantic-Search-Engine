
from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import EMBEDDING_MODEL_NAME, EMBED_BATCH_SIZE

_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def embed_texts(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Returns:
        numpy.ndarray with shape (len(texts), embedding_dim)
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes cosine similarity faster/cleaner
    )
    return embeddings

def embed_single(text: str) -> np.ndarray:
    """Convenience wrapper to embed a single text into a 1D vector."""
    return embed_texts([text])[0]
