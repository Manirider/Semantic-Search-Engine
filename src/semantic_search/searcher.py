
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    EMBEDDINGS_PATH,
    METADATA_PATH,
    DEFAULT_TOP_K,
)
from .embeddings import embed_single

@dataclass
class SearchResult:
    doc_id: str
    score: float
    snippet: str

class SemanticSearchEngine:
    def __init__(
        self,
        embeddings_path: Path = EMBEDDINGS_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}. "
                f"Did you run the index build step?"
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                f"Did you run the index build step?"
            )

        self.embeddings = np.load(embeddings_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.doc_entries = self.metadata["docs"]
        if self.embeddings.shape[0] != len(self.doc_entries):
            raise ValueError(
                "Mismatch between number of embeddings and metadata entries:"
                f" {self.embeddings.shape[0]} vs {len(self.doc_entries)}"
            )

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[SearchResult]:
        """Return top_k most similar documents for a string query."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        query_vec = embed_single(query).reshape(1, -1)

        # Since we normalized embeddings, cosine similarity == dot product
        scores = cosine_similarity(query_vec, self.embeddings)[0]  # shape (num_docs,)

        # Get indices of top_k highest scores
        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        # Sort those indices by score descending
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[SearchResult] = []
        for idx in top_indices:
            entry = self.doc_entries[int(idx)]
            results.append(
                SearchResult(
                    doc_id=entry["id"],
                    score=float(scores[idx]),
                    snippet=entry["snippet"],
                )
            )
        return results

    def search_batch(
        self, queries: List[str], top_k: int = DEFAULT_TOP_K
    ) -> List[List[SearchResult]]:
        """Run semantic search for multiple queries."""
        return [self.search(q, top_k=top_k) for q in queries]
