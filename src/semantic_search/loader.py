
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import os

def load_text_documents(directory: Path) -> List[Tuple[str, str]]:
    """
    Load all .txt documents from a directory.

    Returns:
        List of (doc_id, text), where doc_id is typically the filename.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    docs: List[Tuple[str, str]] = []
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".txt":
            try:
                text = entry.read_text(encoding="utf-8", errors="ignore")
                doc_id = entry.name
                docs.append((doc_id, text))
            except OSError as e:
                # Professional: don't crash on one bad file
                print(f"[WARN] Failed to read {entry}: {e}")

    if len(docs) == 0:
        raise ValueError(f"No .txt files found in directory: {directory}")

    return docs
