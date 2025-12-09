# scripts/generate_dummy_docs.py
from pathlib import Path

DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

texts = [
    "Deep learning is a subset of machine learning focused on neural networks.",
    "Support Vector Machines are supervised learning models for classification.",
    "Reinforcement learning is about training agents via rewards and penalties.",
    "Natural Language Processing is used to process and understand human language.",
]

for i in range(1, 101):
    content = texts[i % len(texts)] + f" This is sample document number {i}."
    (DOCS_DIR / f"doc_{i:03d}.txt").write_text(content, encoding="utf-8")

print("Generated 100 dummy docs in data/docs/")
