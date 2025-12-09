import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------- Data Classes for Metadata ----------

@dataclass
class DocumentMetadata:
    id: str
    filename: str
    path: str


@dataclass
class IndexMetadata:
    model_name: str
    created_at: str
    docs_dir: str
    num_documents: int
    documents: List[DocumentMetadata]


# ---------- Utility Functions ----------

def read_text_file(file_path: str, encoding: str = "utf-8") -> str:
    """Reads a text file safely, returning empty string on failure."""
    try:
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] Failed to read {file_path}: {e}", file=sys.stderr)
        return ""


def iter_text_files(docs_dir: str) -> List[str]:
    """Return a list of absolute paths to all .txt files in docs_dir (recursive)."""
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")

    file_paths = []
    for root, _, files in os.walk(docs_dir):
        for fname in files:
            if fname.lower().endswith(".txt"):
                file_paths.append(os.path.join(root, fname))

    if not file_paths:
        raise ValueError(f"No .txt files found in directory: {docs_dir}")

    return sorted(file_paths)


def load_model(model_name: str) -> SentenceTransformer:
    """Load a sentence transformer model."""
    print(f"[INFO] Loading model: {model_name}")
    start = time.perf_counter()
    model = SentenceTransformer(model_name)
    elapsed = time.perf_counter() - start
    print(f"[INFO] Model loaded in {elapsed:.2f} seconds.")
    return model


def compute_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32
) -> np.ndarray:
    """Compute embeddings for a list of texts in batches."""
    print(f"[INFO] Computing embeddings for {len(texts)} documents...")
    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll rely on cosine_similarity to handle this
    )
    elapsed = time.perf_counter() - start
    print(f"[INFO] Embeddings computed in {elapsed:.2f} seconds.")
    return embeddings


def build_index(
    docs_dir: str,
    index_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> None:
    """Ingest documents, compute embeddings, and save index to disk."""
    os.makedirs(index_dir, exist_ok=True)

    # 1. Collect document paths
    file_paths = iter_text_files(docs_dir)
    print(f"[INFO] Found {len(file_paths)} text documents in {docs_dir}.")

    # 2. Read documents
    docs_content = []
    doc_metas: List[DocumentMetadata] = []
    for idx, path in enumerate(tqdm(file_paths, desc="Reading documents")):
        content = read_text_file(path)
        if not content.strip():
            print(f"[WARN] Document is empty or unreadable, skipping: {path}", file=sys.stderr)
            continue
        docs_content.append(content)
        doc_id = f"doc_{idx:05d}"
        doc_metas.append(
            DocumentMetadata(
                id=doc_id,
                filename=os.path.basename(path),
                path=os.path.abspath(path),
            )
        )

    if not docs_content:
        raise ValueError("No valid documents to index.")

    # 3. Load model and compute embeddings
    model = load_model(model_name)
    embeddings = compute_embeddings(model, docs_content, batch_size=batch_size)

    # 4. Save embeddings to NumPy file
    embeddings_path = os.path.join(index_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"[INFO] Saved embeddings to {embeddings_path}")

    # 5. Save metadata to JSON
    metadata = IndexMetadata(
        model_name=model_name,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        docs_dir=os.path.abspath(docs_dir),
        num_documents=len(doc_metas),
        documents=doc_metas,
    )

    # Convert dataclasses to plain dict for JSON
    meta_dict = asdict(metadata)
    metadata_path = os.path.join(index_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"[INFO] Saved metadata to {metadata_path}")

    print(f"[INFO] Indexing complete. {len(doc_metas)} documents indexed.")


def load_index(index_dir: str) -> Tuple[np.ndarray, IndexMetadata]:
    """Load embeddings and metadata from disk."""
    embeddings_path = os.path.join(index_dir, "embeddings.npy")
    metadata_path = os.path.join(index_dir, "metadata.json")

    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    embeddings = np.load(embeddings_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    # Reconstruct IndexMetadata + DocumentMetadata objects
    docs_meta = [
        DocumentMetadata(**doc_dict)
        for doc_dict in meta_raw["documents"]
    ]
    metadata = IndexMetadata(
        model_name=meta_raw["model_name"],
        created_at=meta_raw["created_at"],
        docs_dir=meta_raw["docs_dir"],
        num_documents=meta_raw["num_documents"],
        documents=docs_meta,
    )

    print(f"[INFO] Loaded index from {index_dir}.")
    print(f"[INFO] {metadata.num_documents} documents, embeddings shape: {embeddings.shape}")
    return embeddings, metadata


def generate_snippet(path: str, max_chars: int = 200) -> str:
    """Return the first max_chars characters of the document as a snippet."""
    text = read_text_file(path)
    snippet = text[:max_chars].replace("\n", " ").strip()
    return snippet


def search(
    index_dir: str,
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Run semantic search for a single query; return top-k results."""
    if not query.strip():
        raise ValueError("Query is empty.")

    embeddings, metadata = load_index(index_dir)
    model = load_model(metadata.model_name)

    # 1. Compute query embedding
    print(f"[INFO] Computing embedding for query: {query!r}")
    start = time.perf_counter()
    query_vec = model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    query_vec = query_vec.reshape(1, -1)

    # 2. Compute cosine similarity
    sims = cosine_similarity(query_vec, embeddings)[0]
    elapsed = time.perf_counter() - start
    print(f"[INFO] Search computation completed in {elapsed:.4f} seconds.")

    # 3. Get top-k indices
    top_k = min(top_k, len(sims))
    top_indices = np.argsort(-sims)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        doc_meta = metadata.documents[idx]
        score = float(sims[idx])
        snippet = generate_snippet(doc_meta.path, max_chars=200)
        results.append(
            {
                "rank": rank,
                "doc_id": doc_meta.id,
                "filename": doc_meta.filename,
                "path": doc_meta.path,
                "score": score,
                "snippet": snippet,
            }
        )

    return results


def print_search_results(query: str, results: List[Dict[str, Any]]) -> None:
    """Nicely format the search results to stdout."""
    print("\n" + "=" * 80)
    print(f"Query: {query}")
    print("=" * 80)
    for r in results:
        print(f"[{r['rank']}] {r['filename']} (ID: {r['doc_id']})")
        print(f"     Score: {r['score']:.4f}")
        print(f"     Path : {r['path']}")
        print(f"     Snip : {r['snippet']}")
        print("-" * 80)
    print()


def batch_search(
    index_dir: str,
    queries_file: str,
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run semantic search for multiple queries listed line-by-line in a file.

    Lines starting with '#' or empty lines are skipped.
    Returns a dict mapping query -> results list.
    """
    if not os.path.isfile(queries_file):
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    embeddings, metadata = load_index(index_dir)
    model = load_model(metadata.model_name)

    # Load queries
    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            queries.append(q)

    if not queries:
        raise ValueError("No valid queries found in queries file.")

    print(f"[INFO] Running batch search for {len(queries)} queries...")

    # Compute all query embeddings in one batch for efficiency
    start = time.perf_counter()
    query_embeddings = model.encode(
        queries,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    # Compute cosine similarities: shape [num_queries, num_docs]
    similarities = cosine_similarity(query_embeddings, embeddings)
    elapsed = time.perf_counter() - start
    print(f"[INFO] Batch search computation completed in {elapsed:.4f} seconds.")

    results_per_query: Dict[str, List[Dict[str, Any]]] = {}

    for qi, query in enumerate(queries):
        sims = similarities[qi]
        k = min(top_k, len(sims))
        top_indices = np.argsort(-sims)[:k]
        query_results = []
        for rank, idx in enumerate(top_indices, start=1):
            doc_meta = metadata.documents[idx]
            score = float(sims[idx])
            snippet = generate_snippet(doc_meta.path, max_chars=200)
            query_results.append(
                {
                    "rank": rank,
                    "doc_id": doc_meta.id,
                    "filename": doc_meta.filename,
                    "path": doc_meta.path,
                    "score": score,
                    "snippet": snippet,
                }
            )
        results_per_query[query] = query_results

    return results_per_query


# ---------- CLI Interface ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic search over a collection of text documents."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # index command
    index_parser = subparsers.add_parser(
        "index",
        help="Build the semantic search index from a directory of .txt files.",
    )
    index_parser.add_argument(
        "--docs_dir",
        type=str,
        required=True,
        help="Directory containing .txt documents.",
    )
    index_parser.add_argument(
        "--index_dir",
        type=str,
        default="index",
        help="Directory to store index files (embeddings + metadata).",
    )
    index_parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformer model name.",
    )
    index_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )

    # search command (single query)
    search_parser = subparsers.add_parser(
        "search",
        help="Run semantic search for a single query.",
    )
    search_parser.add_argument(
        "--index_dir",
        type=str,
        default="index",
        help="Directory containing index files.",
    )
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query string.",
    )
    search_parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return.",
    )

    # batch-search command
    batch_parser = subparsers.add_parser(
        "batch-search",
        help="Run semantic search for multiple queries from a file.",
    )
    batch_parser.add_argument(
        "--index_dir",
        type=str,
        default="index",
        help="Directory containing index files.",
    )
    batch_parser.add_argument(
        "--queries_file",
        type=str,
        required=True,
        help="Text file with one query per line.",
    )
    batch_parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return.",
    )
    batch_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results in JSONL format.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "index":
        start = time.perf_counter()
        try:
            build_index(
                docs_dir=args.docs_dir,
                index_dir=args.index_dir,
                model_name=args.model,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"[ERROR] Indexing failed: {e}", file=sys.stderr)
            sys.exit(1)
        elapsed = time.perf_counter() - start
        print(f"[BENCHMARK] Total indexing time: {elapsed:.2f} seconds.")

    elif args.command == "search":
        start = time.perf_counter()
        try:
            results = search(
                index_dir=args.index_dir,
                query=args.query,
                top_k=args.top_k,
            )
            print_search_results(args.query, results)
        except Exception as e:
            print(f"[ERROR] Search failed: {e}", file=sys.stderr)
            sys.exit(1)
        elapsed = time.perf_counter() - start
        print(f"[BENCHMARK] Total search time (including I/O + model load): {elapsed:.4f} seconds.")

    elif args.command == "batch-search":
        start = time.perf_counter()
        try:
            all_results = batch_search(
                index_dir=args.index_dir,
                queries_file=args.queries_file,
                top_k=args.top_k,
            )

            # Print results to stdout
            for query, results in all_results.items():
                print_search_results(query, results)

            # Optionally write to JSONL
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    for query, results in all_results.items():
                        record = {
                            "query": query,
                            "results": results,
                        }
                        f.write(json.dumps(record) + "\n")
                print(f"[INFO] Batch results saved to {args.output}")

        except Exception as e:
            print(f"[ERROR] Batch search failed: {e}", file=sys.stderr)
            sys.exit(1)
        elapsed = time.perf_counter() - start
        print(f"[BENCHMARK] Total batch search time: {elapsed:.4f} seconds.")

    else:
        print("[ERROR] Unknown command.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
