**Semantic Search Engine**

Author & Background
Developed by:
S.Manikanta Suryasai
This project is a reflection of my continuous effort to learn, build, and deploy industry-relevant AI systems.


A lightweight, production-style semantic search system built using Sentence Transformers, NumPy, and cosine similarity.

This project implements an end-to-end semantic search engine capable of finding conceptually similar documents rather than relying on keyword matching. It uses pretrained transformer embeddings, persistent indexing, and a clean command-line interface for searching both single queries and batch queries.

This repository also demonstrates the fundamentals of building modern information retrieval systems and serves as a foundation for RAG (Retrieval-Augmented Generation) pipelines.

 
**-Features-**
✅ Document Ingestion

Ingests .txt documents from any folder

Extracts ID + snippet automatically

Handles 100+ documents smoothly

✅ Embeddings + Indexing

Uses all-MiniLM-L6-v2 from Sentence Transformers

Embeddings are normalized for faster cosine similarity

Saves index as:

index/embeddings.npy

index/metadata.json

✅ Semantic Search

Query encoder → vector representation

Computes cosine similarity across all documents

Returns top-k most similar documents (ranked)

✅ CLI Tool

Commands include:

index – Build the index

search – Single query

search-batch – File with multiple queries

✅ Professional Codebase

Modular components: loader, embeddings, indexer, searcher, CLI

Batch embedding for memory efficiency

Error handling for missing directories

Easily extendable (FAISS, FastAPI, Streamlit UI)

**-Project Structure-**

semantic-search-engine/
├── README.md
├── requirements.txt
├── data/
│   ├── docs/                  # Your text documents (100+)
│   └── queries.txt            # Example batch queries
├── index/                     # Auto-generated index files
├── scripts/
│   ├── generate_dummy_docs.py
│   └── benchmark_search.py
├── src/
│   └── semantic_search/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── loader.py
│       ├── embeddings.py
│       ├── indexer.py
│       ├── searcher.py
└── notebooks/
    └── demo.ipynb             # Optional demo notebook

**-Installation-**

1️⃣ Clone repo
git clone <your_repo_url>.git
cd semantic-search-engine

2️⃣ Create a virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

**Preparing the Dataset**

Place your .txt files inside:

data/docs/
If you need sample documents, run the helper script:

python scripts/generate_dummy_docs.py


This will generate 100 sample documents automatically.

**Build the Index**

Before running the CLI, set PYTHONPATH:

$env:PYTHONPATH = "src"


Now build the index:

python -m semantic_search.cli index --docs-dir data/docs --index-dir index


This will:

Load all documents

Generate embeddings

Save the index in the index/ folder

**Performing a Search**
Single Query Search
python -m semantic_search.cli search "deep learning in NLP" -k 5


Output Example:

#	Document ID	Score	Snippet
1	doc_100.txt	0.5699	Deep learning is a subset...
2	doc_060.txt	0.5697	Deep learning is a subset...
…	…	…	…
Batch Search

Prepare a file:

data/queries.txt


Then run:

python -m semantic_search.cli search-batch data/queries.txt -k 3


This prints results for each query in a formatted table.

**Performance Benchmarks**

All benchmarks executed on CPU-only, Windows 11, Python 3.11.
Dataset: 100 text documents.

**Indexing Performance**
Metric	Value
Documents indexed	100
Model used	all-MiniLM-L6-v2
Total indexing time	Displayed in CLI during index build

Indexing includes:

Reading documents

Batch embedding (32 docs/batch)

Writing embeddings and metadata

**Search Latency Benchmark**

Benchmark script:

python scripts/benchmark_search.py


Results:

Total queries      : 25
Total time         : 5.9246 seconds
Average per query  : 236.98 ms

Interpretation

~237ms per query is standard for CPU transformers

Speed improves significantly with:

GPU inference

Smaller embedding models

FAISS / HNSW ANN indexing (sub-ms retrieval)

**-Design Considerations-**

This project follows a modular architecture:

1. loader.py

Safely loads .txt files + document IDs.

2. embeddings.py

Handles:

Model loading

Batch text embedding

Normalized vectors

3. indexer.py

Creates:

embeddings.npy

metadata.json

4. searcher.py

Performs:

Query embedding

Cosine similarity

Ranking

Top-k retrieval

5. CLI (cli.py)

Provides:

index

search

search-batch

**Future Improvements**

These enhancements can take this project to the next level:

Add FAISS Indexing (for millions of docs)

Use FAISS or Annoy for sub-millisecond ANN search.

**Add REST API (FastAPI)**

Expose:

/search

/batch-search

**Add Frontend UI**

Use Streamlit or React for:

Search bar

Ranked results

Document preview
**Integrate Into RAG Pipeline**

Use this as the retrieval layer for LLM-based question answering.

**MIT License**
Open-source. Free to use, modify, and distribute.
