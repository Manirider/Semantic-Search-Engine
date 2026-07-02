# Semantic-Search-Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![License](https://img.shields.io/github/license/Manirider/Semantic-Search-Engine?style=flat-square) ![Last Commit](https://img.shields.io/github/last-commit/Manirider/Semantic-Search-Engine?style=flat-square) ![Issues](https://img.shields.io/github/issues/Manirider/Semantic-Search-Engine?style=flat-square)

`portfolio-project`

## Project Overview

An information retrieval engine utilizing Sentence Transformers to map text queries to document embeddings, returning search results ranked by semantic similarity.

## Core Features

- Text embedding generation using the all-MiniLM-L6-v2 transformer model.
- Vector comparison pipeline calculating cosine similarity scores.
- Document index cataloging text assets and embeddings.
- FastAPI interface serving search queries.
- Preprocessing pipelines cleaning and chunking source text files.

## Technical Flow & Execution

Documents are chunked, embedded, and added to the index. When a user submits a query, the system generates a query embedding, calculates similarity scores, and returns the top-ranked matches.

## Getting Started

### Requirements

- Python 3.10 or higher
- Pip package manager

### Environment Configuration

```bash
# Clone this repository
git clone https://github.com/Manirider/Semantic-Search-Engine.git
cd Semantic-Search-Engine

# Create a virtual environment to manage dependencies locally
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required library dependencies
pip install -r requirements.txt
```

### Execution

```bash
python main.py
```

## Directory Layout

```
Semantic-Search-Engine/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
└── (source files)
```

## Contributing to the Project

I welcome issues and pull requests to make this project better. Please see the detailed guidelines in the [Contributing Guide](CONTRIBUTING.md).

## Project License

This repository is distributed under the MIT License. For complete terms, see the [LICENSE](LICENSE) file.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)
