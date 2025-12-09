
from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table

from .config import DATA_DIR, INDEX_DIR, DEFAULT_TOP_K
from .indexer import build_index
from .searcher import SemanticSearchEngine

app = typer.Typer(help="Semantic Search CLI")
console = Console()

@app.command()
def index(
    docs_dir: Path = typer.Option(
        DATA_DIR, "--docs-dir", "-d", help="Directory containing .txt documents"
    ),
    index_dir: Path = typer.Option(
        INDEX_DIR, "--index-dir", "-i", help="Directory to store index files"
    ),
):
    """Build the semantic search index."""
    console.print(f"[bold green]Building index[/bold green]")
    build_index(docs_dir=docs_dir, index_dir=index_dir)

@app.command()
def search(
    query: str = typer.Argument(..., help="Query string"),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", "-k", help="Number of results"),
):
    """Run a single semantic search query."""
    engine = SemanticSearchEngine()

    results = engine.search(query, top_k=top_k)
    _print_results(query, results)

@app.command("search-batch")
def search_batch(
    queries_file: Path = typer.Argument(
        ..., help="Path to text file containing one query per line"
    ),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", "-k", help="Number of results"),
):
    """Run semantic search for multiple queries from a file."""
    if not queries_file.exists():
        raise typer.BadParameter(f"File not found: {queries_file}")

    queries: List[str] = [
        line.strip() for line in queries_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    engine = SemanticSearchEngine()
    for q in queries:
        results = engine.search(q, top_k=top_k)
        _print_results(q, results)
        console.print()  # blank line between queries

def _print_results(query: str, results):
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Document ID")
    table.add_column("Score", justify="right")
    table.add_column("Snippet")

    for i, r in enumerate(results, start=1):
        table.add_row(str(i), r.doc_id, f"{r.score:.4f}", r.snippet)

    console.print(table)

def main():
    app()

if __name__ == "__main__":
    main()
