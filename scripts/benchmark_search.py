from time import perf_counter
from semantic_search.searcher import SemanticSearchEngine

def main():
    print("[INFO] Initializing search engine...")
    engine = SemanticSearchEngine()

    # define queries
    queries = [
        "what is deep learning",
        "applications of reinforcement learning",
        "support vector machines for classification",
        "natural language processing tasks",
        "how does transfer learning work",
    ] * 5  # 25 queries

    print(f"[INFO] Running benchmark on {len(queries)} queries...")
    start = perf_counter()
    for q in queries:
        engine.search(q, top_k=5)
    elapsed = perf_counter() - start
    avg = elapsed / len(queries)

    print("\n=== Benchmark Results ===")
    print(f"Total queries      : {len(queries)}")
    print(f"Total time         : {elapsed:.4f} seconds")
    print(f"Average per query  : {avg * 1000:.2f} ms")

if __name__ == "__main__":
    main()
