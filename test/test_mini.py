"""
Mini smoke test — verifies all three pipeline generators work correctly
on the pre-built _smoketest ChromaDB collections (no Ollama/RAGAS required
for the retrieval logic check; queries Ollama for generation).

Run: uv run python test_mini.py
"""
import os
import sys
import traceback
import chromadb
from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)
sys.path.append(_REPO_ROOT)

SMOKETEST_ROOT = os.path.join(_REPO_ROOT, "data", "_smoketest")

TEST_QUESTION = "What method or model is proposed and what dataset is used for evaluation?"

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check_collection(db_dir: str, col_name: str) -> tuple[bool, int]:
    try:
        c = chromadb.PersistentClient(path=db_dir)
        col = c.get_collection(col_name)
        return True, col.count()
    except Exception:
        return False, 0


def test_retrieval_only():
    """Test two-channel retrieval logic directly without Ollama (fast)."""
    print("\n--- [1] Retrieval Logic (no Ollama) ---")
    db_dir = os.path.join(SMOKETEST_ROOT, "embeddings")
    ok, count = check_collection(db_dir, "qasper_graph_rag")
    if not ok or count == 0:
        print(f"  {SKIP} Graph with markdown smoketest DB not found at {db_dir}")
        return False

    import chromadb
    client = chromadb.PersistentClient(path=db_dir)
    col = client.get_collection("qasper_graph_rag")

    # Use a fake 1024-dim zero embedding to test filter logic without bge-m3
    fake_emb = [0.0] * 1024  # bge-m3 is 1024-dim

    try:
        # Channel 1: semantic sections
        sem_r = col.query(
            query_embeddings=[fake_emb],
            n_results=5,
            where={"type": {"$eq": "semantic_section"}},
            include=["documents", "metadatas"]
        )
        sem_docs = sem_r["documents"][0]
        sem_metas = sem_r["metadatas"][0]
        paper_ids = list({m["paper_id"] for m in sem_metas if "paper_id" in m})
        print(f"  Channel 1: {len(sem_docs)} semantic chunks from papers {paper_ids}")

        # Channel 2: graph edges scoped to retrieved papers
        graph_r = col.query(
            query_embeddings=[fake_emb],
            n_results=5,
            where={"$and": [{"type": {"$eq": "graph_edge"}}, {"paper_id": {"$in": paper_ids}}]},
            include=["documents"]
        )
        graph_docs = graph_r["documents"][0]
        print(f"  Channel 2: {len(graph_docs)} graph edges (scoped to same papers)")
        print(f"  {PASS} Two-channel retrieval filters work correctly")
        return True
    except Exception as e:
        print(f"  {FAIL} Retrieval filter error: {e}")
        traceback.print_exc()
        return False


def test_baseline_generator():
    """Test Baseline Generator end-to-end."""
    print("\n--- [2] Baseline Generator ---")
    db_dir = os.path.join(SMOKETEST_ROOT, "base_embeddings")
    ok, count = check_collection(db_dir, "baseline_rag")
    if not ok or count == 0:
        print(f"  {SKIP} Baseline smoketest DB not found")
        return False

    try:
        from src.core.ollama_manager import OllamaManager
        from src.baseline.generator import BaselineGenerator
        ollama = OllamaManager()
        gen = BaselineGenerator(ollama, db_dir, embed_model="bge-m3", llm_model="llama3.1:8b")
        result = gen.query(TEST_QUESTION, top_k=5)
        ans = result["answer"]
        ctx = result["context"]
        print(f"  Contexts retrieved: {len(ctx)}")
        print(f"  Answer preview: {ans[:200]}")
        assert len(ans) > 0, "Empty answer"
        print(f"  {PASS} Baseline generator OK")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        traceback.print_exc()
        return False


def test_graphnomd_generator():
    """Test Graph no markdown Generator (fixed chunks + graph, reuses Generator class)."""
    print("\n--- [3] Graph no markdown Generator ---")
    db_dir = os.path.join(SMOKETEST_ROOT, "p3_embeddings")
    ok, count = check_collection(db_dir, "qasper_graph_rag")
    if not ok or count == 0:
        print(f"  {SKIP} Graph no markdown smoketest DB not found")
        return False

    try:
        from src.core.ollama_manager import OllamaManager
        from src.components.generator import Generator
        ollama = OllamaManager()
        gen = Generator(ollama, db_dir, embed_model="bge-m3", llm_model="llama3.1:8b")
        result = gen.query(TEST_QUESTION, top_k=10)
        ans = result["answer"]
        ctx = result["context"]
        print(f"  Contexts retrieved: {len(ctx)}")
        print(f"  Answer preview: {ans[:200]}")
        # Graph no markdown has baseline_chunk (not semantic_section) + graph_edge
        # Channel 1 filter for semantic_section returns 0 → fallback activates
        assert len(ans) > 0, "Empty answer"
        print(f"  {PASS} Graph no markdown generator OK (fallback path for non-semantic chunks)")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        traceback.print_exc()
        return False


def test_graphmd_generator():
    """Test Graph with markdown Generator end-to-end."""
    print("\n--- [4] Graph with markdown Generator ---")
    db_dir = os.path.join(SMOKETEST_ROOT, "embeddings")
    ok, count = check_collection(db_dir, "qasper_graph_rag")
    if not ok or count == 0:
        print(f"  {SKIP} Graph with markdown smoketest DB not found")
        return False

    try:
        from src.core.ollama_manager import OllamaManager
        from src.components.generator import Generator
        ollama = OllamaManager()
        gen = Generator(ollama, db_dir, embed_model="bge-m3", llm_model="llama3.1:8b")
        result = gen.query(TEST_QUESTION, top_k=10)
        ans = result["answer"]
        ctx = result["context"]
        print(f"  Contexts retrieved: {len(ctx)}")
        print(f"  Answer preview: {ans[:200]}")
        assert len(ans) > 0, "Empty answer"
        assert len(ctx) > 0, "No context retrieved"
        print(f"  {PASS} Graph with markdown generator OK")
        return True
    except Exception as e:
        print(f"  {FAIL} {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 55)
    print("  PIPELINE SMOKE TEST (Mini Data)")
    print("=" * 55)
    print(f"  Test question: {TEST_QUESTION}")

    results = {
        "Retrieval logic":        test_retrieval_only(),
        "Baseline":               test_baseline_generator(),
        "Graph no markdown":      test_graphnomd_generator(),
        "Graph with markdown":    test_graphmd_generator(),
    }

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    all_ok = True
    for name, ok in results.items():
        status = PASS if ok else (SKIP if ok is None else FAIL)
        print(f"  {status}  {name}")
        if ok is False:
            all_ok = False

    if all_ok:
        print("\n  All checks passed. Safe to run full benchmarks.")
    else:
        print("\n  Fix failures above before running full benchmarks.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
