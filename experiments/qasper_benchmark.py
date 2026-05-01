"""
Qasper Benchmark — Baseline vs Graph no markdown vs Graph with markdown.
Dataset: qasper-dev-v0.3.json — 355 QA pairs with free-form ground truth.
Run: uv run python qasper_benchmark.py
"""
import os
import sys
import json
import pandas as pd
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_REPO_ROOT, ".env"), override=True)
sys.path.append(_REPO_ROOT)

from src.core.ollama_manager import OllamaManager
from src.components.loader import QasperLoader
from src.components.graph_builder import GraphBuilder
from src.components.embedder import Embedder
from src.components.generator import Generator
from src.baseline.loader import BaselineLoader
from src.baseline.embedder import BaselineEmbedder
from src.baseline.generator import BaselineGenerator
from src.ablation.p3_embedder import P3Embedder
from src.components.evaluator import Evaluator

# --- CONFIG ---
DATA_ROOT      = os.path.join(_REPO_ROOT, "data", "qasper")
RAW_DIR        = os.path.join(_REPO_ROOT, "data", "raw")
PARSED_DIR     = os.path.join(DATA_ROOT, "parsed")
BASE_PARSED    = os.path.join(DATA_ROOT, "parsed_txt")
GRAPH_DIR      = os.path.join(DATA_ROOT, "graph")
EMBED_DIR      = os.path.join(DATA_ROOT, "embeddings", "graphmd")
BASE_EMBED_DIR = os.path.join(DATA_ROOT, "embeddings", "baseline")
GRAPHNOMD_DIR  = os.path.join(DATA_ROOT, "embeddings", "graphnomd")
RESULTS_DIR    = os.path.join(DATA_ROOT, "results")

SOURCE_JSON = "qasper-dev-v0.3.json"

for d in [PARSED_DIR, BASE_PARSED, GRAPH_DIR, EMBED_DIR, BASE_EMBED_DIR, GRAPHNOMD_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ #
#  DATA                                                                #
# ------------------------------------------------------------------ #

def extract_qa(json_path: str) -> list:
    data = json.load(open(json_path, "r", encoding="utf-8"))
    qa_list = []
    for paper_id, paper in data.items():
        for qa in paper.get("qas", []):
            question = qa.get("question", "")
            for ans in qa.get("answers", []):
                gt = ans.get("answer", {}).get("free_form_answer", "")
                if gt.strip():
                    qa_list.append({"question": question, "ground_truth": gt})
                    break
    return qa_list


def prepare_files():
    """Parse QASPER JSON → .md and .txt files if not already done."""
    json_path = os.path.join(RAW_DIR, SOURCE_JSON)
    md_count  = len([f for f in os.listdir(PARSED_DIR)  if f.endswith(".md")])
    txt_count = len([f for f in os.listdir(BASE_PARSED) if f.endswith(".txt")])

    if md_count == 0:
        print("[Prep] Writing .md files (Graph with markdown)...")
        QasperLoader(RAW_DIR, PARSED_DIR).process_file(SOURCE_JSON)
    else:
        print(f"[Prep] {md_count} .md files found. Skipping.")

    if txt_count == 0:
        print("[Prep] Writing .txt files (Baseline / Graph no markdown)...")
        BaselineLoader(RAW_DIR, BASE_PARSED).process_file(SOURCE_JSON)
    else:
        print(f"[Prep] {txt_count} .txt files found. Skipping.")


# ------------------------------------------------------------------ #
#  INGESTION                                                           #
# ------------------------------------------------------------------ #

def _count(db_dir: str, col: str) -> int:
    try:
        return chromadb.PersistentClient(path=db_dir).get_collection(col).count()
    except Exception:
        return 0


def ingest_baseline(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Baseline <<<")
    print("="*50)
    if _count(BASE_EMBED_DIR, "baseline_rag") > 0:
        print(f"Baseline DB exists ({_count(BASE_EMBED_DIR, 'baseline_rag')} chunks). Skipping.")
        return
    BaselineEmbedder(ollama, BASE_PARSED, BASE_EMBED_DIR, model_name="bge-m3").process_all()


def ingest_graphnomd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph no markdown <<<")
    print("="*50)
    P3Embedder(ollama, txt_dir=BASE_PARSED, graph_dir=GRAPH_DIR,
               db_dir=GRAPHNOMD_DIR, model_name="bge-m3").process_all()


def ingest_graphmd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph with markdown <<<")
    print("="*50)
    if _count(EMBED_DIR, "qasper_graph_rag") > 0:
        print(f"Graph with markdown DB exists ({_count(EMBED_DIR, 'qasper_graph_rag')} chunks). Skipping.")
        return

    print("[1] GraphBuilder (Qwen 7B)...")
    GraphBuilder(ollama, PARSED_DIR, GRAPH_DIR, model_name="qwen2.5:7b").process_all()

    print("\n[2] Embedder (BGE-M3) — semantic sections + graph edges...")
    Embedder(ollama, PARSED_DIR, GRAPH_DIR, EMBED_DIR, model_name="bge-m3").process_all()


# ------------------------------------------------------------------ #
#  GENERATION                                                          #
# ------------------------------------------------------------------ #

def run_generation(ollama: OllamaManager, qa_list: list):
    print("\n" + "="*50)
    print(f">>> GENERATION — All 3 Pipelines ({len(qa_list)} questions) <<<")
    print("="*50)

    baseline_gen  = BaselineGenerator(ollama, BASE_EMBED_DIR,  embed_model="bge-m3", llm_model="llama3.1:8b")
    graphnomd_gen = Generator(ollama, GRAPHNOMD_DIR,            embed_model="bge-m3", llm_model="llama3.1:8b")
    graphmd_gen   = Generator(ollama, EMBED_DIR,                embed_model="bge-m3", llm_model="llama3.1:8b")

    baseline_results, graphnomd_results, graphmd_results = [], [], []

    for i, qa in enumerate(qa_list):
        print(f"\n[{i+1}/{len(qa_list)}] {qa['question'][:80]}")
        for gen, store, top_k, label in [
            (baseline_gen,  baseline_results,  5,  "Baseline"),
            (graphnomd_gen, graphnomd_results, 10, "Graph no markdown"),
            (graphmd_gen,   graphmd_results,   10, "Graph with markdown"),
        ]:
            try:
                res = gen.query(qa["question"], top_k=top_k)
                store.append({
                    "question":     qa["question"],
                    "answer":       res["answer"],
                    "contexts":     res["context"],
                    "ground_truth": qa["ground_truth"],
                })
            except Exception as e:
                print(f"  {label} error: {e}")

    pd.DataFrame(baseline_results).to_json( os.path.join(RESULTS_DIR, "baseline_raw.jsonl"),  orient="records", lines=True, force_ascii=False)
    pd.DataFrame(graphnomd_results).to_json(os.path.join(RESULTS_DIR, "graphnomd_raw.jsonl"),  orient="records", lines=True, force_ascii=False)
    pd.DataFrame(graphmd_results).to_json(  os.path.join(RESULTS_DIR, "graphmd_raw.jsonl"),    orient="records", lines=True, force_ascii=False)
    print(f"\nRaw saved — Baseline:{len(baseline_results)}, GraphNoMD:{len(graphnomd_results)}, GraphMD:{len(graphmd_results)}")

    return baseline_results, graphnomd_results, graphmd_results


# ------------------------------------------------------------------ #
#  EVALUATION                                                          #
# ------------------------------------------------------------------ #

def run_ragas(baseline_results, graphnomd_results, graphmd_results):
    print("\n" + "="*50)
    print(">>> RAGAS EVALUATION (GPT-4o-mini) <<<")
    print("="*50)

    evaluator = Evaluator(use_local_model=False)

    for label, results, out_csv in [
        ("Baseline",            baseline_results,  "baseline_metrics.csv"),
        ("Graph no markdown",   graphnomd_results, "graphnomd_metrics.csv"),
        ("Graph with markdown", graphmd_results,   "graphmd_metrics.csv"),
    ]:
        print(f"\n--- {label} ---")
        try:
            df = evaluator.evaluate_dataframe(results)
            df.to_csv(os.path.join(RESULTS_DIR, out_csv), index=False)
            print(f"=> Saved: qasper/results/{out_csv}")
        except Exception as e:
            print(f"RAGAS error ({label}): {e}")


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main():
    json_path = os.path.join(RAW_DIR, SOURCE_JSON)
    if not os.path.exists(json_path):
        print(f"[ERROR] QASPER JSON not found: {json_path}")
        sys.exit(1)

    prepare_files()

    qa_list = extract_qa(json_path)
    print(f"\n[Data] {len(qa_list)} QA pairs extracted.")

    ollama = OllamaManager()

    ingest_baseline(ollama)
    ingest_graphnomd(ollama)
    ingest_graphmd(ollama)

    baseline_results, graphnomd_results, graphmd_results = run_generation(ollama, qa_list)

    run_ragas(baseline_results, graphnomd_results, graphmd_results)

    print("\n" + "="*50)
    print("  QASPER BENCHMARK COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
