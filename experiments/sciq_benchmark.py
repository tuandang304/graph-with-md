"""
SciQ Benchmark — Baseline vs Graph no markdown vs Graph with markdown.
Dataset: allenai/sciq — Science textbook QA (train + val + test).
Ground truth: correct_answer + support paragraph.

Run: uv run python experiments/sciq_benchmark.py
"""
import os
import sys
import random
import pandas as pd
import chromadb
from tqdm import tqdm
from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_REPO_ROOT, ".env"), override=True)
sys.path.append(_REPO_ROOT)

from src.core.ollama_manager import OllamaManager
from src.components.graph_builder import GraphBuilder
from src.components.embedder import Embedder
from src.components.generator import Generator
from src.baseline.embedder import BaselineEmbedder
from src.baseline.generator import BaselineGenerator
from src.ablation.graph_no_markdown_embedder import GraphNoMarkdownEmbedder
from src.components.evaluator import Evaluator

# --- CONFIG ---
DATA_ROOT           = os.path.join(_REPO_ROOT, "data", "sciq")
PARSED_DIR          = os.path.join(DATA_ROOT, "parsed")
GRAPH_DIR           = os.path.join(DATA_ROOT, "graph")
EMBED_DIR           = os.path.join(DATA_ROOT, "embeddings", "graphmd")
BASE_PARSED_DIR     = os.path.join(DATA_ROOT, "parsed_txt")
BASE_EMBED_DIR      = os.path.join(DATA_ROOT, "embeddings", "baseline")
GRAPHNOMD_EMBED_DIR = os.path.join(DATA_ROOT, "embeddings", "graphnomd")
RESULTS_DIR         = os.path.join(DATA_ROOT, "results")

SAMPLE_SIZE = 1000   # None for all ~12k samples with support
SEED        = 42

for d in [PARSED_DIR, GRAPH_DIR, EMBED_DIR, BASE_PARSED_DIR, BASE_EMBED_DIR, GRAPHNOMD_EMBED_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ #
#  DATA                                                                #
# ------------------------------------------------------------------ #

def load_sciq() -> list:
    from datasets import load_dataset
    print("[Data] Loading SciQ (allenai/sciq) from HuggingFace...")
    ds = load_dataset("allenai/sciq")

    # Combine all splits and filter out samples with empty support
    all_data = []
    for split in ["train", "validation", "test"]:
        for item in ds[split]:
            support = item["support"].strip()
            if support:  # Only keep samples with non-empty support
                all_data.append(item)

    print(f"[Data] {len(all_data)} samples with non-empty support (filtered from {sum(len(ds[s]) for s in ds)}).")

    if SAMPLE_SIZE and len(all_data) > SAMPLE_SIZE:
        random.seed(SEED)
        all_data = random.sample(all_data, SAMPLE_SIZE)

    print(f"[Data] Using {len(all_data)} samples (seed={SEED}).")
    return all_data


def prepare_files(samples: list) -> list:
    """Write .md (Graph with markdown) and .txt (Baseline/Graph no markdown). Return QA list."""
    print(f"\n[Prep] Writing context files...")
    qa_list = []

    for i, item in enumerate(tqdm(samples, desc="Writing files")):
        doc_id = f"sciq_{i:05d}"
        support = item["support"].strip()
        question = item["question"]
        correct = item["correct_answer"]

        # Ground truth: combine correct answer with the support paragraph
        # This gives RAGAS enough text to evaluate context_recall properly
        ground_truth = f"{correct}. {support}"

        qa_list.append({
            "id": doc_id,
            "question": question,
            "ground_truth": ground_truth,
        })

        # --- .md file (Graph with markdown) ---
        # Structure the support text with markdown headings for semantic chunking
        md_path = os.path.join(PARSED_DIR, f"{doc_id}.md")
        if not os.path.exists(md_path):
            md = f"# Science Textbook — {doc_id}\n\n"
            md += f"## Content\n{support}\n\n"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)

        # --- .txt file (Baseline / Graph no markdown) ---
        txt_path = os.path.join(BASE_PARSED_DIR, f"{doc_id}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(support)

    return qa_list


# ------------------------------------------------------------------ #
#  INGESTION                                                           #
# ------------------------------------------------------------------ #

def _chroma_count(db_dir: str, collection: str) -> int:
    try:
        c = chromadb.PersistentClient(path=db_dir)
        return c.get_collection(collection).count()
    except Exception:
        return 0


def ingest_baseline(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Baseline <<<")
    print("="*50)
    if _chroma_count(BASE_EMBED_DIR, "baseline_rag") > 0:
        print(f"Baseline DB exists ({_chroma_count(BASE_EMBED_DIR, 'baseline_rag')} chunks). Skipping.")
        return
    BaselineEmbedder(ollama, BASE_PARSED_DIR, BASE_EMBED_DIR, model_name="bge-m3").process_all()


def ingest_graphnomd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph no markdown <<<")
    print("="*50)
    GraphNoMarkdownEmbedder(ollama, txt_dir=BASE_PARSED_DIR, graph_dir=GRAPH_DIR,
                            db_dir=GRAPHNOMD_EMBED_DIR, model_name="bge-m3").process_all()


def ingest_graphmd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph with markdown <<<")
    print("="*50)
    if _chroma_count(EMBED_DIR, "qasper_graph_rag") > 0:
        print(f"Graph with markdown DB exists ({_chroma_count(EMBED_DIR, 'qasper_graph_rag')} chunks). Skipping.")
        return

    print("[1] GraphBuilder (Qwen 7B)...")
    GraphBuilder(ollama, PARSED_DIR, GRAPH_DIR, model_name="qwen2.5:7b-instruct-q4_K_M").process_all()

    print("\n[2] Embedder (BGE-M3) — semantic sections + graph edges...")
    Embedder(ollama, PARSED_DIR, GRAPH_DIR, EMBED_DIR, model_name="bge-m3").process_all()


# ------------------------------------------------------------------ #
#  GENERATION                                                          #
# ------------------------------------------------------------------ #

def run_generation(ollama: OllamaManager, qa_list: list):
    print("\n" + "="*50)
    print(f">>> GENERATION — All 3 Pipelines ({len(qa_list)} questions) <<<")
    print("="*50)

    baseline_gen  = BaselineGenerator(ollama, BASE_EMBED_DIR,     embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M")
    mdonly_gen    = Generator(ollama, EMBED_DIR,                   embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M", use_graph=False)
    graphnomd_gen = Generator(ollama, GRAPHNOMD_EMBED_DIR,         embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M", graph_dir=GRAPH_DIR)
    graphmd_gen   = Generator(ollama, EMBED_DIR,                   embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M", graph_dir=GRAPH_DIR)

    baseline_results, mdonly_results, graphnomd_results, graphmd_results = [], [], [], []

    for i, qa in enumerate(qa_list):
        print(f"\n[{i+1}/{len(qa_list)}] {qa['question'][:80]}")

        for gen, store, top_k, label in [
            (baseline_gen,  baseline_results,  5,  "Baseline"),
            (mdonly_gen,    mdonly_results,    10, "Markdown only"),
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
    pd.DataFrame(mdonly_results).to_json(   os.path.join(RESULTS_DIR, "mdonly_raw.jsonl"),     orient="records", lines=True, force_ascii=False)
    pd.DataFrame(graphnomd_results).to_json(os.path.join(RESULTS_DIR, "graphnomd_raw.jsonl"),  orient="records", lines=True, force_ascii=False)
    pd.DataFrame(graphmd_results).to_json(  os.path.join(RESULTS_DIR, "graphmd_raw.jsonl"),    orient="records", lines=True, force_ascii=False)
    print(f"\nRaw saved — Baseline:{len(baseline_results)}, MDOnly:{len(mdonly_results)}, GraphNoMD:{len(graphnomd_results)}, GraphMD:{len(graphmd_results)}")

    return baseline_results, mdonly_results, graphnomd_results, graphmd_results


# ------------------------------------------------------------------ #
#  EVALUATION                                                          #
# ------------------------------------------------------------------ #

def run_ragas(baseline_results, mdonly_results, graphnomd_results, graphmd_results):
    print("\n" + "="*50)
    print(">>> RAGAS EVALUATION (Local Qwen) <<<")
    print("="*50)

    evaluator = Evaluator(use_local_model=True)

    scores = {}
    for label, results, out_csv in [
        ("Baseline",            baseline_results,  "baseline_metrics.csv"),
        ("Markdown only",       mdonly_results,    "mdonly_metrics.csv"),
        ("Graph no markdown",   graphnomd_results, "graphnomd_metrics.csv"),
        ("Graph with markdown", graphmd_results,   "graphmd_metrics.csv"),
    ]:
        print(f"\n--- {label} ---")
        try:
            df = evaluator.evaluate_dataframe(results)
            df.to_csv(os.path.join(RESULTS_DIR, out_csv), index=False)
            print(f"=> Saved: sciq/results/{out_csv}")
            scores[label] = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().to_dict()
        except Exception as e:
            print(f"RAGAS error ({label}): {e}")

    if scores:
        print_comparison(scores)


def print_comparison(scores: dict):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 72)
    print(f"  SCIQ RESULTS ({SAMPLE_SIZE or 'all'} samples)")
    print("=" * 72)
    print(f"{'Pipeline':<30}" + "".join(f"{m[:10]:>12}" for m in metrics))
    print("-" * 72)
    for label, means in scores.items():
        print(f"{label:<30}" + "".join(f"{means.get(m, float('nan')):>12.4f}" for m in metrics))
    print("=" * 72)


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main():
    samples = load_sciq()
    qa_list = prepare_files(samples)

    ollama = OllamaManager()

    ingest_baseline(ollama)
    ingest_graphnomd(ollama)
    ingest_graphmd(ollama)

    baseline_results, mdonly_results, graphnomd_results, graphmd_results = run_generation(ollama, qa_list)

    run_ragas(baseline_results, mdonly_results, graphnomd_results, graphmd_results)

    print("\n" + "="*50)
    print("  SCIQ BENCHMARK COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
