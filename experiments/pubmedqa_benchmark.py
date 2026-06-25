"""
PubMedQA Benchmark — Baseline vs Graph no markdown vs Graph with markdown.
Dataset: qiaojin/PubMedQA pqa_labeled — 1000 biomedical QA samples.
Ground truth: long_answer (free-form, suitable for all RAGAS metrics).

Modes:
  Full (default): 500 samples, seed=42, full ingestion + generation + RAGAS.
  Mini:           10 samples,  seed=99, skips ingestion (requires existing embeddings).
                  Set MINI = True below, or run: uv run python pubmedqa_benchmark.py mini

Run: uv run python pubmedqa_benchmark.py
     uv run python pubmedqa_benchmark.py mini
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
from src.ablation.p3_embedder import P3Embedder
from src.components.evaluator import Evaluator

# --- CONFIG ---
MINI = "mini" in sys.argv  # True = 10-sample quick validation mode

DATA_ROOT           = os.path.join(_REPO_ROOT, "data", "pubmedqa")
PARSED_DIR          = os.path.join(DATA_ROOT, "parsed")
GRAPH_DIR           = os.path.join(DATA_ROOT, "graph")
EMBED_DIR           = os.path.join(DATA_ROOT, "embeddings", "graphmd")
BASE_PARSED_DIR     = os.path.join(DATA_ROOT, "parsed_txt")
BASE_EMBED_DIR      = os.path.join(DATA_ROOT, "embeddings", "baseline")
GRAPHNOMD_EMBED_DIR = os.path.join(DATA_ROOT, "embeddings", "graphnomd")
RESULTS_DIR         = os.path.join(DATA_ROOT, "results")
MINI_RESULTS_DIR    = os.path.join(DATA_ROOT, "results", "mini")

SAMPLE_SIZE = 10  if MINI else None   # None means all 1000 samples
SEED        = 99  if MINI else 42

for d in [PARSED_DIR, GRAPH_DIR, EMBED_DIR, BASE_PARSED_DIR, BASE_EMBED_DIR, GRAPHNOMD_EMBED_DIR, RESULTS_DIR, MINI_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ #
#  DATA                                                                #
# ------------------------------------------------------------------ #

def load_pubmedqa() -> list:
    from datasets import load_dataset
    print(f"[Data] Loading PubMedQA (pqa_labeled) — {'MINI mode' if MINI else 'full mode'}...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    data = list(ds)
    if SAMPLE_SIZE and len(data) > SAMPLE_SIZE:
        random.seed(SEED)
        data = random.sample(data, SAMPLE_SIZE)
    print(f"[Data] Using {len(data)} samples (seed={SEED}).")
    return data


def prepare_files(samples: list) -> list:
    """Write .md (Graph with markdown) and .txt (Baseline/Graph no markdown). Return QA list."""
    print(f"\n[Prep] Writing context files...")
    qa_list = []
    for item in tqdm(samples, desc="Writing files"):
        pubid    = str(item["pubid"])
        contexts = item["context"]["contexts"]
        labels   = item["context"]["labels"]

        qa_list.append({
            "id": pubid,
            "question": item["question"],
            "ground_truth": item["long_answer"],
        })

        md_path = os.path.join(PARSED_DIR, f"{pubid}.md")
        if not os.path.exists(md_path):
            md = f"# PubMed {pubid}\n\n"
            for label, ctx in zip(labels, contexts):
                md += f"## {label}\n{ctx}\n\n"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)

        txt_path = os.path.join(BASE_PARSED_DIR, f"{pubid}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(" ".join(contexts))

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

    print("[1] BaselineEmbedder (BGE-M3) — fixed-size chunks...")
    BaselineEmbedder(ollama, BASE_PARSED_DIR, BASE_EMBED_DIR, model_name="bge-m3").process_all()


def ingest_graphnomd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph no markdown <<<")
    print("="*50)

    print("[1] P3Embedder (BGE-M3) — fixed-size text + graph edges...")
    P3Embedder(ollama, txt_dir=BASE_PARSED_DIR, graph_dir=GRAPH_DIR,
               db_dir=GRAPHNOMD_EMBED_DIR, model_name="bge-m3").process_all()


def ingest_graphmd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph with markdown <<<")
    print("="*50)
    if _chroma_count(EMBED_DIR, "qasper_graph_rag") > 0:
        print(f"Graph with markdown DB exists ({_chroma_count(EMBED_DIR, 'qasper_graph_rag')} chunks). Skipping.")
        return

    print("[1] GraphBuilder (Qwen 14B)...")
    GraphBuilder(ollama, PARSED_DIR, GRAPH_DIR, model_name="qwen2.5:7b").process_all()

    print("\n[2] Embedder (BGE-M3) — semantic sections + graph edges...")
    Embedder(ollama, PARSED_DIR, GRAPH_DIR, EMBED_DIR, model_name="bge-m3").process_all()


# ------------------------------------------------------------------ #
#  GENERATION                                                          #
# ------------------------------------------------------------------ #

def run_generation(ollama: OllamaManager, qa_list: list):
    print("\n" + "="*50)
    print(">>> GENERATION — All 3 Pipelines (Llama 3.1:8B) <<<")
    print("="*50)

    baseline_gen  = BaselineGenerator(ollama, BASE_EMBED_DIR,     embed_model="bge-m3", llm_model="llama3.1:8b")
    mdonly_gen    = Generator(ollama, EMBED_DIR,                   embed_model="bge-m3", llm_model="llama3.1:8b", use_graph=False)
    graphnomd_gen = Generator(ollama, GRAPHNOMD_EMBED_DIR,         embed_model="bge-m3", llm_model="llama3.1:8b", graph_dir=GRAPH_DIR)
    graphmd_gen   = Generator(ollama, EMBED_DIR,                   embed_model="bge-m3", llm_model="llama3.1:8b", graph_dir=GRAPH_DIR)

    baseline_results, mdonly_results, graphnomd_results, graphmd_results = [], [], [], []

    for i, qa in enumerate(qa_list):
        print(f"\n[{i+1}/{len(qa_list)}] {qa['question'][:80]}")

        for gen, store, top_k, label in [
            (baseline_gen,  baseline_results,  5,  "Baseline"),
            (mdonly_gen,    mdonly_results,    10,  "Markdown only"),
            (graphnomd_gen, graphnomd_results, 10,  "Graph no markdown"),
            (graphmd_gen,   graphmd_results,   10,  "Graph with markdown"),
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

    out_dir = MINI_RESULTS_DIR if MINI else RESULTS_DIR
    pd.DataFrame(baseline_results).to_json( os.path.join(out_dir, "baseline_raw.jsonl"),  orient="records", lines=True, force_ascii=False)
    pd.DataFrame(mdonly_results).to_json(   os.path.join(out_dir, "mdonly_raw.jsonl"),     orient="records", lines=True, force_ascii=False)
    pd.DataFrame(graphnomd_results).to_json(os.path.join(out_dir, "graphnomd_raw.jsonl"),  orient="records", lines=True, force_ascii=False)
    pd.DataFrame(graphmd_results).to_json(  os.path.join(out_dir, "graphmd_raw.jsonl"),    orient="records", lines=True, force_ascii=False)
    print(f"\nRaw saved — Baseline:{len(baseline_results)}, MDOnly:{len(mdonly_results)}, GraphNoMD:{len(graphnomd_results)}, GraphMD:{len(graphmd_results)}")

    return baseline_results, mdonly_results, graphnomd_results, graphmd_results


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def print_comparison(scores: dict):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 72)
    print(f"  PUBMEDQA RESULTS ({'MINI — ' + str(SAMPLE_SIZE) + ' samples' if MINI else str(SAMPLE_SIZE) + ' samples'})")
    print("=" * 72)
    print(f"{'Pipeline':<30}" + "".join(f"{m[:10]:>12}" for m in metrics))
    print("-" * 72)
    for label, means in scores.items():
        print(f"{label:<30}" + "".join(f"{means.get(m, float('nan')):>12.4f}" for m in metrics))
    print("=" * 72)
    if "Graph with markdown" in scores and "Graph no markdown" in scores:
        gmd, gnomd = scores["Graph with markdown"], scores["Graph no markdown"]
        print("\n  Markdown effect — GraphMD vs GraphNoMD delta (positive = GraphMD better):")
        for m in metrics:
            d = gmd.get(m, 0) - gnomd.get(m, 0)
            print(f"    {m:<22} {'+' if d >= 0 else ''}{d:.4f}")
    if "Graph with markdown" in scores and "Markdown only" in scores:
        gmd, mdo = scores["Graph with markdown"], scores["Markdown only"]
        print("\n  Graph effect — GraphMD vs MarkdownOnly delta (positive = graph helps):")
        for m in metrics:
            d = gmd.get(m, 0) - mdo.get(m, 0)
            print(f"    {m:<22} {'+' if d >= 0 else ''}{d:.4f}")


def main():
    if MINI:
        print(f"\n[MINI MODE] Checking embeddings exist (seed={SEED}, n={SAMPLE_SIZE})...")
        for name, path, col in [
            ("Baseline",            BASE_EMBED_DIR,      "baseline_rag"),
            ("Graph no markdown",   GRAPHNOMD_EMBED_DIR, "qasper_graph_rag"),
            ("Graph with markdown", EMBED_DIR,            "qasper_graph_rag"),
        ]:
            cnt = _chroma_count(path, col)
            if cnt == 0:
                print(f"[ERROR] {name} DB missing at {path}. Run full benchmark first.")
                sys.exit(1)
            print(f"  {name}: {cnt} chunks OK")

    samples = load_pubmedqa()
    qa_list = prepare_files(samples)

    ollama = OllamaManager()

    if not MINI:
        ingest_baseline(ollama)
        ingest_graphnomd(ollama)
        ingest_graphmd(ollama)

    baseline_results, mdonly_results, graphnomd_results, graphmd_results = run_generation(ollama, qa_list)

    scores = {}
    evaluator = Evaluator(use_local_model=False)
    print("\n" + "="*50)
    print(">>> RAGAS EVALUATION (GPT-4o-mini) <<<")
    print("="*50)
    out_dir = MINI_RESULTS_DIR if MINI else RESULTS_DIR
    for label, results, out_csv in [
        ("Baseline",            baseline_results,  "baseline_metrics.csv"),
        ("Markdown only",       mdonly_results,    "mdonly_metrics.csv"),
        ("Graph no markdown",   graphnomd_results, "graphnomd_metrics.csv"),
        ("Graph with markdown", graphmd_results,   "graphmd_metrics.csv"),
    ]:
        print(f"\n--- {label} ---")
        try:
            df = evaluator.evaluate_dataframe(results)
            df.to_csv(os.path.join(out_dir, out_csv), index=False)
            print(f"=> Saved: pubmedqa/results/{out_csv}")
            scores[label] = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().to_dict()
        except Exception as e:
            print(f"RAGAS error ({label}): {e}")

    if scores:
        print_comparison(scores)

    print("\n" + "="*50)
    print("  PUBMEDQA BENCHMARK COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
