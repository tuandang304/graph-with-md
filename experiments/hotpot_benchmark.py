"""
HotpotQA Benchmark — Baseline vs Graph no markdown vs Graph with markdown.
Dataset: HotpotQA Dev Distractor — 500 samples auto-downloaded.
Run: uv run python hotpot_benchmark.py
"""
import os
import sys
import json
import random
import urllib.request
import pandas as pd
from tqdm import tqdm
import chromadb
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

# --- CONFIG DIRS ---
DATA_ROOT           = os.path.join(_REPO_ROOT, "data", "hotpotqa")
RAW_DIR             = os.path.join(DATA_ROOT, "raw")
PARSED_DIR          = os.path.join(DATA_ROOT, "parsed")
BASE_PARSED_DIR     = os.path.join(DATA_ROOT, "parsed_txt")
GRAPH_DIR           = os.path.join(DATA_ROOT, "graph")
EMBED_DIR           = os.path.join(DATA_ROOT, "embeddings", "graphmd")
BASE_EMBED_DIR      = os.path.join(DATA_ROOT, "embeddings", "baseline")
GRAPHNOMD_EMBED_DIR = os.path.join(DATA_ROOT, "embeddings", "graphnomd")
RESULTS_DIR         = os.path.join(DATA_ROOT, "results")

for d in [RAW_DIR, PARSED_DIR, BASE_PARSED_DIR, GRAPH_DIR, EMBED_DIR, BASE_EMBED_DIR, GRAPHNOMD_EMBED_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

HOTPOT_URL  = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOT_FILE = os.path.join(RAW_DIR, "hotpot_dev_distractor_v1.json")
SAMPLE_SIZE = 500


# ------------------------------------------------------------------ #
#  DATA                                                                #
# ------------------------------------------------------------------ #

def download_hotpotqa():
    if not os.path.exists(HOTPOT_FILE):
        print(f"Downloading HotpotQA Dev Distractor...")
        urllib.request.urlretrieve(HOTPOT_URL, HOTPOT_FILE)
        print("Download complete.")
    else:
        print("HotpotQA file found.")


def prepare_hotpot_data() -> list:
    print(f"\n[Data Prep] Sampling {SAMPLE_SIZE} questions from HotpotQA...")
    with open(HOTPOT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    random.seed(42)
    sample_data = random.sample(data, min(SAMPLE_SIZE, len(data)))
    qa_list = []

    for item in tqdm(sample_data, desc="Parsing HotpotQA Contexts"):
        q_id     = item["_id"]
        contexts = item["context"]  # list of [title, [sentences]]

        qa_list.append({
            "id":           q_id,
            "question":     item["question"],
            "ground_truth": item["answer"]
        })

        md_path = os.path.join(PARSED_DIR, f"{q_id}.md")
        if not os.path.exists(md_path):
            md_content = f"# Context for {q_id}\n\n"
            for title, sentences in contexts:
                md_content += f"## {title}\n" + " ".join(sentences) + "\n\n"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

        txt_path = os.path.join(BASE_PARSED_DIR, f"{q_id}.txt")
        if not os.path.exists(txt_path):
            txt_content = ""
            for title, sentences in contexts:
                txt_content += f"{title}. " + " ".join(sentences) + "\n"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(txt_content)

    return qa_list


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
    print(">>> INGESTION — Baseline (HOTPOT) <<<")
    print("="*50)
    n = _count(BASE_EMBED_DIR, "baseline_rag")
    if n > 0:
        print(f"Baseline DB exists ({n} chunks). Skipping.")
        return
    BaselineEmbedder(ollama, BASE_PARSED_DIR, BASE_EMBED_DIR, model_name="bge-m3").process_all()


def ingest_graphnomd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph no markdown (HOTPOT) <<<")
    print("="*50)
    n = _count(GRAPHNOMD_EMBED_DIR, "qasper_graph_rag")
    if n > 0:
        print(f"Graph no markdown DB exists ({n} chunks). Skipping.")
        return
    P3Embedder(ollama, txt_dir=BASE_PARSED_DIR, graph_dir=GRAPH_DIR,
               db_dir=GRAPHNOMD_EMBED_DIR, model_name="bge-m3").process_all()


def ingest_graphmd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph with markdown (HOTPOT) <<<")
    print("="*50)
    n = _count(EMBED_DIR, "qasper_graph_rag")
    if n > 0:
        print(f"Graph with markdown DB exists ({n} chunks). Skipping.")
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

    baseline_gen  = BaselineGenerator(ollama, BASE_EMBED_DIR,   embed_model="bge-m3", llm_model="llama3.1:8b")
    graphnomd_gen = Generator(ollama, GRAPHNOMD_EMBED_DIR,       embed_model="bge-m3", llm_model="llama3.1:8b")
    graphmd_gen   = Generator(ollama, EMBED_DIR,                 embed_model="bge-m3", llm_model="llama3.1:8b")

    baseline_results, graphnomd_results, graphmd_results = [], [], []

    for i, qa in enumerate(qa_list):
        print(f"\n[{i+1}/{len(qa_list)}] Q: {qa['question']}")
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
            print(f"=> Saved: hotpotqa/results/{out_csv}")
        except Exception as e:
            print(f"RAGAS error ({label}): {e}")


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main():
    download_hotpotqa()
    qa_list = prepare_hotpot_data()

    ollama = OllamaManager()

    ingest_baseline(ollama)
    ingest_graphnomd(ollama)
    ingest_graphmd(ollama)

    baseline_results, graphnomd_results, graphmd_results = run_generation(ollama, qa_list)

    run_ragas(baseline_results, graphnomd_results, graphmd_results)

    print("\n" + "="*50)
    print("  HOTPOT BENCHMARK COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
