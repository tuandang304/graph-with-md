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
from src.ablation.graph_no_markdown_embedder import GraphNoMarkdownEmbedder
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
    GraphNoMarkdownEmbedder(ollama, txt_dir=BASE_PARSED_DIR, graph_dir=GRAPH_DIR,
                            db_dir=GRAPHNOMD_EMBED_DIR, model_name="bge-m3").process_all()


def build_graphs(ollama: OllamaManager):
    """Build knowledge graphs — must run before graphnomd and graphmd ingestion."""
    print("\n" + "="*50)
    print(">>> GRAPH BUILDING — Qwen 7B (HOTPOT) <<<")
    print("="*50)
    md_files  = [f for f in os.listdir(PARSED_DIR) if f.endswith('.md')]
    done_files = [f for f in os.listdir(GRAPH_DIR)  if f.endswith('_graph.json')]
    if len(md_files) > 0 and len(done_files) >= len(md_files):
        print(f"Graphs complete ({len(done_files)} files). Skipping.")
        return
    print(f"Building graphs: {len(done_files)}/{len(md_files)} done...")
    GraphBuilder(ollama, PARSED_DIR, GRAPH_DIR, model_name="qwen2.5:7b-instruct-q4_K_M").process_all()


def ingest_graphmd(ollama: OllamaManager):
    print("\n" + "="*50)
    print(">>> INGESTION — Graph with markdown (HOTPOT) <<<")
    print("="*50)
    n = _count(EMBED_DIR, "qasper_graph_rag")
    if n > 0:
        print(f"Graph with markdown DB exists ({n} chunks). Skipping.")
        return
    print("[Embedder] BGE-M3 — semantic sections + graph edges...")
    Embedder(ollama, PARSED_DIR, GRAPH_DIR, EMBED_DIR, model_name="bge-m3").process_all()


# ------------------------------------------------------------------ #
#  GENERATION                                                          #
# ------------------------------------------------------------------ #

def _load_existing_raw(path: str) -> tuple:
    """Load JSONL checkpoint, return (records_list, done_ids_set)."""
    if not os.path.exists(path):
        return [], set()
    records, done_ids = [], set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append(rec)
                if "id" in rec:
                    done_ids.add(rec["id"])
    return records, done_ids


def run_generation(ollama: OllamaManager, qa_list: list):
    print("\n" + "="*50)
    print(f">>> GENERATION — All 3 Pipelines ({len(qa_list)} questions) <<<")
    print("="*50)

    baseline_gen  = BaselineGenerator(ollama, BASE_EMBED_DIR,   embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M")
    mdonly_gen    = Generator(ollama, EMBED_DIR,                 embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M", use_graph=False)
    graphnomd_gen = Generator(ollama, GRAPHNOMD_EMBED_DIR,       embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M", graph_dir=GRAPH_DIR)
    graphmd_gen   = Generator(ollama, EMBED_DIR,                 embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M", graph_dir=GRAPH_DIR)

    raw_paths = {
        "Baseline":            os.path.join(RESULTS_DIR, "baseline_raw.jsonl"),
        "Markdown only":       os.path.join(RESULTS_DIR, "mdonly_raw.jsonl"),
        "Graph no markdown":   os.path.join(RESULTS_DIR, "graphnomd_raw.jsonl"),
        "Graph with markdown": os.path.join(RESULTS_DIR, "graphmd_raw.jsonl"),
    }

    baseline_results,  baseline_done  = _load_existing_raw(raw_paths["Baseline"])
    mdonly_results,    mdonly_done    = _load_existing_raw(raw_paths["Markdown only"])
    graphnomd_results, graphnomd_done = _load_existing_raw(raw_paths["Graph no markdown"])
    graphmd_results,   graphmd_done   = _load_existing_raw(raw_paths["Graph with markdown"])

    if baseline_done or mdonly_done or graphnomd_done or graphmd_done:
        print(f"Resuming: Baseline={len(baseline_done)}, MDOnly={len(mdonly_done)}, GraphNoMD={len(graphnomd_done)}, GraphMD={len(graphmd_done)} already done.")

    # Append mode — write each answer immediately so crashes don't lose progress
    f_base = open(raw_paths["Baseline"],            'a', encoding='utf-8')
    f_mdo  = open(raw_paths["Markdown only"],        'a', encoding='utf-8')
    f_nomd = open(raw_paths["Graph no markdown"],   'a', encoding='utf-8')
    f_md   = open(raw_paths["Graph with markdown"], 'a', encoding='utf-8')

    try:
        for i, qa in enumerate(qa_list):
            print(f"\n[{i+1}/{len(qa_list)}] Q: {qa['question']}")
            for gen, store, done_ids, fh, top_k, label in [
                (baseline_gen,  baseline_results,  baseline_done,  f_base, 5,  "Baseline"),
                (mdonly_gen,    mdonly_results,    mdonly_done,    f_mdo,  10, "Markdown only"),
                (graphnomd_gen, graphnomd_results, graphnomd_done, f_nomd, 10, "Graph no markdown"),
                (graphmd_gen,   graphmd_results,   graphmd_done,   f_md,   10, "Graph with markdown"),
            ]:
                if qa["id"] in done_ids:
                    print(f"  {label}: already done, skip.")
                    continue
                try:
                    res = gen.query(qa["question"], top_k=top_k)
                    record = {
                        "id":           qa["id"],
                        "question":     qa["question"],
                        "answer":       res["answer"],
                        "contexts":     res["context"],
                        "ground_truth": qa["ground_truth"],
                    }
                    store.append(record)
                    done_ids.add(qa["id"])
                    fh.write(json.dumps(record, ensure_ascii=False) + '\n')
                    fh.flush()
                except Exception as e:
                    print(f"  {label} error: {e}")
    finally:
        f_base.close()
        f_mdo.close()
        f_nomd.close()
        f_md.close()

    print(f"\nGeneration complete — Baseline:{len(baseline_results)}, MDOnly:{len(mdonly_results)}, GraphNoMD:{len(graphnomd_results)}, GraphMD:{len(graphmd_results)}")
    return baseline_results, mdonly_results, graphnomd_results, graphmd_results


# ------------------------------------------------------------------ #
#  EVALUATION                                                          #
# ------------------------------------------------------------------ #

def run_ragas(baseline_results, mdonly_results, graphnomd_results, graphmd_results):
    print("\n" + "="*50)
    print(">>> RAGAS EVALUATION (Local Qwen) <<<")
    print("="*50)

    evaluator = Evaluator(use_local_model=True)

    for label, results, out_csv in [
        ("Baseline",            baseline_results,  "baseline_metrics.csv"),
        ("Markdown only",       mdonly_results,    "mdonly_metrics.csv"),
        ("Graph no markdown",   graphnomd_results, "graphnomd_metrics.csv"),
        ("Graph with markdown", graphmd_results,   "graphmd_metrics.csv"),
    ]:
        csv_path = os.path.join(RESULTS_DIR, out_csv)
        print(f"\n--- {label} ---")
        if os.path.exists(csv_path):
            print(f"  CSV exists, skipping. ({csv_path})")
            continue
        if not results:
            print(f"  No results to evaluate, skipping.")
            continue
        try:
            df = evaluator.evaluate_dataframe(results)
            df.to_csv(csv_path, index=False)
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

    # Graphs must exist before graphnomd and graphmd ingestion
    build_graphs(ollama)
    ingest_baseline(ollama)
    ingest_graphnomd(ollama)
    ingest_graphmd(ollama)

    baseline_results, mdonly_results, graphnomd_results, graphmd_results = run_generation(ollama, qa_list)

    run_ragas(baseline_results, mdonly_results, graphnomd_results, graphmd_results)

    print("\n" + "="*50)
    print("  HOTPOT BENCHMARK COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
