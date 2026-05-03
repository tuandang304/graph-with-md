"""
Natural Questions Benchmark — Baseline vs Graph no markdown vs Graph with markdown.
Dataset: google-research-datasets/natural_questions validation split — 500 QA pairs (seed=42).
Documents: Wikipedia articles. One .md/.txt file per NQ example id.

Modes:
  Full (default): 500 samples, seed=42, full ingestion + generation + RAGAS.
  Mini:           10 samples,  seed=99, skips ingestion (requires existing embeddings).
                  Run: uv run python experiments/natural_questions_benchmark.py mini

Run: uv run python experiments/natural_questions_benchmark.py
     uv run python experiments/natural_questions_benchmark.py mini
"""
import os
import sys
import re
import json
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
MINI = "mini" in sys.argv

DATA_ROOT           = os.path.join(_REPO_ROOT, "data", "naturalquestions")
PARSED_DIR          = os.path.join(DATA_ROOT, "parsed")
GRAPH_DIR           = os.path.join(DATA_ROOT, "graph")
EMBED_DIR           = os.path.join(DATA_ROOT, "embeddings", "graphmd")
BASE_PARSED_DIR     = os.path.join(DATA_ROOT, "parsed_txt")
BASE_EMBED_DIR      = os.path.join(DATA_ROOT, "embeddings", "baseline")
GRAPHNOMD_EMBED_DIR = os.path.join(DATA_ROOT, "embeddings", "graphnomd")
RESULTS_DIR         = os.path.join(DATA_ROOT, "results")
MINI_RESULTS_DIR    = os.path.join(DATA_ROOT, "results", "mini")

SAMPLE_SIZE = 10  if MINI else 500
SEED        = 99  if MINI else 42

RAW_TEXT_CHAR_LIMIT = 150_000  # Wikipedia pages can be very large
MAX_SECTIONS        = 50
FALLBACK_WINDOW     = 3_000

NQ_GRAPH_PROMPT = (
    "You are an expert knowledge graph extractor for encyclopedic text. "
    "Given Wikipedia article passages, extract the most important entities and relationships. "
    "Focus on: factual statements, named entities (people, places, organizations), "
    "dates, causal relationships, and key attributes. "
    "Output EXCLUSIVELY a JSON array: "
    '[{"source": "...", "target": "...", "relation": "..."}]. '
    "No markdown code blocks. Limit to top 20 relationships."
)

NQ_SYSTEM_PROMPT = (
    "You are a precise question answering assistant. "
    "Answer questions strictly based on the provided Wikipedia passages and knowledge graph facts. "
    "Give concise, factual answers. Do not add information beyond what is given."
)

NQ_BASELINE_PROMPT = (
    "You are a question answering assistant. "
    "Answer based on the provided context. If the context does not contain the answer, say Unknown."
)

for d in [PARSED_DIR, GRAPH_DIR, EMBED_DIR, BASE_PARSED_DIR, BASE_EMBED_DIR,
          GRAPHNOMD_EMBED_DIR, RESULTS_DIR, MINI_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ #
#  DATA                                                                #
# ------------------------------------------------------------------ #

def _extract_text(tokens) -> str:
    """Extract non-HTML text from document tokens (handles both HF formats)."""
    if isinstance(tokens, dict):
        # Columnar HF format: {"token": [...], "is_html": [...]}
        return " ".join(
            t for t, h in zip(tokens.get("token", []), tokens.get("is_html", []))
            if not h
        )
    # Row format: [{"token": "...", "is_html": False}, ...]
    return " ".join(t["token"] for t in tokens if not t.get("is_html", False))


def _extract_tokens_slice(tokens, start: int, end: int) -> str:
    """Extract a token span as plain text, skipping HTML tokens."""
    if isinstance(tokens, dict):
        token_list   = tokens.get("token", [])
        is_html_list = tokens.get("is_html", [])
        return " ".join(
            token_list[i]
            for i in range(start, min(end, len(token_list)))
            if not is_html_list[i]
        )
    return " ".join(
        tokens[i]["token"]
        for i in range(start, min(end, len(tokens)))
        if not tokens[i].get("is_html", False)
    )


def _get_ground_truth(item: dict) -> str:
    """
    Extract ground truth from NQ item.
    Priority: short answer text > yes/no > long answer span.

    HF streaming format (confirmed):
      annotations = {
        "short_answers":  [{"text": [str], ...}, ...],  # list, one dict per annotator
        "yes_no_answer":  [-1 | 0 | 1, ...],            # -1=none, 0=No, 1=Yes
        "long_answer":    [{"start_token": int, "end_token": int, ...}, ...],
      }
    """
    annotations = item.get("annotations", {})
    tokens      = item.get("document", {}).get("tokens", {})

    short_answers = annotations.get("short_answers", [])
    yes_no_list   = annotations.get("yes_no_answer", [])
    long_answers  = annotations.get("long_answer", [])

    # Short answer: first annotator with non-empty text wins
    for sa in short_answers:
        texts = sa.get("text", []) if isinstance(sa, dict) else []
        if texts:
            return texts[0]

    # Yes/no: 1=Yes, 0=No, -1=none
    for yn in yes_no_list:
        if yn == 1:
            return "Yes"
        if yn == 0:
            return "No"

    # Long answer: first annotator with a valid token span
    for la in long_answers:
        if not isinstance(la, dict):
            continue
        st = la.get("start_token", -1)
        en = la.get("end_token",   -1)
        if isinstance(st, int) and st >= 0 and en > st:
            text = _extract_tokens_slice(tokens, st, en)
            if text:
                return text[:400]

    return ""


def _split_wikipedia_sections(raw_text: str) -> list:
    """
    Split text on == Section == markers; fall back to fixed windows.
    Returns list of (title, body) pairs, capped at MAX_SECTIONS.
    """
    text    = raw_text[:RAW_TEXT_CHAR_LIMIT]
    pattern = re.compile(r'(?:^|\n)(={2,3}\s*[^=\n]+\s*={2,3})')
    splits  = list(pattern.finditer(text))

    if len(splits) < 2:
        sections = []
        for i, start in enumerate(range(0, len(text), FALLBACK_WINDOW)):
            sections.append((f"Part {i + 1}", text[start:start + FALLBACK_WINDOW]))
            if len(sections) >= MAX_SECTIONS:
                break
        return sections

    sections = []
    for i, match in enumerate(splits[:MAX_SECTIONS]):
        title = match.group(1).strip().strip("=").strip()
        start = match.end()
        end   = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        body  = text[start:end].strip()
        if body:
            sections.append((title, body))

    return sections or [("Text", text[:FALLBACK_WINDOW])]


def load_naturalquestions() -> list:
    from datasets import load_dataset
    from itertools import islice
    print(f"[Data] Loading Natural Questions validation split (streaming) — {'MINI' if MINI else 'full'} mode...")
    ds   = load_dataset("google-research-datasets/natural_questions", split="validation", streaming=True)
    ds   = ds.shuffle(seed=SEED, buffer_size=10_000)
    data = list(islice(ds, SAMPLE_SIZE))
    print(f"[Data] Using {len(data)} samples (seed={SEED}).")
    return data


def prepare_files(samples: list) -> list:
    """
    Write one .md + one .txt per NQ item (keyed by item id).
    Skips items with no extractable ground truth.
    Returns qa_list with one entry per valid QA pair.
    """
    print(f"\n[Prep] Writing context files ({len(samples)} samples)...")
    qa_list = []
    skipped = 0

    for item in tqdm(samples, desc="Writing files"):
        item_id      = str(item["id"])
        title        = item["document"]["title"]
        tokens       = item["document"]["tokens"]
        question     = item["question"]["text"]
        ground_truth = _get_ground_truth(item)

        if not ground_truth:
            skipped += 1
            continue

        qa_list.append({
            "id":           item_id,
            "question":     question,
            "ground_truth": ground_truth,
        })

        raw_text = _extract_text(tokens)

        md_path = os.path.join(PARSED_DIR, f"{item_id}.md")
        if not os.path.exists(md_path):
            md = f"# {title}\n\n"
            for sec_title, sec_body in _split_wikipedia_sections(raw_text):
                md += f"## {sec_title}\n{sec_body}\n\n"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)

        txt_path = os.path.join(BASE_PARSED_DIR, f"{item_id}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(raw_text[:RAW_TEXT_CHAR_LIMIT])

    print(f"[Prep] {len(qa_list)} valid QA pairs written. {skipped} skipped (no ground truth).")
    return qa_list


# ------------------------------------------------------------------ #
#  INGESTION                                                           #
# ------------------------------------------------------------------ #

def _count(db_dir: str, col: str) -> int:
    try:
        return chromadb.PersistentClient(path=db_dir).get_collection(col).count()
    except Exception:
        return 0


def build_graphs(ollama: OllamaManager):
    print("\n" + "=" * 50)
    print(">>> GRAPH BUILDING — Qwen 7B (NQ) <<<")
    print("=" * 50)
    md_files   = [f for f in os.listdir(PARSED_DIR) if f.endswith(".md")]
    done_files = [f for f in os.listdir(GRAPH_DIR)  if f.endswith("_graph.json")]
    if md_files and len(done_files) >= len(md_files):
        print(f"Graphs complete ({len(done_files)} files). Skipping.")
        return
    print(f"Building graphs: {len(done_files)}/{len(md_files)} done...")
    GraphBuilder(
        ollama, PARSED_DIR, GRAPH_DIR,
        model_name="qwen2.5:7b",
        system_prompt=NQ_GRAPH_PROMPT,
    ).process_all()


def ingest_baseline(ollama: OllamaManager):
    print("\n" + "=" * 50)
    print(">>> INGESTION — Baseline (NQ) <<<")
    print("=" * 50)
    n = _count(BASE_EMBED_DIR, "baseline_rag")
    if n > 0:
        print(f"Baseline DB exists ({n} chunks). Skipping.")
        return
    BaselineEmbedder(ollama, BASE_PARSED_DIR, BASE_EMBED_DIR, model_name="bge-m3").process_all()


def ingest_graphnomd(ollama: OllamaManager):
    print("\n" + "=" * 50)
    print(">>> INGESTION — Graph no markdown (NQ) <<<")
    print("=" * 50)
    n = _count(GRAPHNOMD_EMBED_DIR, "qasper_graph_rag")
    if n > 0:
        print(f"Graph no markdown DB exists ({n} chunks). Skipping.")
        return
    P3Embedder(
        ollama,
        txt_dir=BASE_PARSED_DIR,
        graph_dir=GRAPH_DIR,
        db_dir=GRAPHNOMD_EMBED_DIR,
        model_name="bge-m3",
    ).process_all()


def ingest_graphmd(ollama: OllamaManager):
    print("\n" + "=" * 50)
    print(">>> INGESTION — Graph with markdown (NQ) <<<")
    print("=" * 50)
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
    """Load JSONL checkpoint. Return (records_list, done_ids_set)."""
    if not os.path.exists(path):
        return [], set()
    records, done_ids = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append(rec)
                if "id" in rec:
                    done_ids.add(rec["id"])
    return records, done_ids


def run_generation(ollama: OllamaManager, qa_list: list):
    print("\n" + "=" * 50)
    print(f">>> GENERATION — All 3 Pipelines ({len(qa_list)} questions) <<<")
    print("=" * 50)

    baseline_gen = BaselineGenerator(
        ollama, BASE_EMBED_DIR,
        embed_model="bge-m3", llm_model="llama3.1:8b",
        system_prompt=NQ_BASELINE_PROMPT,
    )
    graphnomd_gen = Generator(
        ollama, GRAPHNOMD_EMBED_DIR,
        embed_model="bge-m3", llm_model="llama3.1:8b",
        system_prompt=NQ_SYSTEM_PROMPT,
    )
    graphmd_gen = Generator(
        ollama, EMBED_DIR,
        embed_model="bge-m3", llm_model="llama3.1:8b",
        system_prompt=NQ_SYSTEM_PROMPT,
    )

    out_dir   = MINI_RESULTS_DIR if MINI else RESULTS_DIR
    raw_paths = {
        "Baseline":            os.path.join(out_dir, "baseline_raw.jsonl"),
        "Graph no markdown":   os.path.join(out_dir, "graphnomd_raw.jsonl"),
        "Graph with markdown": os.path.join(out_dir, "graphmd_raw.jsonl"),
    }

    baseline_results,  baseline_done  = _load_existing_raw(raw_paths["Baseline"])
    graphnomd_results, graphnomd_done = _load_existing_raw(raw_paths["Graph no markdown"])
    graphmd_results,   graphmd_done   = _load_existing_raw(raw_paths["Graph with markdown"])

    if baseline_done or graphnomd_done or graphmd_done:
        print(f"Resuming: Baseline={len(baseline_done)}, GraphNoMD={len(graphnomd_done)}, GraphMD={len(graphmd_done)} done.")

    f_base = open(raw_paths["Baseline"],            "a", encoding="utf-8")
    f_nomd = open(raw_paths["Graph no markdown"],   "a", encoding="utf-8")
    f_md   = open(raw_paths["Graph with markdown"], "a", encoding="utf-8")

    try:
        for i, qa in enumerate(qa_list):
            print(f"\n[{i+1}/{len(qa_list)}] Q: {qa['question'][:100]}")
            for gen, store, done_ids, fh, top_k, label in [
                (baseline_gen,  baseline_results,  baseline_done,  f_base, 5,  "Baseline"),
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
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                except Exception as e:
                    print(f"  {label} error: {e}")
    finally:
        f_base.close()
        f_nomd.close()
        f_md.close()

    print(f"\nGeneration complete — Baseline:{len(baseline_results)}, GraphNoMD:{len(graphnomd_results)}, GraphMD:{len(graphmd_results)}")
    return baseline_results, graphnomd_results, graphmd_results


# ------------------------------------------------------------------ #
#  EVALUATION                                                          #
# ------------------------------------------------------------------ #

def run_ragas(baseline_results, graphnomd_results, graphmd_results):
    print("\n" + "=" * 50)
    print(">>> RAGAS EVALUATION (GPT-4o-mini) <<<")
    print("=" * 50)

    out_dir   = MINI_RESULTS_DIR if MINI else RESULTS_DIR
    evaluator = Evaluator(use_local_model=False)
    scores    = {}

    for label, results, out_csv in [
        ("Baseline",            baseline_results,  "baseline_metrics.csv"),
        ("Graph no markdown",   graphnomd_results, "graphnomd_metrics.csv"),
        ("Graph with markdown", graphmd_results,   "graphmd_metrics.csv"),
    ]:
        csv_path = os.path.join(out_dir, out_csv)
        print(f"\n--- {label} ---")
        if os.path.exists(csv_path):
            print(f"  CSV exists, skipping. ({csv_path})")
            try:
                df = pd.read_csv(csv_path)
                scores[label] = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().to_dict()
            except Exception:
                pass
            continue
        if not results:
            print(f"  No results to evaluate, skipping.")
            continue
        try:
            df = evaluator.evaluate_dataframe(results)
            df.to_csv(csv_path, index=False)
            print(f"=> Saved: naturalquestions/results/{out_csv}")
            scores[label] = df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean().to_dict()
        except Exception as e:
            print(f"RAGAS error ({label}): {e}")

    return scores


def print_comparison(scores: dict):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n" + "=" * 72)
    print(f"  NATURAL QUESTIONS RESULTS ({'MINI — ' + str(SAMPLE_SIZE) + ' samples' if MINI else str(SAMPLE_SIZE) + ' samples'})")
    print("=" * 72)
    print(f"{'Pipeline':<30}" + "".join(f"{m[:10]:>12}" for m in metrics))
    print("-" * 72)
    for label, means in scores.items():
        print(f"{label:<30}" + "".join(f"{means.get(m, float('nan')):>12.4f}" for m in metrics))
    print("=" * 72)
    if "Graph with markdown" in scores and "Graph no markdown" in scores:
        gmd, gnomd = scores["Graph with markdown"], scores["Graph no markdown"]
        print("\n  GraphMD vs GraphNoMD delta (positive = GraphMD better):")
        for m in metrics:
            d = gmd.get(m, 0) - gnomd.get(m, 0)
            print(f"    {m:<22} {'+' if d >= 0 else ''}{d:.4f}")


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main():
    if MINI:
        print(f"\n[MINI MODE] Checking embeddings exist (seed={SEED}, n={SAMPLE_SIZE})...")
        for name, path, col in [
            ("Baseline",            BASE_EMBED_DIR,      "baseline_rag"),
            ("Graph no markdown",   GRAPHNOMD_EMBED_DIR, "qasper_graph_rag"),
            ("Graph with markdown", EMBED_DIR,           "qasper_graph_rag"),
        ]:
            cnt = _count(path, col)
            if cnt == 0:
                print(f"[ERROR] {name} DB missing at {path}. Run full benchmark first.")
                sys.exit(1)
            print(f"  {name}: {cnt} chunks OK")

    samples = load_naturalquestions()
    qa_list = prepare_files(samples)

    if not qa_list:
        print("[ERROR] No valid QA pairs after filtering. Check dataset loading.")
        sys.exit(1)

    ollama = OllamaManager()

    if not MINI:
        build_graphs(ollama)
        ingest_baseline(ollama)
        ingest_graphnomd(ollama)
        ingest_graphmd(ollama)

    baseline_results, graphnomd_results, graphmd_results = run_generation(ollama, qa_list)

    scores = run_ragas(baseline_results, graphnomd_results, graphmd_results)
    if scores:
        print_comparison(scores)

    print("\n" + "=" * 50)
    print("  NATURAL QUESTIONS BENCHMARK COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
