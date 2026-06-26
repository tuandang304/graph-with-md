# MD-GraphRAG

Evaluation framework for **MD-GraphRAG** (APWeb/WAIM 2026): *Synergizing Markdown-Aware Chunking and
Knowledge Graph Enrichment for Retrieval-Augmented Generation*.

It compares **four** RAG pipelines as a 2×2 factorial of *chunking* × *graph enrichment* across seven
QA datasets, scoring every run with [RAGAS](https://docs.ragas.io/).

| Chunking ↓ \ Graph → | **No graph** | **+ Knowledge graph** |
|---|---|---|
| **Fixed-size**        | `Baseline` (NaiveRAG)   | `Graph no markdown` |
| **Markdown sections** | `Markdown only`         | `Graph with markdown` (MD-GraphRAG) |

- **Baseline** — fixed-size chunking, dense retrieval, no graph.
- **Markdown only** — semantic chunking on Markdown `##` boundaries, no graph.
- **Graph no markdown** — fixed-size chunking + **NetworkX Knowledge Graph** (multi-hop traversal, subgraph extraction).
- **Graph with markdown** — semantic chunking + **NetworkX Knowledge Graph** + hybrid two-channel retrieval:
  structural graph traversal + vector search (the proposed method).

The knowledge graph is a **true NetworkX `DiGraph`** — not just embedded triplet strings. LLM-extracted
triplets are stored as nodes and edges supporting multi-hop traversal, shortest path, subgraph
extraction, and entity-aware retrieval at query time.

The `Markdown only` cell isolates the contribution of the graph (it reuses the Graph-with-markdown
index with the graph retrieval channel disabled), so the four cells together decompose the overall
gain into a *chunking* effect, a *graph* effect, and their *interaction*. See
[`docs/PAPER_REVISIONS.md`](docs/PAPER_REVISIONS.md) for how this maps to the reviewer feedback.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **[uv](https://docs.astral.sh/uv/)** | Manages the virtualenv (`.venv/`) and Python 3.12. All commands below run through `uv run`. |
| **[Ollama](https://ollama.com/)** | Local model server at `http://127.0.0.1:11434`. Used for graph extraction, embeddings, and generation. |
| **GPU** | Developed on an RTX 5060 Ti (16 GB). Models are loaded one at a time (`keep_alive=0`) so they never co-reside in VRAM. |
| **OpenAI API key** | RAGAS uses `gpt-4o-mini` as judge and `text-embedding-3-small` for scoring. |
| **Internet** | Most datasets auto-download (HuggingFace / HTTP) on first run. QASPER is the exception (see §3). |

### 1.1 Install Python dependencies

```bash
uv sync
```

This creates `.venv/` and installs everything in `pyproject.toml`.

### 1.2 Configure the OpenAI key

Create a `.env` file in the repo root (it is gitignored):

```
OPENAI_API_KEY=sk-...
```

### 1.3 Pull the Ollama models

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M   # knowledge-graph extraction & answer generation
ollama pull bge-m3                       # embeddings (ingestion + query)
```

Make sure the Ollama server is running (`ollama serve`, or the desktop app) before launching a
benchmark.

---

## 2. Quick smoke test (no GPU-heavy ingestion)

Verifies the KnowledgeGraph component and all generators work against the pre-built mini data in
`data/_smoketest/`:

```bash
uv run python test/test_mini.py
```

The smoke test includes a standalone KnowledgeGraph unit test (build, traverse, subgraph, shortest
path, entity matching, save/load) that runs without Ollama.

---

## 3. Datasets

All commands are run **from the repo root**. Each benchmark writes everything under
`data/<dataset>/` (gitignored).

| Benchmark | Dataset | Source | Samples |
|---|---|---|---|
| `qasper_benchmark.py`            | QASPER dev            | **manual** local JSON | 355 QA pairs |
| `hotpot_benchmark.py`            | HotpotQA dev distractor | auto-download (HTTP) | 500 |
| `pubmedqa_benchmark.py`          | PubMedQA `pqa_labeled` | auto (HuggingFace) | 500 (full) / 10 (mini) |
| `narrativeqa_benchmark.py`       | NarrativeQA test       | auto (HuggingFace) | 500 (full) / 10 (mini) |
| `natural_questions_benchmark.py` | Natural Questions val  | auto (HuggingFace) | 500 (full) / 10 (mini) |
| `quality_benchmark.py`           | QuALITY                | auto (HuggingFace) | 500 |
| `sciq_benchmark.py`              | SciQ                   | auto (HuggingFace) | 1000 |

**QASPER only** must be placed manually:

```
data/raw/qasper-dev-v0.3.json
```

(Download from the [QASPER release](https://allenai.org/data/qasper).)

---

## 4. Running a full benchmark

A full run executes four stages automatically, skipping any that are already complete:

1. **Prepare files** — dataset → `.md` (semantic) and `.txt` (flat) per document.
2. **Build Knowledge Graph** — LLM extraction (Qwen 2.5 7B) → triplet JSON + **NetworkX `.graphml`**
   files. The graph supports multi-hop traversal, subgraph extraction, and shortest path.
3. **Ingest** — embed text chunks + **node-centric graph context** (BGE-M3) into three ChromaDB
   indexes: `baseline`, `graphnomd`, `graphmd`.
4. **Generate** — answer every question with all four pipelines (Llama 3.1 8B) using **hybrid
   retrieval**: vector search (Channel 1) + structural graph traversal (Channel 2). Checkpointing
   raw output to JSONL.
5. **Evaluate** — score each pipeline with RAGAS and write per-pipeline CSVs.

```bash
uv run python experiments/qasper_benchmark.py
uv run python experiments/hotpot_benchmark.py
uv run python experiments/pubmedqa_benchmark.py
uv run python experiments/narrativeqa_benchmark.py
uv run python experiments/natural_questions_benchmark.py
uv run python experiments/quality_benchmark.py
uv run python experiments/sciq_benchmark.py
```

Each full run takes **hours** (the graph-extraction and generation stages dominate). They are
**resumable**: ingestion skips indexes that already exist, and generation skips questions already
present in the raw JSONL, so you can safely stop (`Ctrl-C`) and re-run the same command.

> Note: `Markdown only` adds **no extra ingestion** — it reuses the `graphmd` index with the graph
> retrieval channel disabled. It only adds one more generation + evaluation pass.

### 4.1 Mini / quick-validation mode

`pubmedqa`, `narrativeqa`, and `natural_questions` support a 10-sample mode that **reuses existing
embeddings** (run the full benchmark at least once first) and writes to `results/mini/`:

```bash
uv run python experiments/pubmedqa_benchmark.py mini
uv run python experiments/narrativeqa_benchmark.py mini
uv run python experiments/natural_questions_benchmark.py mini
```

---

## 5. Outputs

Per dataset, under `data/<dataset>/results/` (mini runs go to `results/mini/`):

| File | Pipeline |
|---|---|
| `baseline_raw.jsonl`  / `baseline_metrics.csv`  | Baseline |
| `mdonly_raw.jsonl`    / `mdonly_metrics.csv`    | Markdown only |
| `graphnomd_raw.jsonl` / `graphnomd_metrics.csv` | Graph no markdown |
| `graphmd_raw.jsonl`   / `graphmd_metrics.csv`   | Graph with markdown |

- `*_raw.jsonl` — one record per question: `question`, `answer`, `contexts`, `ground_truth` (plus
  `id`/`story_id` on datasets that track them). Saved **before** RAGAS so you can re-score without
  re-running inference.
- `*_metrics.csv` — per-question RAGAS scores: `faithfulness`, `answer_relevancy`, and (when ground
  truth is available) `context_precision`, `context_recall`.

The console prints a summary table plus two deltas: **Markdown effect** (`GraphMD − GraphNoMD`) and
**Graph effect** (`GraphMD − MarkdownOnly`).

### 5.1 Re-running RAGAS only (no re-inference)

```python
import pandas as pd
from src.components.evaluator import Evaluator

records = pd.read_json("data/qasper/results/graphmd_raw.jsonl", lines=True).to_dict("records")
df = Evaluator(use_local_model=False).evaluate_dataframe(records)
df.to_csv("data/qasper/results/graphmd_metrics.csv", index=False)
```

---

## 6. Repository layout

```
experiments/   # one full benchmark script per dataset (all 4 pipelines)
test/          # test_mini.py smoke test (includes KnowledgeGraph unit tests)
src/
  core/ollama_manager.py     # single HTTP client for all Ollama calls (VRAM-aware)
  components/                # graph-with-markdown pipeline
    knowledge_graph.py       # NetworkX KnowledgeGraph (multi-hop, subgraph, shortest path)
    loader.py                # QASPER JSON → .md files
    graph_builder.py         # LLM → triplet JSON + NetworkX .graphml
    embedder.py              # semantic sections + node-centric graph context → ChromaDB
    generator.py             # hybrid retrieval (vector + graph traversal) + LLM generation
    evaluator.py             # RAGAS scoring via GPT-4o-mini
  baseline/                  # baseline pipeline (loader, embedder, generator)
  ablation/p3_embedder.py    # graph-no-markdown embedder (fixed chunks + KG context)
data/          # all outputs (gitignored); QASPER source goes in data/raw/
  <dataset>/graph/           # _graph.json + _graph.graphml (NetworkX) per document
docs/
  REVIEW.md            # reviewer feedback
  PAPER_REVISIONS.md   # action plan responding to the reviews
```

See [`CLAUDE.md`](CLAUDE.md) for a deeper architecture description.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY not found` during RAGAS | Create `.env` with the key (§1.2); inference still runs and raw JSONL is saved, only scoring fails. |
| Connection refused to `127.0.0.1:11434` | Start Ollama (`ollama serve`) and confirm the three models are pulled. |
| Ollama 500 on embeddings | Usually an over-long/malformed chunk; the embedder truncates to 15k chars and skips bad chunks automatically. |
| `[ERROR] ... DB missing` in mini mode | Run the dataset's full benchmark once before using `mini`. |
| QASPER `JSON not found` | Place `qasper-dev-v0.3.json` in `data/raw/` (§3). |
| Re-running is slow | It shouldn't re-do completed work — ingestion and generation both resume. Delete a `data/<dataset>/` subfolder to force a clean rebuild. |
