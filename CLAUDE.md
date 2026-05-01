# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG evaluation framework for APWeb/WAIM 2026 paper. Compares three pipelines on QASPER, HotpotQA, and PubMedQA datasets:
- **Baseline**: Fixed-size chunking, no graph
- **Graph no markdown**: Fixed-size chunking + LLM-extracted knowledge graph edges (raw text, no markdown structure)
- **Graph with markdown**: Semantic section chunking (markdown `##` boundaries) + LLM-extracted knowledge graph edges

## Repository Structure

```
experiments/          # Full dataset benchmarks (all 3 pipelines per dataset)
  qasper_benchmark.py     — QASPER dev set, 355 QA pairs
  hotpot_benchmark.py     — HotpotQA, 500 samples (auto-downloads)
  pubmedqa_benchmark.py   — PubMedQA, 500 samples; add 'mini' arg for 10-sample mode

test/                 # Smoke tests (fast, minimal data)
  test_mini.py            — verifies all 3 generators against _smoketest ChromaDB

src/                  # Pipeline source code
  core/
    ollama_manager.py     — HTTP client for all Ollama model calls
  components/             # Graph with markdown pipeline
    loader.py             — QASPER JSON → .md files
    graph_builder.py      — Qwen 7B → _graph.json
    embedder.py           — markdown sections + graph edges → ChromaDB
    generator.py          — two-channel retrieval + Llama generation
    evaluator.py          — RAGAS scoring via GPT-4o-mini
  baseline/               # Baseline pipeline
    loader.py             — QASPER JSON → flat .txt files
    embedder.py           — fixed-size chunks → ChromaDB
    generator.py          — flat retrieval + Llama generation
  ablation/               # Graph no markdown pipeline
    p3_embedder.py        — fixed-size chunks + graph edges → ChromaDB
  pipeline.py             — Graph with markdown orchestrator (load → embed → query → eval)
  baseline_pipeline.py    — Baseline orchestrator

data/                 # All pipeline outputs (gitignored)
  raw/                    — source datasets (QASPER JSON)
  _smoketest/             — pre-built mini ChromaDB for test_mini.py
  qasper/
    parsed/               — .md files (Graph with markdown)
    parsed_txt/           — .txt files (Baseline, Graph no markdown)
    graph/                — _graph.json files
    embeddings/baseline/  — ChromaDB baseline_rag
    embeddings/graphnomd/ — ChromaDB qasper_graph_rag (Graph no markdown)
    embeddings/graphmd/   — ChromaDB qasper_graph_rag (Graph with markdown)
    results/              — metrics CSVs and raw JSONLs
  hotpotqa/               — same structure as qasper/ (raw/ holds downloaded JSON)
  pubmedqa/               — same structure as qasper/ (results/mini/ for 10-sample runs)
```

## Setup

```bash
# Install dependencies (uv manages virtualenv at .venv/, Python 3.14)
uv sync

# Required: .env file at <repo>/.env with:
# OPENAI_API_KEY=sk-...

# Required: Ollama running locally at http://127.0.0.1:11434 with models pulled:
# ollama pull qwen2.5:7b   # graph extraction
# ollama pull bge-m3       # embeddings
# ollama pull llama3.1:8b  # generation
```

## Running

All scripts run via `uv run` from the **repo root**.

```bash
# Smoke test — verifies all 3 generators work (needs _smoketest DBs)
uv run python test/test_mini.py

# Full benchmarks — ingestion + generation + RAGAS (hours each)
uv run python experiments/qasper_benchmark.py
uv run python experiments/hotpot_benchmark.py
uv run python experiments/pubmedqa_benchmark.py

# PubMedQA quick validation (10 samples, reuses existing embeddings)
uv run python experiments/pubmedqa_benchmark.py mini
```

## Architecture

### Data Flow (Graph with markdown)
```
QASPER JSON → QasperLoader → .md files
                           → GraphBuilder (Qwen 7B) → _graph.json
                           → Embedder (BGE-M3) → ChromaDB (qasper_graph_rag)
                           → Generator (BGE-M3 + Llama 3.1:8B) → answers
                           → Evaluator (RAGAS + GPT-4o-mini) → metrics CSV
```

### Key Modules

**`src/core/ollama_manager.py`** — Single HTTP client for all local model calls. All calls default `keep_alive=0` to immediately free VRAM. **Critical**: RTX 5060 Ti has 16 GB VRAM — never load two models simultaneously. Exception: batch embedding loops use `keep_alive=300` then call `unload_model()` at end.

**`src/components/embedder.py`** — Chunks markdown by `##` section boundaries (merges chunks < 400 chars), adds graph edge strings. Chunk metadata: `{"paper_id": str, "type": "semantic_section"|"graph_edge"}`. Uses `upsert` so re-runs are safe.

**`src/ablation/p3_embedder.py`** — Graph no markdown embedder. `RecursiveCharacterTextSplitter(chunk_size=1000)` on raw `.txt` + graph edges. Chunk metadata type: `"baseline_chunk"`. Same ChromaDB collection name (`qasper_graph_rag`) so Generator class works without changes.

**`src/components/generator.py`** — Two-channel retrieval for Graph with markdown: Channel 1 = `semantic_section` chunks, Channel 2 = `graph_edge` chunks scoped to papers from Channel 1. Fallback to unfiltered retrieval when Channel 1 returns empty (Graph no markdown path, which has `baseline_chunk` type).

### Data Paths
- `.env` → `<repo>/.env` (gitignored) — must contain `OPENAI_API_KEY`
- QASPER source → `<repo>/data/raw/qasper-dev-v0.3.json`
- All outputs → `<repo>/data/<dataset>/` (gitignored)
- ChromaDB collections: `qasper_graph_rag` (Graph with markdown, Graph no markdown), `baseline_rag` (Baseline)

### Output File Naming
Within each `data/<dataset>/results/`:
- `baseline_raw.jsonl` / `baseline_metrics.csv`
- `graphnomd_raw.jsonl` / `graphnomd_metrics.csv`
- `graphmd_raw.jsonl` / `graphmd_metrics.csv`

Raw inference saved as `.jsonl` before RAGAS — re-run RAGAS without re-inference by loading the raw file and calling `Evaluator.evaluate_dataframe()`.
