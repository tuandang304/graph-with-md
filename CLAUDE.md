# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG evaluation framework for APWeb/WAIM 2026 paper. Compares four pipelines — a 2×2 factorial of
chunking (fixed-size vs. markdown sections) × graph enrichment (off vs. on) — across seven QA datasets:
- **Baseline**: Fixed-size chunking, no graph
- **Markdown only**: Semantic section chunking (markdown `##` boundaries), no graph
- **Graph no markdown**: Fixed-size chunking + LLM-extracted knowledge graph edges (raw text, no markdown structure)
- **Graph with markdown**: Semantic section chunking + LLM-extracted knowledge graph edges (the proposed method)

Each benchmark runs all four pipelines, saves raw inference as JSONL, then scores with RAGAS.

**Markdown only** was added to answer reviewer weakness W1 (missing markdown-only ablation); it reuses
the Graph-with-markdown ChromaDB index with the graph retrieval channel disabled (`Generator(use_graph=False)`),
so no extra ingestion is needed. See `docs/PAPER_REVISIONS.md` for the full reviewer-response plan and
`README.md` for run instructions.

## Repository Structure

```
experiments/          # Full dataset benchmarks (all 3 pipelines per dataset)
  qasper_benchmark.py            — QASPER dev set, 355 QA pairs (local JSON at data/raw/)
  hotpot_benchmark.py            — HotpotQA dev distractor, 500 samples (auto-downloads JSON)
  pubmedqa_benchmark.py          — PubMedQA pqa_labeled, 500 samples; add 'mini' arg for 10-sample mode
  narrativeqa_benchmark.py       — NarrativeQA test split, 500 samples; 'mini' mode; narrative-domain prompts
  natural_questions_benchmark.py — Natural Questions validation, 500 samples; 'mini' mode; encyclopedic prompts
  quality_benchmark.py           — QuALITY long-document QA, 500 QA pairs sampled (HuggingFace)
  sciq_benchmark.py              — SciQ science QA, 1000 samples with support (HuggingFace)

test/                 # Smoke tests (fast, minimal data)
  test_mini.py            — verifies all 3 generators against _smoketest ChromaDB

src/                  # Pipeline source code
  core/
    ollama_manager.py     — HTTP client for all Ollama model calls
  components/             # Graph with markdown pipeline
    loader.py             — QASPER JSON → .md files
    graph_builder.py      — Qwen 2.5 7B → _graph.json (accepts custom extraction prompt)
    embedder.py           — markdown sections + graph edges → ChromaDB
    generator.py          — two-channel retrieval + Llama generation (accepts custom system prompt)
    evaluator.py          — RAGAS scoring via GPT-4o-mini
  baseline/               # Baseline pipeline
    loader.py             — QASPER JSON → flat .txt files
    embedder.py           — fixed-size chunks → ChromaDB
    generator.py          — flat retrieval + Llama generation
  ablation/               # Graph no markdown pipeline
    p3_embedder.py        — fixed-size chunks + graph edges → ChromaDB
  pipeline.py             — legacy single-dataset Graph-with-markdown orchestrator (not used by experiments/)
  baseline_pipeline.py    — legacy single-dataset Baseline orchestrator (not used by experiments/)

data/                 # All pipeline outputs (gitignored)
  raw/                    — source datasets (QASPER JSON only; other datasets download to their own dirs)
  _smoketest/             — pre-built mini ChromaDB for test_mini.py
  qasper/
    parsed/               — .md files (Graph with markdown)
    parsed_txt/           — .txt files (Baseline, Graph no markdown)
    graph/                — _graph.json files
    embeddings/baseline/  — ChromaDB baseline_rag
    embeddings/graphnomd/ — ChromaDB qasper_graph_rag (Graph no markdown)
    embeddings/graphmd/   — ChromaDB qasper_graph_rag (Graph with markdown)
    results/              — metrics CSVs and raw JSONLs
  hotpotqa/, pubmedqa/, narrativeqa/, naturalquestions/, quality/, sciq/
                          — same structure as qasper/ (each benchmark sets its own DATA_ROOT)
                            hotpotqa/ has raw/ for downloaded JSON; pubmedqa/, narrativeqa/,
                            naturalquestions/ have results/mini/ for 10-sample mini runs
```

Note: root-level `compare_results.py`, `pubmedqa_stats.py`, and `sciq_stats.py` were removed; benchmarks now print comparisons inline.

## Setup

```bash
# Install dependencies (uv manages virtualenv at .venv/, Python 3.12+)
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
uv run python experiments/narrativeqa_benchmark.py
uv run python experiments/natural_questions_benchmark.py
uv run python experiments/quality_benchmark.py
uv run python experiments/sciq_benchmark.py

# Quick validation (10 samples, reuses existing embeddings) — supported by these benchmarks:
uv run python experiments/pubmedqa_benchmark.py mini
uv run python experiments/narrativeqa_benchmark.py mini
uv run python experiments/natural_questions_benchmark.py mini
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

**`src/components/generator.py`** — Two-channel retrieval for Graph with markdown: Channel 1 = `semantic_section` chunks (`sem_k = max(top_k-3, 5)`), Channel 2 = `graph_edge` chunks scoped to papers from Channel 1 (`graph_k = min(5, top_k//2)`). Fallback to unfiltered retrieval when Channel 1 returns empty (Graph no markdown path, which has `baseline_chunk` type). Section chunks are truncated to 800 chars in the prompt. Accepts an optional `system_prompt` so domain benchmarks (NarrativeQA, Natural Questions) can override the default academic-assistant prompt. The `use_graph=False` flag powers the **Markdown only** ablation: Channel 2 is skipped and Channel 1 retrieves the full `top_k` semantic sections (reuses the Graph-with-markdown index).

**`src/components/graph_builder.py`** — Qwen 2.5 7B → `_graph.json`. All benchmarks use `model_name="qwen2.5:7b"`. NarrativeQA and Natural Questions pass a domain-specific extraction prompt (narrative vs. encyclopedic) defined at the top of their benchmark scripts.

**`src/components/evaluator.py`** — RAGAS via GPT-4o-mini judge + `text-embedding-3-small`. Always computes `faithfulness` and `answer_relevancy`; adds `context_precision` and `context_recall` only when `ground_truth` is present in the results. Generation top_k: Baseline=5, Graph pipelines=10.

### Data Paths
- `.env` → `<repo>/.env` (gitignored) — must contain `OPENAI_API_KEY`
- QASPER source → `<repo>/data/raw/qasper-dev-v0.3.json` (only dataset shipped locally; others download via `datasets`/HTTP into their own `data/<dataset>/` dir)
- All outputs → `<repo>/data/<dataset>/` (gitignored)
- ChromaDB collections: `qasper_graph_rag` (Graph with markdown, Graph no markdown — same name across both), `baseline_rag` (Baseline)

### Output File Naming
Within each `data/<dataset>/results/`:
- `baseline_raw.jsonl` / `baseline_metrics.csv`
- `mdonly_raw.jsonl` / `mdonly_metrics.csv` (Markdown only)
- `graphnomd_raw.jsonl` / `graphnomd_metrics.csv`
- `graphmd_raw.jsonl` / `graphmd_metrics.csv`

Raw inference saved as `.jsonl` before RAGAS — re-run RAGAS without re-inference by loading the raw file and calling `Evaluator.evaluate_dataframe()`.
