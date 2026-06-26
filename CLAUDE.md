# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG evaluation framework for APWeb/WAIM 2026 paper. Compares four pipelines — a 2×2 factorial of
chunking (fixed-size vs. markdown sections) × graph enrichment (off vs. on) — across seven QA datasets:
- **Baseline**: Fixed-size chunking, no graph
- **Markdown only**: Semantic section chunking (markdown `##` boundaries), no graph
- **Graph no markdown**: Fixed-size chunking + NetworkX Knowledge Graph (multi-hop traversal, subgraph extraction)
- **Graph with markdown**: Semantic section chunking + NetworkX Knowledge Graph + hybrid two-channel retrieval (the proposed method)

Each benchmark runs all four pipelines, saves raw inference as JSONL, then scores with RAGAS.

**Markdown only** was added to answer reviewer weakness W1 (missing markdown-only ablation); it reuses
the Graph-with-markdown ChromaDB index with the graph retrieval channel disabled (`Generator(use_graph=False)`),
so no extra ingestion is needed. See `docs/PAPER_REVISIONS.md` for the full reviewer-response plan and
`README.md` for run instructions.

## Repository Structure

```
experiments/          # Full dataset benchmarks (all 4 pipelines per dataset)
  qasper_benchmark.py            — QASPER dev set, 355 QA pairs (local JSON at data/raw/)
  hotpot_benchmark.py            — HotpotQA dev distractor, 500 samples (auto-downloads JSON)
  pubmedqa_benchmark.py          — PubMedQA pqa_labeled, 500 samples; add 'mini' arg for 10-sample mode
  narrativeqa_benchmark.py       — NarrativeQA test split, 500 samples; 'mini' mode; narrative-domain prompts
  natural_questions_benchmark.py — Natural Questions validation, 500 samples; 'mini' mode; encyclopedic prompts
  quality_benchmark.py           — QuALITY long-document QA, 500 QA pairs sampled (HuggingFace)
  sciq_benchmark.py              — SciQ science QA, 1000 samples with support (HuggingFace)

test/                 # Smoke tests (fast, minimal data)
  test_mini.py            — KnowledgeGraph unit tests + verifies all 3 generators against _smoketest data

src/                  # Pipeline source code
  core/
    ollama_manager.py     — HTTP client for all Ollama model calls
  components/             # Graph with markdown pipeline
    knowledge_graph.py    — NetworkX DiGraph: multi-hop traversal, subgraph extraction, shortest path, entity matching
    loader.py             — QASPER JSON → .md files
    graph_builder.py      — Qwen 2.5 7B → _graph.json + _graph.graphml (NetworkX)
    embedder.py           — markdown sections + node-centric graph context → ChromaDB
    generator.py          — hybrid retrieval (vector + graph traversal) + Llama generation
    evaluator.py          — RAGAS scoring via GPT-4o-mini
  baseline/               # Baseline pipeline
    loader.py             — QASPER JSON → flat .txt files
    embedder.py           — fixed-size chunks → ChromaDB
    generator.py          — flat retrieval + Llama generation
  ablation/               # Graph no markdown pipeline
    p3_embedder.py        — fixed-size chunks + node-centric graph context → ChromaDB
  pipeline.py             — legacy single-dataset Graph-with-markdown orchestrator (not used by experiments/)
  baseline_pipeline.py    — legacy single-dataset Baseline orchestrator (not used by experiments/)

data/                 # All pipeline outputs (gitignored)
  raw/                    — source datasets (QASPER JSON only; other datasets download to their own dirs)
  _smoketest/             — pre-built mini data for test_mini.py
  qasper/
    parsed/               — .md files (Graph with markdown)
    parsed_txt/           — .txt files (Baseline, Graph no markdown)
    graph/                — _graph.json + _graph.graphml (NetworkX) files
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
                           → GraphBuilder (Qwen 7B) → _graph.json + _graph.graphml (NetworkX)
                           → Embedder (BGE-M3) → ChromaDB (semantic_section + graph_context)
                           → Generator (BGE-M3 + Llama 3.1:8B)
                               Channel 1: vector search (semantic sections)
                               Channel 2: NetworkX graph traversal (multi-hop subgraph)
                               Channel 2b: vector fallback (graph_context chunks)
                           → Evaluator (RAGAS + GPT-4o-mini) → metrics CSV
```

### Key Modules

**`src/core/ollama_manager.py`** — Single HTTP client for all local model calls. All calls default `keep_alive=0` to immediately free VRAM. **Critical**: RTX 5060 Ti has 16 GB VRAM — never load two models simultaneously. Exception: batch embedding loops use `keep_alive=300` then call `unload_model()` at end.

**`src/components/knowledge_graph.py`** — NetworkX `DiGraph` wrapper. Provides: `add_triplets()` to build graph, `save()`/`load()` for `.graphml` persistence, `get_neighbors(entity, hops=2)` for multi-hop ego-graph, `get_subgraph_for_entities()` for union of ego-graphs, `shortest_path()`, `find_matching_entities()` for fuzzy entity matching, `extract_query_entities()` for n-gram-based entity extraction from questions, `get_entity_context()` for generating rich node-centric text, and `subgraph_to_text()` for LLM-readable structured output. Entity IDs are normalized (lowercase, stripped) for deduplication; original surface forms stored as node attributes.

**`src/components/graph_builder.py`** — Qwen 2.5 7B → `_graph.json` + `_graph.graphml`. After LLM extracts triplets, they are saved as JSON (backward compatible) AND inserted into a `KnowledgeGraph` instance which is persisted as `.graphml`. All benchmarks use `model_name="qwen2.5:7b"`. NarrativeQA and Natural Questions pass a domain-specific extraction prompt.

**`src/components/embedder.py`** — Chunks markdown by `##` section boundaries (merges chunks < 400 chars), then generates **node-centric graph context** for each entity in the KnowledgeGraph (2-hop neighborhood descriptions). Chunk metadata: `{"paper_id": str, "type": "semantic_section"|"graph_context", "entity": str}`. Uses `upsert` so re-runs are safe.

**`src/ablation/p3_embedder.py`** — Graph no markdown embedder. `RecursiveCharacterTextSplitter(chunk_size=1000)` on raw `.txt` + node-centric graph context from KnowledgeGraph. Chunk metadata type: `"baseline_chunk"` or `"graph_context"`. Same ChromaDB collection name (`qasper_graph_rag`) so Generator class works without changes.

**`src/components/generator.py`** — Hybrid two-channel retrieval for Graph with markdown: Channel 1 = vector search for `semantic_section` chunks (`sem_k = max(top_k-3, 5)`). Channel 2 = **NetworkX structural traversal**: extracts query entities, matches them against graph nodes, performs 2-hop subgraph extraction, converts to structured text (e.g., `Entity "BERT": → (is_a) → Pre-trained LM`). Falls back to vector-retrieved `graph_context` chunks scoped to papers from Channel 1. New `graph_dir` parameter loads the KnowledgeGraph. Fallback to unfiltered retrieval when Channel 1 returns empty (Graph no markdown path). The `use_graph=False` flag powers the **Markdown only** ablation.

**`src/components/evaluator.py`** — RAGAS via GPT-4o-mini judge + `text-embedding-3-small`. Always computes `faithfulness` and `answer_relevancy`; adds `context_precision` and `context_recall` only when `ground_truth` is present in the results. Generation top_k: Baseline=5, Graph pipelines=10.

### Data Paths
- `.env` → `<repo>/.env` (gitignored) — must contain `OPENAI_API_KEY`
- QASPER source → `<repo>/data/raw/qasper-dev-v0.3.json` (only dataset shipped locally; others download via `datasets`/HTTP into their own `data/<dataset>/` dir)
- All outputs → `<repo>/data/<dataset>/` (gitignored)
- Graph files: `_graph.json` (backward-compatible triplet JSON) + `_graph.graphml` (NetworkX)
- ChromaDB collections: `qasper_graph_rag` (Graph with markdown, Graph no markdown — same name across both), `baseline_rag` (Baseline)
- Chunk metadata types: `semantic_section`, `graph_context` (node-centric), `baseline_chunk`

### Output File Naming
Within each `data/<dataset>/results/`:
- `baseline_raw.jsonl` / `baseline_metrics.csv`
- `mdonly_raw.jsonl` / `mdonly_metrics.csv` (Markdown only)
- `graphnomd_raw.jsonl` / `graphnomd_metrics.csv`
- `graphmd_raw.jsonl` / `graphmd_metrics.csv`

Raw inference saved as `.jsonl` before RAGAS — re-run RAGAS without re-inference by loading the raw file and calling `Evaluator.evaluate_dataframe()`.
