# Research Context — APWeb/WAIM 2026

## Research Question

Does combining document structure (markdown section boundaries) with LLM-extracted knowledge graphs improve RAG retrieval quality?

## Hypothesis

1. Semantic chunking at `##` section boundaries preserves coherence better than fixed-size cuts.
2. Knowledge graph edges add relational facts that dense text alone misses.
3. Combining both → highest RAGAS scores.

---

## Ablation Design

Three pipelines isolate exactly two variables: chunking strategy and graph enrichment.

| Pipeline | Chunking | Knowledge Graph |
|---|---|---|
| Baseline | Fixed-size 1000 chars, 200 overlap | No |
| Graph no markdown | Fixed-size 1000 chars, 200 overlap | Yes |
| Graph with markdown | Semantic sections (`##` boundaries, merge < 400 chars) | Yes |

---

## Technical Stack (all local except evaluator)

| Role | Model |
|---|---|
| Graph extraction | Qwen 2.5:7B (Ollama) |
| Embeddings | BGE-M3 (Ollama) |
| Generation | Llama 3.1:8B (Ollama) |
| Evaluation | RAGAS + GPT-4o-mini (OpenAI) |
| Vector DB | ChromaDB (persistent, local) |

Hardware constraint: RTX 5060 Ti 16 GB VRAM — never load two models simultaneously.

---

## Pipeline Architecture

### Graph with markdown (full pipeline)

```
QASPER/HotpotQA/PubMedQA
  → QasperLoader → .md files (one per document, ## section headers)
  → GraphBuilder (Qwen 7B) → _graph.json (up to 20 relations per doc)
  → Embedder (BGE-M3)
      ├── semantic_section chunks → ChromaDB (qasper_graph_rag)
      └── graph_edge chunks       → ChromaDB (qasper_graph_rag)
  → Generator (two-channel retrieval)
      ├── Channel 1: query → top-7 semantic_section chunks
      └── Channel 2: query → top-3 graph_edge chunks scoped to papers from Channel 1
  → Llama 3.1:8B → answer
  → RAGAS (GPT-4o-mini) → metrics CSV
```

### Key design: two-channel retrieval

Channel 2 scopes graph_edge lookup to `paper_id` values returned by Channel 1.
Prevents off-topic relations from diluting `context_precision`.
Fallback to unfiltered retrieval when Channel 1 returns empty (handles Graph no markdown path transparently).

### Graph no markdown

Same as above but:
- Input: raw `.txt` (no markdown structure)
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=1000)` on flat text
- Chunk metadata type: `baseline_chunk` (not `semantic_section`)
- Generator fallback path activates automatically

### Baseline

- Input: raw `.txt`
- Chunking: `RecursiveCharacterTextSplitter(chunk_size=1000)`
- No graph edges
- Separate ChromaDB collection: `baseline_rag`
- Simpler generator: flat top-k retrieval, no channel split

---

## Datasets

| Dataset | Samples | Task type |
|---|---|---|
| QASPER | 355 QA pairs | Long-form NLP paper QA |
| HotpotQA | 500 samples (seed=42) | Multi-hop reasoning |
| PubMedQA | 500 samples (seed=42) | Biomedical long-answer QA |

---

## Evaluation Metrics (RAGAS)

| Metric | Measures |
|---|---|
| `faithfulness` | Answer grounded in retrieved context? |
| `answer_relevancy` | Answer actually addresses the question? |
| `context_precision` | Retrieved chunks relevant to question? |
| `context_recall` | Ground truth covered by retrieved context? |

---

## Output Structure

```
data/<dataset>/results/
  baseline_raw.jsonl       — raw inference (question, answer, contexts, ground_truth)
  baseline_metrics.csv     — RAGAS scores per question
  graphnomd_raw.jsonl
  graphnomd_metrics.csv
  graphmd_raw.jsonl
  graphmd_metrics.csv
```

Raw JSONL saved before RAGAS — allows re-running evaluation without re-inference.

---

## Key Implementation Details

- `keep_alive=0` on all single calls → VRAM freed immediately after each model use
- `keep_alive=300` during batch embedding loops → model stays loaded across batches, unloaded once at end
- ChromaDB `upsert` (not `add`) → safe to re-run after mid-run crashes
- Graph extraction truncated to 8000 chars per doc to fit Qwen context window
- Embedding truncated to 15000 chars per chunk to avoid BGE-M3 500 errors

---

## Conference Target

**APWeb-WAIM 2026**, Da Nang, Vietnam.
