# Paper Revision Plan — MD-GraphRAG (APWeb/WAIM 2026, 2nd Round)

This document turns the reviewer feedback in [`REVIEW.md`](./REVIEW.md) into a concrete action
list for the next revision. Each item is tagged:

- **[CODE — DONE]** implemented in this repository now.
- **[CODE — TODO]** a code change worth doing before the next submission (pointers given).
- **[PAPER]** a writing / figure / framing change with no code impact.

The three reviewers split **Weak Accept / Reject / Weak Accept**. The Reject (R2) and the
strongest Weak-Accept concern (R1-W1) both attack the *experimental design and the GraphRAG
construction*, so those are the highest-leverage fixes.

---

## 0. Headline change already made: the Markdown-only ablation (answers R1-W1)

> R1-W1: "The main weakness is the missing Markdown-only ablation. Without this baseline, it is
> difficult to determine whether the improvements come from Markdown-aware chunking, graph
> enrichment, or their combination."

**[CODE — DONE]** A fourth pipeline, **Markdown only** (semantic `##` chunking, *no* graph), was
added to all seven benchmarks. It reuses the existing Graph-with-markdown ChromaDB index but
disables the graph retrieval channel (`Generator(..., use_graph=False)`), so it is a perfectly
controlled ablation: identical chunks, graph on/off. This completes the **2×2 factorial**:

| Chunking ↓ \ Graph → | **No graph** | **+ Graph** |
|---|---|---|
| **Fixed-size**        | Baseline (NaiveRAG)   | Graph no markdown |
| **Markdown sections** | **Markdown only (NEW)** | Graph with markdown (MD-GraphRAG) |

With four cells the paper can now *decompose* the gain into independent effects:

- **Markdown effect, no graph:** `Markdown only − Baseline`
- **Markdown effect, with graph:** `Graph with markdown − Graph no markdown`
- **Graph effect, with markdown:** `Graph with markdown − Markdown only`
- **Graph effect, no markdown:** `Graph no markdown − Baseline`
- **Synergy / interaction:** `(GraphMD − MDonly) − (GraphNoMD − Baseline)`

The benchmarks now emit `mdonly_raw.jsonl` / `mdonly_metrics.csv` alongside the existing three,
and the console comparison prints both the *Markdown effect* and the *Graph effect* deltas.

**[PAPER]** Replace the current 3-row results tables with the 2×2 grid above, and add a short
"Decomposition of gains" paragraph reporting the deltas. This single change directly answers the
review's #1 weakness and reinforces R3's strong point S4 ("synergy ... through a controlled
ablation study"), which is currently only partially supported.

> ⚠️ Retrieval-budget caveat to state explicitly in the paper: Markdown-only retrieves the full
> `top_k` (=10) as semantic sections, whereas Graph-with-markdown spends ~3 of those 10 slots on
> graph edges. So `GraphMD − MDonly` is a *fair, budget-matched* test of whether graph edges earn
> their slot. The `MDonly − Baseline` comparison is *not* budget-matched (Baseline uses `top_k`=5
> on most datasets). If a perfectly clean chunking-only comparison is wanted, re-run Baseline at
> `top_k`=10 — see item 1.

---

## 1. Retrieval-budget consistency (clean comparisons)

**[CODE — TODO]** `top_k` is currently inconsistent: Baseline uses 5 on most datasets (10 on
QuALITY), graph pipelines use 10. For the camera-ready, run every pipeline at a single shared
`top_k` (recommend 10) so all four cells of the grid are budget-matched, then report `top_k`
sensitivity (5/10/15) in an appendix. The change is a one-line edit per benchmark (the `5` in the
generation loop tuple).

**[PAPER]** State the retrieval budget and chunk-size settings for every pipeline in one config
table. Reviewers cannot currently tell that the budgets differ.

---

## 2. GraphRAG construction quality (answers R2-6, R3-2, R3-3, and R1-W3)

> R2-6: "the quality and effectiveness of a graphRAG heavily relies on the quality of the extracted
> knowledge graph. The current one 'extract up to 20 salient relational triplets' using Qwen and
> converted into string seems to be a naive pipeline."
> R3-3: "Graph extraction limited to first 8,000 characters, potentially missing important
> late-document information."

These are the core of the Reject. Two concrete, implementable improvements:

**[CODE — DONE] NetworkX Knowledge Graph with structural traversal.** `src/components/graph_builder.py`
now builds a real **NetworkX `DiGraph`** alongside the JSON output. The new
`src/components/knowledge_graph.py` supports multi-hop traversal (`get_neighbors(entity, hops=2)`),
subgraph extraction (`get_subgraph_for_entities()`), shortest path, and fuzzy entity matching.
Graph files are persisted as `.graphml`. The Generator now performs **structural graph traversal**
at query time (extract entities from question → match against graph → 2-hop subgraph → serialize
as structured LLM context), replacing the old flat triplet-text embedding approach. This directly
addresses the "naive pipeline" concern by introducing real graph-based reasoning.

**[CODE — TODO] Sliding-window / full-document extraction.** `src/components/graph_builder.py`
currently truncates each document to the first `chunk_char_limit = 8000` chars (one LLM call). Replace
with a sliding window over the whole document (e.g. 8k-char windows, 1k overlap), de-duplicate
triplets across windows, and cap the union. This directly removes the "first 8k chars only"
limitation (R3-3) and improves recall of late-document relations. Keep the per-window cap so VRAM
and latency stay bounded.

**[CODE — TODO] Triplet verification layer (R3-2 hallucinated triplets).** After extraction, add a
cheap grounding check: keep a triplet only if its surface forms (source/target) actually occur in
the source window, optionally with a second-pass LLM "is this supported by the text? yes/no". Report
the rejection rate as evidence of graph quality control.

**[PAPER]** Reframe the contribution: the paper now uses a real NetworkX graph with multi-hop
traversal and subgraph extraction — not just embedded triplet strings. Describe the KnowledgeGraph
architecture (entity normalization, `.graphml` persistence, 2-hop ego-graph retrieval, structured
text serialization) in a subsection. The claim becomes "structure-aware chunking + structural graph
traversal via lightweight KG enables effective multi-hop retrieval." Add a small manual audit of
triplet precision on a sample (e.g. 50 triplets) to evidence graph quality.

---

## 3. Two-channel retrieval robustness (answers R3-4, R3-5)

> R3-4: "Two-channel retrieval may fail when query terms appear only in graph edges, not semantic
> chunks." R3-5: "Strict scoping can block relevant graph information if initial semantic retrieval
> is incomplete."

**[CODE — TODO] Soft scoping instead of hard filtering.** `src/components/generator.py` Channel 2
currently *hard-filters* graph edges to `paper_id ∈ retrieved_paper_ids`. Add a soft-scoping mode:
retrieve graph edges globally but down-weight those outside the Channel-1 document set
(score = similarity − λ·[out-of-scope]) rather than excluding them. Expose `scoping="hard"|"soft"`
and report both.

**[CODE — TODO] Dual-entry retrieval (optional).** Allow graph edges to *also* seed document
selection: if a graph edge scores very high, pull its parent document into the Channel-1 set even if
no section was retrieved. This addresses the "query terms only in graph edges" failure mode.

**[PAPER]** Add an ablation row for hard vs soft scoping. Acknowledge the failure modes explicitly
in a "Limitations" paragraph (reviewers reward this).

---

## 4. Markdown-robustness / hybrid chunking (answers R1-W4, R3-1, R3-7)

> R1-W4 / R3-1: performance may depend on clean Markdown; degrades when headings are missing/noisy.

**[CODE — partially present, make it explicit]** The NarrativeQA and Natural Questions benchmarks
*already* fall back to fixed-size windows when no headings are detected
(`_split_narrative_sections`, `_split_wikipedia_sections` with `FALLBACK_WINDOW`). Promote this into a
named **hybrid chunker** in `src/components/` and reuse it everywhere, so the paper can describe a
single principled "headings when present, paragraph/sentence windows otherwise" strategy.

**[PAPER]** Add a *robustness experiment*: take one dataset, progressively corrupt/strip the Markdown
headings (e.g. drop 25/50/100% of `##`), and plot metric degradation. This is the single most
convincing answer to R1-W4 and shows the fallback works. It is cheap (reuses existing indexes/QA).

---

## 5. Stronger baselines (answers R1-W5, R3-6)

> R1-W5: compare with Markdown-only RAG, Parent Document Retrieval, hybrid retrieval, or other
> structure-aware methods. R3-6: hybrid dense+sparse (BM25).

- **Markdown-only RAG** — **[CODE — DONE]** (item 0).
- **[CODE — TODO] Parent Document Retrieval** — retrieve small chunks, return their parent section.
  Implementable on top of the existing section metadata.
- **[CODE — TODO] Hybrid dense+sparse (BM25)** — add a BM25 channel and fuse with RRF. ChromaDB
  results can be reranked against a BM25 index over the same chunks.

**[PAPER]** Add at least one structure-aware baseline (Parent Document Retrieval is the cheapest)
and one hybrid baseline (BM25+dense). Position MD-GraphRAG against them in the main table.

---

## 6. Novelty / contribution framing (answers R1-W2, R2-1, R2-3, R2-7)

> R1-W2 / R2-1 / R2-3: "novelty is incremental"; "no technical contribution"; "simple experiment
> report."

**[PAPER]** This is a *framing* problem, not (only) a results problem. Recommended reframing:

1. State the contribution as an **empirical finding with a mechanism**, not a new component:
   "Markdown-structure-aware chunking and lightweight KG enrichment are *complementary*, and the
   benefit of a cheap KG is unlocked only when retrieval is *scoped* by structure-derived document
   boundaries." Back it with the 2×2 decomposition and the interaction/synergy term (item 0).
2. Make the **two-channel scoped retrieval** the named technical contribution (R1-S3, R3-S3 already
   credit it) and ablate it properly (item 3).
3. Broaden the evaluation story: the repo already runs **7 datasets** (QASPER, HotpotQA, PubMedQA,
   NarrativeQA, Natural Questions, QuALITY, SciQ), not the 3 in the current draft. Reporting all 7
   directly rebuts "the current experiment only covers simple ones" (R2-2).

---

## 7. GraphRAG-baseline clarity (answers R1-W3, R2-2)

> R1-W3: the paper's "GraphRAG" embeds triplets as text, unlike global-graph GraphRAG methods.

**[CODE — DONE]** This concern is now substantially addressed: the codebase uses a real **NetworkX
`DiGraph`** with multi-hop traversal, subgraph extraction, shortest path, and entity-aware retrieval.
The Generator performs structural graph traversal at query time, not just vector search over embedded
triplet strings. This moves the implementation much closer to "real GraphRAG" territory.

**[PAPER]** Update the paper to describe the NetworkX-based architecture. Suggested labels matching
the code: **NaiveRAG** (Baseline), **KG-RAG (no structure)** (Graph no markdown), **MD-KG-RAG /
MD-GraphRAG** (Graph with markdown), **MD-RAG** (Markdown only). Highlight the structural graph
traversal (multi-hop ego-graph, entity matching, subgraph extraction) as a distinguishing feature
from flat triplet-text approaches, while still distinguishing from Microsoft-style community-based
GraphRAG.

---

## 8. Figures and presentation (answers R2-3-plots, R2-D1, R2-D2)

> R2-D1: "plots are very space-wasting ... words very small ... the `##` marks are still in the
> textboxes ... converted from markdown?" R2-D2: "Figure 1 ... repeating the text ... meaningless."

**[PAPER]**
- Regenerate all plots as vector (PDF/SVG) with readable font sizes; **strip stray `##` markdown
  tokens** from any figure that was exported from a Markdown render. (Embarrassing but trivial; do not
  ship raw `##` in figures.)
- Replace the redundant Figure 1 with something that carries information: the 2×2 ablation grid as a
  schematic, or the two-channel retrieval data-flow annotated with the scoping step.
- Make tables compact; move per-dataset breakdowns to an appendix and keep a single summary table in
  the body.

---

## 9. Evaluation rigor (answers R3-8, R3-9, R3-10)

- **[PAPER] Human evaluation (R3-8):** add a small human study (e.g. 50–100 answers rated for
  correctness/grounding) to complement RAGAS. Note RAGAS has been transitioned to use a local LLM judge (`qwen2.5:7b-instruct-q4_K_M`) and embedding model (`bge-m3`) to run fully locally.
- **[CODE/PAPER] Larger LLM (R3-9):** the pipeline is model-agnostic (`llm_model` arg). Re-run one
  dataset with a stronger generator/judge to show the gains are not an artifact of the default Qwen model.
- **[PAPER] Scalability (R3-10):** report index size, ingestion time, and query latency per pipeline,
  and discuss behavior on larger corpora. Cheap to collect from existing runs.

---

## Priority order for the next revision

1. **2×2 decomposition tables + deltas** (item 0) — already runnable; rewrite results section.
2. **Graph extraction: sliding window + verification** (item 2) — defuses the Reject.
3. **Robustness-to-noisy-Markdown experiment** (item 4) — strongest answer to R1-W4.
4. **Soft scoping ablation** (item 3) and **one structure-aware + one hybrid baseline** (item 5).
5. **Reframing + figures + naming** (items 6–8) — no compute, high reviewer-sentiment payoff.
6. **Human eval + larger-LLM + scalability notes** (item 9).
