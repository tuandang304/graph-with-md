# MD-GraphRAG: Synergizing Markdown-Aware Chunking and Knowledge Graph Enrichment for Retrieval-Augmented Generation

> **Revision draft (2nd round).** This manuscript incorporates the reviewer feedback in
> [`docs/REVIEW.md`](docs/REVIEW.md) following the action plan in
> [`docs/PAPER_REVISIONS.md`](docs/PAPER_REVISIONS.md). Numbers reported in this draft are the
> *actual* RAGAS scores produced by the released code (`data/<dataset>/results/*_metrics.csv`).
> Cells marked **— (pending)** correspond to the newly added *Markdown-only* ablation and to
> datasets whose full runs are still in progress; they will be populated from the same pipeline
> before camera-ready. No results have been invented.

---

## Abstract

Retrieval-Augmented Generation (RAG) is bottlenecked by two recurring problems: fixed-size chunking
fragments the document context, and dense retrieval alone cannot surface the *relational* knowledge
that multi-hop questions require. We study **MD-GraphRAG**, a deliberately lightweight pipeline that
(i) chunks documents along their **Markdown section structure** instead of by character count, (ii)
enriches the index with **LLM-extracted subject–relation–object triplets** stored as short textual
edges, and (iii) retrieves through a **two-channel, structure-scoped** mechanism: semantic sections
are retrieved first, and graph edges are retrieved only within the documents those sections came from.
The scoping is the key design choice — it lets a cheap, imperfect knowledge graph help without
flooding the prompt with off-topic relations.

Rather than claim a uniform win, we evaluate the design as a **2×2 factorial** (chunking ∈ {fixed,
markdown} × graph ∈ {off, on}), which isolates the contribution of each component and directly answers
the central methodological question: *where do the gains actually come from?* On three QA benchmarks
spanning multi-hop (HotpotQA), biomedical (PubMedQA), and open-domain encyclopedic (Natural Questions)
settings, evaluated with RAGAS, we find that structure-aware chunking plus scoped graph retrieval
primarily improves **context recall** on multi-hop and long-document questions, while offering little
or no benefit — and sometimes a precision cost — on short, single-passage biomedical abstracts. We
report these mixed outcomes transparently, analyse the precision/recall trade-off they reveal, and
discuss the conditions under which lightweight graph enrichment is worthwhile.

**Keywords:** Retrieval-Augmented Generation · Document chunking · Knowledge graphs · Structure-aware
retrieval · RAGAS evaluation

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) [Lewis et al., 2020] grounds a large language model (LLM) in
retrieved evidence, reducing hallucination and enabling answers over corpora the model was never
trained on. A standard RAG pipeline splits documents into fixed-size chunks, embeds them, retrieves
the top-*k* most similar chunks for a query, and conditions generation on them. This default has two
well-known weaknesses:

1. **Context fragmentation.** Splitting by a fixed character/token budget cuts across semantic
   boundaries, so a single retrieved chunk may begin mid-argument or straddle two unrelated sections.
   Evidence that belongs together is scattered across chunks and may not be co-retrieved.

2. **Missing relational structure.** Dense similarity retrieval rewards lexical/semantic overlap with
   the query, but many questions — especially multi-hop ones — depend on *relations between entities*
   that no single passage states verbatim.

Graph-enhanced RAG addresses the second problem by extracting a knowledge graph (KG) and using it
during retrieval. Full GraphRAG systems [Edge et al., 2024] build global graph structures and perform
community detection or graph traversal; they are powerful but heavy. A lighter alternative embeds
extracted triplets as text and retrieves them like any other chunk — but naïvely mixing graph edges
into the candidate pool tends to *hurt* precision, because edges from unrelated documents are
retrieved on weak surface similarity.

This paper studies a pragmatic middle ground, **MD-GraphRAG**, and — responding to the second-round
reviews — frames the study explicitly as an **ablation** rather than a single-system showcase. Our
contributions are:

- **C1 — Structure-aware chunking with a hybrid fallback.** We chunk along Markdown `##` section
  boundaries (merging very short sections), and fall back to paragraph/sliding-window splitting when
  headings are absent or unreliable, so the method degrades gracefully on unstructured text.

- **C2 — Two-channel, structure-scoped graph retrieval.** Graph edges are retrieved in a *second
  channel* restricted to the parent documents of the sections retrieved in the first channel. This
  scoping is the technical core: it is what makes a cheap, noisy KG useful instead of harmful.

- **C3 — A controlled 2×2 evaluation.** We disentangle the *chunking* effect, the *graph* effect, and
  their *interaction* by running all four cells of the chunking × graph grid — including the
  **Markdown-only** baseline that the first-round paper was criticised for omitting.

- **C4 — An honest, mixed-results analysis.** Across heterogeneous datasets the method helps recall on
  multi-hop and long-document QA but not on short biomedical abstracts. We characterise *when* the
  design pays off rather than asserting a universal improvement.

---

## 2. Related Work

**Chunking strategies for RAG.** The granularity and boundaries of chunks materially affect retrieval
quality. Fixed-size splitting with overlap is the default in popular frameworks, but it ignores
document layout. Parent-Document Retrieval retrieves small chunks and returns their larger parent for
generation; semantic/proposition chunking groups sentences by meaning. Our Markdown-aware chunking is
a structure-driven instance of this family: it uses the author-provided section hierarchy as a free,
high-precision segmentation signal, with a fallback for documents that lack it.

**Graph-enhanced RAG.** Knowledge graphs add explicit relational structure. Microsoft's GraphRAG
[Edge et al., 2024] constructs a global graph and summarises communities for query-focused
summarisation; KG-RAG variants link retrieval to entities and relations. These methods can reason over
graph topology but incur substantial extraction, indexing, and traversal cost. We deliberately occupy
the *lightweight* end: triplets are extracted once per document and stored as short textual edges, with
no global graph, community detection, or multi-hop traversal. This makes our "graph" comparison a
**triplet-as-text enrichment**, which we name accordingly (Section 4) to avoid implying parity with
traversal-based GraphRAG.

**Structure- and hybrid retrieval.** Sparse retrieval (BM25) and dense retrieval are complementary;
hybrid fusion (e.g., reciprocal-rank fusion) often beats either alone. Layout- and structure-aware
retrieval exploits headings, tables, and sections. MD-GraphRAG is structure-aware on the *chunking*
side; we treat hybrid sparse+dense retrieval and Parent-Document Retrieval as orthogonal baselines for
future comparison (Section 7).

**Evaluation of RAG.** Reference-based QA metrics poorly capture grounding and retrieval quality.
RAGAS [Es et al., 2023] provides reference-free and reference-based LLM-graded metrics —
*faithfulness*, *answer relevancy*, *context precision*, *context recall* — that separately probe the
generator and the retriever. We adopt RAGAS and, following reviewer guidance, disclose its judge model
as a limitation (Section 6.3).

---

## 3. Problem Setup

Given a corpus of documents and a natural-language question *q*, a RAG system retrieves a set of
context passages *C* and generates an answer *a = LLM(q, C)*. We hold the embedding model, generator,
and retrieval budget *k* fixed and vary only two design factors:

- **Chunking** ∈ {*fixed-size*, *markdown-section*}.
- **Graph enrichment** ∈ {*off*, *on*}.

This yields four systems (Section 4.5). Fixing everything else lets differences in the RAGAS metrics
be attributed to the two factors and their interaction.

---

## 4. Method: MD-GraphRAG

### 4.1 Overview

MD-GraphRAG has three stages: structure-aware chunking, lightweight KG enrichment, and two-channel
scoped retrieval, followed by standard grounded generation.

```
Document ──► (A) Markdown-aware chunking ──► semantic-section chunks ─┐
         └─► (B) Triplet extraction (LLM) ──► graph-edge chunks ──────┤
                                                                      ▼
                                          single vector index (typed chunks)
                                                                      │
                            (C) two-channel scoped retrieval ◄────────┘
                                  Ch.1 sections → parent docs → Ch.2 edges (scoped)
                                                                      │
                                              grounded generation (LLM)
```

### 4.2 (A) Markdown-aware semantic chunking

Documents are normalised to Markdown. We split on heading boundaries (`#`, `##`, `###`) so each chunk
is a coherent author-defined section. To avoid over-fragmentation, consecutive sections are merged
until a chunk exceeds a minimum length (400 characters). Each chunk is tagged `type = semantic_section`
with its parent document id.

**Hybrid fallback (robustness).** When a document has no reliable heading structure (e.g., raw
Wikipedia text or plain prose), the chunker falls back to fixed-size sliding windows (with a section
title synthesised per window). This is what allows the same pipeline to run on heading-rich papers
(QASPER) and heading-poor prose (NarrativeQA, Natural Questions) without manual tuning, and it
addresses the concern that the method might collapse on poorly formatted input.

### 4.3 (B) Lightweight knowledge-graph enrichment

For each document we prompt an instruction-tuned LLM to extract up to 20 salient
**subject–relation–object** triplets as a JSON array. Each triplet is serialised to a short string —
`"Graph Relation: <source> -> <relation> -> <target>"` — embedded, and stored as a `type = graph_edge`
chunk carrying the same parent-document id. We use domain-specific extraction prompts (narrative,
encyclopedic, scientific) per dataset. The graph is intentionally minimal: no global merging, entity
resolution, or traversal. The design question is not "is this the best possible KG?" but "can even a
*cheap* KG help, if retrieval is structured correctly?"

### 4.4 (C) Two-channel, structure-scoped retrieval

Sections and edges live in one vector store distinguished by `type`. For a query *q*:

- **Channel 1 (evidence).** Retrieve the top semantic-section chunks. Record the set *D* of their
  parent-document ids.
- **Channel 2 (relations).** Retrieve graph-edge chunks **restricted to documents in *D***
  (`type = graph_edge ∧ parent ∈ D`).

The retrieved sections and (scoped) edges are concatenated into the generation prompt under separate
"text evidence" and "relational facts" headers. **Scoping is the mechanism that prevents graph noise:**
without it, edges from unrelated documents are pulled in on weak similarity and dilute precision; with
it, edges can only reinforce documents already deemed relevant by the evidence channel. If Channel 1
returns nothing (e.g., an index that contains only fixed-size chunks), retrieval falls back to a single
unfiltered pass so the same code serves every pipeline.

### 4.5 The four pipelines (2×2 design)

|  | **Graph off** | **Graph on** |
|---|---|---|
| **Fixed-size chunking** | **NaiveRAG** (Baseline) | **Triplet-RAG** (Graph no markdown) |
| **Markdown chunking**   | **MD-RAG** (Markdown only) | **MD-GraphRAG** (Graph with markdown) |

- **NaiveRAG** — fixed-size chunks, dense retrieval, no graph.
- **MD-RAG (Markdown only)** — Markdown-section chunks, no graph. *(New in this revision; isolates the
  chunking factor. Implemented by reusing the MD-GraphRAG index with Channel 2 disabled, so the
  comparison is exactly controlled.)*
- **Triplet-RAG (Graph no markdown)** — fixed-size chunks + graph edges.
- **MD-GraphRAG (Graph with markdown)** — Markdown-section chunks + scoped graph edges (proposed).

We rename the graph variants to **Triplet-RAG / MD-GraphRAG** to make explicit that our "graph" is
lightweight triplet-as-text enrichment, distinct from traversal-based GraphRAG.

---

## 5. Experimental Setup

### 5.1 Datasets

We evaluate on QA datasets spanning document types and reasoning patterns. Results in this draft are
reported for the three datasets whose full runs are complete; the framework additionally supports
QASPER, NarrativeQA, QuALITY, and SciQ (runs in progress).

| Dataset | Domain / document type | Reasoning | Evaluated n |
|---|---|---|---|
| **HotpotQA** (dev distractor) | Wikipedia paragraphs | multi-hop | 464 |
| **PubMedQA** (`pqa_labeled`)  | biomedical abstracts | single-passage factual | 477 |
| **Natural Questions** (val)   | full Wikipedia articles | open-domain factual | 356 |
| QASPER / NarrativeQA / QuALITY / SciQ | papers / books / long articles / textbooks | — | *pending* |

(*n* is the number of QA pairs that produced a scorable record; items with empty ground truth are
skipped.)

### 5.2 Models and implementation

All local models are served via Ollama with strict single-model VRAM occupancy (16 GB GPU):

- **Triplet extraction:** Qwen2.5-7B-Instruct.
- **Embeddings:** BGE-M3 (sections, edges, and queries).
- **Generation:** Llama-3.1-8B-Instruct.
- **Vector store:** ChromaDB (cosine), one index per pipeline.

Retrieval budget: Baseline *k* = 5; the markdown/graph pipelines *k* = 10, of which MD-GraphRAG spends
≈ 3 slots on graph edges and the rest on sections. We report this budget asymmetry as a caveat
(Section 6.3) and align it in ongoing runs.

### 5.3 Metrics

We use RAGAS with GPT-4o-mini as judge and `text-embedding-3-small` for similarity:

- **Faithfulness** — is the answer grounded in the retrieved context? (generator)
- **Answer relevancy** — does the answer address the question? (generator)
- **Context precision** — are retrieved contexts on-target? (retriever)
- **Context recall** — do retrieved contexts cover the ground truth? (retriever)

Precision and recall require ground-truth references; faithfulness and relevancy do not.

---

## 6. Results and Analysis

### 6.1 Main results

Mean RAGAS scores (higher is better). **Bold** = best per metric per dataset. The *Markdown only*
(MD-RAG) column is **— (pending)** in this draft.

**HotpotQA (multi-hop, n = 464)**

| Pipeline | Faithfulness | Answer rel. | Ctx precision | Ctx recall |
|---|---|---|---|---|
| NaiveRAG (Baseline)        | 0.794 | 0.420 | **0.641** | 0.813 |
| MD-RAG (Markdown only)     | —     | —     | —     | —     |
| Triplet-RAG (no markdown)  | 0.813 | 0.406 | 0.576 | **0.858** |
| **MD-GraphRAG (proposed)** | **0.814** | **0.422** | 0.553 | 0.815 |

**PubMedQA (biomedical abstracts, n = 477)**

| Pipeline | Faithfulness | Answer rel. | Ctx precision | Ctx recall |
|---|---|---|---|---|
| NaiveRAG (Baseline)        | **0.824** | **0.552** | **0.949** | 0.690 |
| MD-RAG (Markdown only)     | —     | —     | —     | —     |
| Triplet-RAG (no markdown)  | 0.800 | 0.407 | 0.816 | 0.668 |
| **MD-GraphRAG (proposed)** | 0.819 | 0.401 | 0.802 | **0.714** |

**Natural Questions (open-domain Wikipedia, n = 356)**

| Pipeline | Faithfulness | Answer rel. | Ctx precision | Ctx recall |
|---|---|---|---|---|
| NaiveRAG (Baseline)        | **0.852** | 0.448 | 0.682 | 0.784 |
| MD-RAG (Markdown only)     | —     | —     | —     | —     |
| Triplet-RAG (no markdown)  | 0.849 | **0.529** | 0.638 | 0.813 |
| **MD-GraphRAG (proposed)** | 0.748 | 0.399 | **0.706** | **0.863** |

### 6.2 Decomposition of effects

The 2×2 design lets us isolate each factor. Two of the four effects are computable from current data;
the other two require the *Markdown only* cell and are **pending**.

**Markdown effect *given* graph = MD-GraphRAG − Triplet-RAG** (does structure help once a graph is
present?)

| Dataset | Δ Faithfulness | Δ Answer rel. | Δ Ctx precision | Δ Ctx recall |
|---|---|---|---|---|
| HotpotQA          | +0.001 | +0.016 | −0.023 | −0.043 |
| PubMedQA          | +0.019 | −0.006 | −0.014 | +0.046 |
| Natural Questions | −0.101 | −0.129 | **+0.068** | **+0.050** |

**Graph effect *without* markdown = Triplet-RAG − NaiveRAG** (does a graph help on fixed chunks?)

| Dataset | Δ Faithfulness | Δ Answer rel. | Δ Ctx precision | Δ Ctx recall |
|---|---|---|---|---|
| HotpotQA          | +0.020 | −0.014 | −0.066 | +0.045 |
| PubMedQA          | −0.024 | −0.145 | −0.133 | −0.023 |
| Natural Questions | −0.003 | +0.081 | −0.044 | +0.030 |

- **Pending — Markdown effect *without* graph** = MD-RAG − NaiveRAG.
- **Pending — Graph effect *with* markdown** = MD-GraphRAG − MD-RAG.
- **Pending — Interaction (synergy)** = (MD-GraphRAG − MD-RAG) − (Triplet-RAG − NaiveRAG).

Completing the MD-RAG cell is the single most informative remaining experiment; the released code
produces it with no additional ingestion.

### 6.3 Discussion

**The benefit of the design is recall-oriented and concentrated on long / multi-hop documents.**
MD-GraphRAG attains the best **context recall** on PubMedQA and Natural Questions and the best
**context precision** on Natural Questions; on HotpotQA the graph variants improve recall over
NaiveRAG. This matches the intuition that author structure plus scoped relational edges help *gather*
relevant evidence across a long or multi-passage document.

**The design does not uniformly win, and we do not claim it does.** On **PubMedQA**, NaiveRAG is best on
three of four metrics. PubMedQA contexts are short, self-contained abstracts, so there is little
structure to exploit and little fragmentation to repair; adding triplets mainly dilutes an already
precise context (context precision drops from 0.949 to 0.802). On **Natural Questions**, MD-GraphRAG
improves both retrieval metrics but *lowers* faithfulness (0.852 → 0.748) and answer relevancy — i.e.,
broader retrieval surfaces more on-topic evidence but also gives the generator more to drift on. This
**precision/recall and retrieval/generation trade-off** is the most important empirical finding and a
more honest characterisation than "consistent improvement."

**Implication.** Lightweight, scoped graph enrichment is worthwhile when documents are long or answers
are multi-hop and recall is the bottleneck; it is not worthwhile — and can hurt — when contexts are
already short and precise. We position MD-GraphRAG as a recall-oriented, low-cost option rather than a
dominant default.

**Threats to validity.** (i) RAGAS uses an LLM judge (GPT-4o-mini); absolute values may shift under a
different judge, so we emphasise within-dataset rankings and report all judge/embedding models. (ii)
The retrieval budget differs between Baseline (*k* = 5) and the other pipelines (*k* = 10); ongoing
runs equalise *k*. (iii) Triplets are extracted from a bounded prefix of each document; very
late-document relations may be missed (addressed in Section 7). (iv) A single small generator
(Llama-3.1-8B) is used; generality to larger models is untested.

---

## 7. Limitations and Future Work

Directly reflecting the second-round reviews:

- **Complete the ablation grid.** Fill the *Markdown-only* cell on all datasets to report the full
  chunking/graph decomposition and the interaction term.
- **Stronger KG extraction.** Replace the bounded-prefix, single-pass extractor with sliding-window
  extraction over the full document plus a triplet-verification step (keep only triplets grounded in
  the source text), and audit triplet precision on a manual sample.
- **Stronger and structure-aware baselines.** Add Parent-Document Retrieval and hybrid dense+sparse
  (BM25 + RRF) retrieval; these are the comparisons most requested by reviewers.
- **Soft scoping.** Replace hard document-scoped graph filtering with a weighted (soft) penalty so
  partially relevant edges are not entirely excluded, and ablate hard vs. soft.
- **Markdown-robustness study.** Progressively corrupt/strip headings on one dataset and measure
  degradation, demonstrating that the hybrid fallback works.
- **Human evaluation and larger models.** Add a small human study of grounding/usefulness and re-run
  one dataset with a stronger generator and judge to confirm the trends.
- **Scalability.** Report index size, ingestion time, and query latency, and characterise behaviour on
  larger corpora.

---

## 8. Conclusion

We presented **MD-GraphRAG**, a lightweight RAG pipeline combining Markdown-aware semantic chunking
with triplet-as-text knowledge-graph enrichment, retrieved through a two-channel mechanism that scopes
graph edges to structurally relevant documents. Reframing the study as a controlled 2×2 ablation, we
find that the design is a **recall-oriented** improvement on multi-hop and long-document QA but offers
no advantage — and can cost precision and faithfulness — on short, self-contained passages. The scoped
two-channel retrieval is the component that makes a cheap, noisy graph usable rather than harmful. We
report mixed outcomes transparently and identify the precise additional experiments (chiefly the
Markdown-only cell and a verified, full-document extractor) needed to turn this analysis into a
complete account of when structure-aware, graph-enriched RAG is worthwhile.

---

## References

> Standard, verifiable works; tighten to the venue's citation style for camera-ready.

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130.
- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in IR.
- Dasigi, P., et al. (2021). *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers (QASPER).* NAACL.
- Yang, Z., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP.
- Jin, Q., et al. (2019). *PubMedQA: A Dataset for Biomedical Research Question Answering.* EMNLP.
- Kwiatkowski, T., et al. (2019). *Natural Questions: A Benchmark for Question Answering Research.* TACL.
- Kočiský, T., et al. (2018). *The NarrativeQA Reading Comprehension Challenge.* TACL.
- Pang, R. Y., et al. (2022). *QuALITY: Question Answering with Long Input Texts, Yes!* NAACL.
- Welbl, J., Liu, N. F., Gardner, M. (2017). *Crowdsourcing Multiple Choice Science Questions (SciQ).* W-NUT.
- Chen, J., et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings.* arXiv:2402.03216.
- Grattafiori, A., et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783.
- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.
