# NetworkX Knowledge Graph Integration — Tasks

All tasks completed and verified. ✅

- [x] **1. New Component**: `src/components/knowledge_graph.py`
  - [x] KnowledgeGraph class with NetworkX DiGraph
  - [x] Build, persist (.graphml), load operations
  - [x] Multi-hop traversal, subgraph extraction, shortest path
  - [x] Entity normalization & fuzzy matching
  - [x] Subgraph-to-text serialization

- [x] **2. Modify**: `src/components/graph_builder.py`
  - [x] Build NetworkX graph alongside JSON output
  - [x] Persist .graphml files

- [x] **3. Modify**: `src/components/embedder.py`
  - [x] Embed node-centric context instead of raw triplet strings
  - [x] Use KnowledgeGraph.get_entity_context()

- [x] **4. Modify**: `src/components/generator.py`
  - [x] Add graph_dir parameter, load KnowledgeGraph
  - [x] Entity extraction from query
  - [x] Subgraph traversal for Channel 2
  - [x] Hybrid prompt with structural + vector-retrieved context

- [x] **5. Modify**: `src/ablation/graph_no_markdown_embedder.py`
  - [x] Same graph embedding changes as embedder.py

- [x] **6. Modify**: `src/pipeline.py`
  - [x] Pass graph_dir to Generator

- [x] **7. Modify Benchmarks** (all 7):
  - [x] qasper_benchmark.py
  - [x] hotpot_benchmark.py
  - [x] pubmedqa_benchmark.py
  - [x] narrativeqa_benchmark.py
  - [x] natural_questions_benchmark.py
  - [x] quality_benchmark.py
  - [x] sciq_benchmark.py

- [x] **8. Modify**: `test/test_mini.py`
  - [x] Add KnowledgeGraph unit test
  - [x] Update generator tests with graph_dir

- [x] **9. Verify**: Run tests
  - [x] KnowledgeGraph unit test — ALL PASSED
  - [x] All component imports — OK
  - [x] All benchmark syntax — OK

- [x] **10. Clean old data**
  - [x] Deleted old ChromaDB embeddings (~2.2 GB) — incompatible with new `graph_context` type
  - [x] Deleted old results (generated with old triplet-text embeddings)
  - [x] Kept: `parsed/`, `parsed_txt/`, `graph/` (JSON files still valid)

- [x] **11. Update documentation**
  - [x] README.md — updated architecture, data flow, repository layout
  - [x] CLAUDE.md — updated key modules, data paths, chunk metadata types
  - [x] docs/task.md — marked all tasks complete
