# NetworkX Knowledge Graph Integration — Tasks

- [x] **1. New Component**: `src/components/knowledge_graph.py`
  - [x] KnowledgeGraph class with NetworkX DiGraph
  - [x] Build, persist (.graphml), load operations
  - [x] Multi-hop traversal, subgraph extraction, shortest path
  - [x] Entity normalization & fuzzy matching
  - [x] Subgraph-to-text serialization

- [/] **2. Modify**: `src/components/graph_builder.py`
  - [ ] Build NetworkX graph alongside JSON output
  - [ ] Persist .graphml files

- [ ] **3. Modify**: `src/components/embedder.py`
  - [ ] Embed node-centric context instead of raw triplet strings
  - [ ] Use KnowledgeGraph.get_entity_context()

- [ ] **4. Modify**: `src/components/generator.py`
  - [ ] Add graph_dir parameter, load KnowledgeGraph
  - [ ] Entity extraction from query
  - [ ] Subgraph traversal for Channel 2
  - [ ] Hybrid prompt with structural + vector-retrieved context

- [ ] **5. Modify**: `src/ablation/p3_embedder.py`
  - [ ] Same graph embedding changes as embedder.py

- [ ] **6. Modify**: `src/pipeline.py`
  - [ ] Pass graph_dir to Generator

- [ ] **7. Modify Benchmarks** (all 7):
  - [ ] qasper_benchmark.py
  - [ ] hotpot_benchmark.py
  - [ ] pubmedqa_benchmark.py
  - [ ] narrativeqa_benchmark.py
  - [ ] natural_questions_benchmark.py
  - [ ] quality_benchmark.py
  - [ ] sciq_benchmark.py

- [ ] **8. Modify**: `test/test_mini.py`
  - [ ] Add KnowledgeGraph unit test
  - [ ] Update generator tests with graph_dir

- [ ] **9. Verify**: Run tests
