Paper Title
MD-GraphRAG: Synergizing Markdown-Aware Chunking and Knowledge Graph Enrichment for Retrieval-Augmented Generation
Track Name
Research Papers - 2nd Round
Reviewer #1
Questions
1. Paper Significance
Above Average
2. Paper Clarity
Above Average
3. Summary of the paper
This paper studies document chunking and relational knowledge enhancement for Retrieval-Augmented Generation (RAG). The proposed method, MD-GraphRAG, combines Markdown-aware semantic chunking with lightweight knowledge graph enrichment based on LLM-extracted subject-relation-object triplets. It further introduces a two-channel retrieval mechanism, where semantic chunks are retrieved first and graph edges are then retrieved only within the scope of the relevant parent documents. The authors evaluate the method on QASPER, MultiFieldQA, and NarrativeQA, and report improvements over NaiveRAG and a graph-enhanced RAG variant, especially in context precision and context recall.
4. Strong points of the paper (please number S1, S2, S3 ...)
S1: The paper addresses a relevant and practical problem in RAG systems, namely context fragmentation caused by fixed-size chunking.
S2: The proposed design is simple and reasonable. Using Markdown structure for semantic chunking can better preserve document organization.
S3: The two-channel retrieval mechanism is a useful idea for reducing noisy graph-edge retrieval by restricting graph search to relevant documents.
S4: The experiments cover multiple datasets with different document types, and the reported results suggest that the method can improve retrieval quality in long-document QA tasks.
5. Weak points of the paper (please number W1, W2, W3 ...)
W1: The main weakness is the missing Markdown-only ablation. Without this baseline, it is difficult to determine whether the improvements come from Markdown-aware chunking, graph enrichment, or their combination.
W2: The novelty is somewhat incremental. Markdown-based chunking, LLM-extracted triplets, and graph-enhanced retrieval have all been explored in prior work, so the paper should better clarify its unique contribution.
W3: The GraphRAG baseline may be confusing. The paper’s GraphRAG variant mainly embeds extracted triplets as text, which is different from more complete GraphRAG methods involving global graph structures or graph-based reasoning.
W4: The method may depend heavily on the quality of Markdown formatting. If the Markdown headings are noisy, missing, or inconsistent, it is unclear whether the proposed chunking strategy would still be effective.
W5: The baseline comparison could be stronger. Besides NaiveRAG and the graph-enhanced variant, the paper should compare with Markdown-only RAG, Parent Document Retrieval, hybrid retrieval, or other structure-aware retrieval methods.
6. Detailed comments (please number D1, D2, D3 ...)
NA
7. Overall Rating
Weak Accept
Reviewer #2
Questions
1. Paper Significance
Below Average
2. Paper Clarity
Below Average
3. Summary of the paper
This work conducts an experimental study on the chunking and KG-based RAG by introducing markdown to help distinguishing the information's hierarchical structures.
4. Strong points of the paper (please number S1, S2, S3 ...)
1. Some of the existing solutions are surveyed
2. Markdown's effectiveness is tested and validated
5. Weak points of the paper (please number W1, W2, W3 ...)
1. There is no technical contribution or new findings
2. The graphRAG pipelines could be improved and the overall survey is not sufficient. The current experiment only covers simple ones
3. The plots are very big with limited information
6. Detailed comments (please number D1, D2, D3 ...)
1. The plots are very space-wasting and the words are very small. It seems to be converted from markdown? The ## marks are still in the textboxes...
2. Figure 1 is essentially repeating the text description. It provides no new information and this visualization is meaningless.
3. This manuscript is just a simple experiment report with no new technical contribution.
4. The discussion in Section 2.1 provides no new information compared with the introduction
5. The discussion in 2.2 and 2.3 are also not clear and sufficient. It contains very few references.
6. The graphRAGs are constructed in a very simple way. In fact, the quality and effectiveness of a graphRAG heavily relies on the quality of the extracted knowledge graph. The current one "extract up to 20 salient relational triplets" using QWen and converted into string seems to be a naive pipeline.
7. The experiment findings are straightforward and not something new.
7. Overall Rating
Reject
Reviewer #3
Questions
1. Paper Significance
Above Average
2. Paper Clarity
Above Average
3. Summary of the paper
This paper described a solution to improving Retrieval-Augmented Generation (RAG) systems. The paper effectively identifies two key weaknesses in standard pipelines: context fragmentation due to fixed-size chunking and the inability of dense retrieval to capture relational knowledge. The proposed MD-GraphRAG framework addresses these issues by combining structure-aware chunking with lightweight knowledge graph enrichment. The authors describe a two-channel retrieval mechanism, which first retrieves relevant semantic sections and then restricts graph-based retrieval to those contexts. This design successfully balances recall and precision, overcoming the noise typically introduced by knowledge graph methods. The experimental evaluation is thorough and demonstrates consistent improvements across multiple datasets. However, the approach depends on reliable Markdown structure and LLM-based graph extraction, which may introduce errors. Overall, the paper offers a compelling, efficient enhancement to existing RAG architectures.
4. Strong points of the paper (please number S1, S2, S3 ...)
1. Introduces MD-GraphRAG combining semantic chunking with knowledge graph enrichment in a unified RAG framework
2. Proposes Markdown-aware chunking to preserve document structure and prevent context fragmentation
3. Presents a novel two-channel retrieval mechanism to balance semantic context and relational information
4. Demonstrates synergy between chunking and graph enrichment through a controlled ablation study
5. Achieves consistent improvements in faithfulness, context precision, and recall across multiple datasets
6. Reduces noise from graph retrieval via scoped filtering based on relevant document contexts
7. Provides thorough evaluation using diverse datasets and established RAGAS metrics
5. Weak points of the paper (please number W1, W2, W3 ...)
Comment 1: Relies on accurate Markdown structure; performance degrades with poorly formatted or unstructured documents. The authors can consider introducing hybrid chunking that falls back to sentence- or paragraph-level splitting when headings are missing or unreliable.

Comment 2: LLM-based graph extraction introduces hallucinated or incorrect relational triplets. To handle this, the authors can consider adding verification layers using consistency checks, retrieval-based validation, or ensemble extraction with multiple models.

Comment 3: Graph extraction limited to first 8,000 characters, potentially missing important late-document information. To handle this, the authors can consider applying a sliding-window extraction or hierarchical summarization to cover full documents without exceeding context limits.

Comment 4: Two-channel retrieval may fail when query terms appear only in graph edges, not semantic chunks. To handle this, the authors can consider enabling dual-entry retrieval where graph search can also guide initial document selection independently of semantic chunks.

Comment 5: Strict scoping can block relevant graph information if initial semantic retrieval is incomplete. The authors can consider introducing soft filtering (weighted scoring) instead of strict filtering to allow partially relevant graph edges.

Comment 6: Performance depends on embedding model quality and semantic similarity accuracy. The authors can consider using hybrid retrieval (dense + sparse methods like BM25) or fine-tune embeddings on domain-specific corpora.

Comment 7: Lacks robustness to noisy or heterogeneous real-world datasets with inconsistent structure. The authors can consider incorporating preprocessing pipelines for structure normalization and adaptive chunking based on document characteristics.

Comment 8: Evaluation relies solely on automated RAGAS metrics without human qualitative validation. The authors can consider adding human annotation studies to validate answer quality, grounding, and usefulness beyond automated metrics.

Comment 9: Uses relatively small models, possibly limiting generalization to stronger LLM settings. The authors can evaluate with larger or more advanced LLMs to confirm scalability and robustness of improvements.

Comment 10: Does not fully explore scalability or performance on very large corpora. Here, there’s a need to benchmark performance on large-scale corpora and optimize indexing/retrieval efficiency using distributed vector databases.

6. Detailed comments (please number D1, D2, D3 ...)
Please see above
7. Overall Rating
Weak Accept