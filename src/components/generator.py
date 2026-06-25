import os
import sys
import chromadb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.ollama_manager import OllamaManager

class Generator:
    """
    RAG Generator for Graph with markdown and Graph no markdown pipelines.
    All model calls use keep_alive=0 to avoid two models loaded simultaneously (RTX 5060 Ti 16GB).

    Two-channel retrieval (Graph with markdown):
      Channel 1: semantic_section chunks (primary evidence)
      Channel 2: graph_edge chunks scoped to papers from Channel 1 (relational facts)
    Scoping prevents off-topic graph edges from diluting context_precision.

    Fallback (Graph no markdown): collection has baseline_chunk not semantic_section,
    so Channel 1 filter returns empty → falls back to unfiltered retrieval.

    Markdown-only ablation (use_graph=False): reuses the Graph-with-markdown index but
    disables Channel 2, so retrieval returns only semantic_section chunks (the full top_k
    budget). This isolates the contribution of the knowledge graph: comparing this against
    the full Graph-with-markdown pipeline measures the graph's effect with chunking held
    fixed, completing the chunking x graph ablation grid (see docs/PAPER_REVISIONS.md, W1).
    """
    DEFAULT_SYSTEM_PROMPT = (
        "You are a precise academic research assistant. "
        "Answer questions strictly based on the provided text evidence and relational facts. "
        "Do not add information beyond what is given. Answer in English."
    )

    def __init__(self, ollama_manager: OllamaManager, db_dir: str, embed_model: str = "bge-m3", llm_model: str = "llama3.1:8b", system_prompt: str = None, use_graph: bool = True):
        self.ollama = ollama_manager
        self.db_dir = db_dir
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.use_graph = use_graph
        self._system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT

        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        try:
            self.collection = self.chroma_client.get_collection(name="qasper_graph_rag")
        except Exception:
            self.collection = None

    def query(self, question: str, top_k: int = 10) -> dict:
        """Execute QA, return Dict with answer and context list."""
        if not self.collection:
            return {"answer": "No vector data indexed in ChromaDB.", "context": []}

        print(f"\n[Generation] Processing question: '{question}'")

        # 1. Compute query vector, free VRAM immediately after
        query_emb = self.ollama.get_embeddings(
            model=self.embed_model,
            prompt=question,
            keep_alive=0  # CRITICAL
        )

        # 2a. Channel 1 — retrieve semantic section chunks (primary text evidence)
        # Markdown-only (use_graph=False) spends the full budget on sections; the graph
        # pipeline reserves ~3 slots for graph edges retrieved in Channel 2.
        sem_k = top_k if not self.use_graph else max(top_k - 3, 5)
        sem_contexts = []
        retrieved_paper_ids = []
        try:
            sem_results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=sem_k,
                where={"type": {"$eq": "semantic_section"}},
                include=["documents", "metadatas", "distances"]
            )
            sem_contexts = sem_results.get("documents", [[]])[0]
            sem_metas = sem_results.get("metadatas", [[]])[0]
            retrieved_paper_ids = list({m["paper_id"] for m in sem_metas if "paper_id" in m})
        except Exception as e:
            print(f"[Generation] Filtered search failed ({e}).")

        # Fallback: if collection has no semantic_section chunks (e.g. P3 uses baseline_chunk),
        # drop to unfiltered retrieval so P3 still gets results via this same class.
        if not sem_contexts:
            fallback = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
            all_contexts = fallback.get("documents", [[]])[0]
            return self._generate_answer(question, all_contexts, all_contexts, [])

        # 2b. Channel 2 — graph edges scoped to papers already retrieved in Channel 1
        # Scoping to retrieved_paper_ids prevents off-topic relational facts from polluting context.
        # Skipped entirely in the Markdown-only ablation (use_graph=False).
        graph_contexts = []
        if self.use_graph and retrieved_paper_ids:
            try:
                graph_k = min(5, top_k // 2)
                graph_results = self.collection.query(
                    query_embeddings=[query_emb],
                    n_results=graph_k,
                    where={
                        "$and": [
                            {"type": {"$eq": "graph_edge"}},
                            {"paper_id": {"$in": retrieved_paper_ids}}
                        ]
                    },
                    include=["documents"]
                )
                graph_contexts = graph_results.get("documents", [[]])[0]
            except Exception as e:
                print(f"[Generation] Graph channel failed ({e}), skipping graph context.")

        all_contexts = sem_contexts + graph_contexts
        return self._generate_answer(question, all_contexts, sem_contexts, graph_contexts)

    def _generate_answer(self, question: str, all_contexts: list, sem_contexts: list, graph_contexts: list) -> dict:
        """Build structured prompt and call LLM."""
        # Truncate chunks to avoid overflowing Llama context window
        sem_str = "\n".join([f"  - {ctx[:800]}" for ctx in sem_contexts]) or "  (none)"
        graph_str = "\n".join([f"  - {ctx}" for ctx in graph_contexts]) or "  (none)"

        prompt = (
            f"Text evidence from paper sections:\n{sem_str}\n\n"
            f"Relational facts from knowledge graph:\n{graph_str}\n\n"
            f"Question: {question}\n\n"
            f"Using only the evidence above, provide a concise factual answer. "
            f"If the evidence does not contain the answer, say "
            f"\"The context does not contain sufficient information.\"\n\n"
            f"Answer:"
        )
        print(f"[Generation] Generating response with model {self.llm_model}...")
        answer = self.ollama.generate(
            model=self.llm_model,
            prompt=prompt,
            system=self._system_prompt,
            keep_alive=0  # CRITICAL
        )

        return {
            "answer": answer.strip(),
            "context": all_contexts
        }


if __name__ == "__main__":
    _root = os.path.join(os.path.dirname(__file__), '..', '..')
    manager = OllamaManager()
    generator = Generator(
        manager,
        db_dir=os.path.join(_root, "data", "embeddings"),
        embed_model="bge-m3",
        llm_model="llama3.1:8b"
    )
    # response = generator.query("What datasets did they experiment with?", top_k=3)
    # print(response["answer"])
