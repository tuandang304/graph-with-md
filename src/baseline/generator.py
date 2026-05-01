import os
import sys
import chromadb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.ollama_manager import OllamaManager

class BaselineGenerator:
    """
    RAG Generator for Baseline pipeline (fixed-size chunks, no graph).
    """
    def __init__(self, ollama_manager: OllamaManager, db_dir: str, embed_model: str = "bge-m3", llm_model: str = "llama3.1:8b"):
        self.ollama = ollama_manager
        self.db_dir = db_dir
        self.embed_model = embed_model
        self.llm_model = llm_model

        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        try:
            self.collection = self.chroma_client.get_collection(name="baseline_rag")
        except Exception:
            self.collection = None

    def query(self, question: str, top_k: int = 5) -> dict:
        if not self.collection:
            return {"answer": "No baseline data available.", "context": []}

        print(f"\n[Baseline Gen] '{question}'")
        # Embed question with bge-m3, then free VRAM
        query_emb = self.ollama.get_embeddings(model=self.embed_model, prompt=question, keep_alive=0)

        # Search Top-K
        results = self.collection.query(query_embeddings=[query_emb], n_results=top_k)
        contexts = results.get("documents", [[]])[0]
        context_str = "\n".join([f"- {ctx}" for ctx in contexts])

        # Call Generator, then free VRAM
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
        system_prompt = "You are an AI assistant. Answer based on the provided context. If unavailable, say Unknown."

        answer = self.ollama.generate(model=self.llm_model, prompt=prompt, system=system_prompt, keep_alive=0)

        return {"answer": answer.strip(), "context": contexts}
