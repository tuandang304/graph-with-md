import os
import sys
import json
import chromadb
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.ollama_manager import OllamaManager

class Embedder:
    """
    Component 3: Creates Vector Embeddings via Ollama and stores them locally in ChromaDB.
    Optimized for pipeline with batch embeddings: bge-m3 stays in VRAM for the entire loop,
    then releases to make room for Component 4 (Llama) only after all data is processed.
    """
    def __init__(self, ollama_manager: OllamaManager, input_parsed_dir: str, input_graph_dir: str, db_dir: str, model_name: str = "bge-m3"):
        self.ollama = ollama_manager
        self.parsed_dir = input_parsed_dir
        self.graph_dir = input_graph_dir
        self.db_dir = db_dir
        self.model_name = model_name

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)

        # Init ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="qasper_graph_rag",
            metadata={"hnsw:space": "cosine"}
        )

    def process_all(self):
        print(f"Starting vectorization pipeline with {self.model_name}...")

        md_files = [f for f in os.listdir(self.parsed_dir) if f.endswith('.md')]
        all_chunks = []
        all_metadata = []
        all_ids = []

        for file in md_files:
            paper_id = file.replace('.md', '')
            with open(os.path.join(self.parsed_dir, file), 'r', encoding='utf-8') as f:
                text = f.read()

            # Chunk by Semantic Unit (Section)
            # Split content on ## boundaries to preserve section structure.
            import re
            raw_sections = re.split(r'\n(?=#{1,3}\s)', text)

            # Merge sections that are too small to avoid "diluting" information
            merged_sections = []
            current_chunk = ""
            for s in raw_sections:
                s = s.strip()
                if not s: continue

                if current_chunk:
                    current_chunk += "\n\n" + s
                else:
                    current_chunk = s

                # Save chunk if large enough (> 400 chars) or if it's the last section
                if len(current_chunk) > 400:
                    merged_sections.append(current_chunk)
                    current_chunk = ""

            if current_chunk:  # Save final chunk if any remaining
                merged_sections.append(current_chunk)

            for idx, c in enumerate(merged_sections):
                all_chunks.append(c)
                all_metadata.append({"paper_id": paper_id, "type": "semantic_section"})
                all_ids.append(f"{paper_id}_sec_{idx}")

        # Process graph (nodes/relations)
        graph_files = [f for f in os.listdir(self.graph_dir) if f.endswith('_graph.json')]
        for file in graph_files:
            paper_id = file.replace('_graph.json', '')
            with open(os.path.join(self.graph_dir, file), 'r', encoding='utf-8') as f:
                try:
                    graph_data = json.load(f)

                    # Handle case where Qwen returns JSONL instead of standard JSON Array
                    if isinstance(graph_data, dict) and "raw_output" in graph_data:
                        raw = graph_data["raw_output"]
                        rescued_list = []
                        for line in raw.split('\n'):
                            if line.strip().startswith('{'):
                                try:
                                    rescued_list.append(json.loads(line.strip()))
                                except json.JSONDecodeError:
                                    pass
                        graph_data = rescued_list

                    if isinstance(graph_data, list):
                        for idx, rel in enumerate(graph_data):
                            src = rel.get("source", "")
                            tgt = rel.get("target", "")
                            r = rel.get("relation", "")
                            if src and tgt and r:
                                chunk = f"Graph Relation: {src} -> {r} -> {tgt}"
                                all_chunks.append(chunk)
                                all_metadata.append({"paper_id": paper_id, "type": "graph_edge"})
                                all_ids.append(f"{paper_id}_edge_{idx}")
                except Exception:
                    pass

        total_chunks = len(all_chunks)
        print(f"Total {total_chunks} chunks to embed.")
        if total_chunks == 0:
            return

        batch_size = 50
        try:
            for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding Batches"):
                batch_text_raw = all_chunks[i:i+batch_size]
                batch_meta_raw = all_metadata[i:i+batch_size]
                batch_id_raw = all_ids[i:i+batch_size]

                valid_texts = []
                valid_metas = []
                valid_ids = []
                embeddings = []

                for idx, text in enumerate(batch_text_raw):
                    try:
                        # Truncate text to avoid context window overflow (500 Error in Ollama bge-m3)
                        safe_text = text[:15000]
                        # Keep bge-m3 in VRAM (keep_alive=300s) for large batch loops
                        emb = self.ollama.get_embeddings(model=self.model_name, prompt=safe_text, keep_alive=300)
                        embeddings.append(emb)
                        valid_texts.append(safe_text)
                        valid_metas.append(batch_meta_raw[idx])
                        valid_ids.append(batch_id_raw[idx])
                    except Exception as inner_e:
                        pass  # Skip failed chunks (usually malformed text from dataset)

                # Use upsert instead of add to allow re-runs after mid-run crashes
                if embeddings:
                    self.collection.upsert(
                        documents=valid_texts,
                        embeddings=embeddings,
                        metadatas=valid_metas,
                        ids=valid_ids
                    )
        except Exception as e:
            print(f"Embedding loop error: {e}")
        finally:
            # --- CRITICAL CONSTRAINT 16GB VRAM EXECUTED ---
            # Unload model after batch to prevent memory leak/model crash
            print("Vectorization complete. Unloading model from VRAM...")
            self.ollama.unload_model(model=self.model_name)

if __name__ == "__main__":
    _root = os.path.join(os.path.dirname(__file__), '..', '..')
    manager = OllamaManager()
    embedder = Embedder(
        manager,
        input_parsed_dir=os.path.join(_root, "data", "parsed"),
        input_graph_dir=os.path.join(_root, "data", "graph"),
        db_dir=os.path.join(_root, "data", "embeddings"),
        model_name="bge-m3"
    )
    # embedder.process_all()
    print("Component 3 is ready.")
