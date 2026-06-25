import os
import sys
import json
import chromadb
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.ollama_manager import OllamaManager
from src.components.knowledge_graph import KnowledgeGraph

class P3Embedder:
    """
    Graph no markdown: Fixed-size Chunking + Knowledge Graph.
    Keeps Qwen-extracted graph relations, but uses static chunking (1000 chars)
    on raw text instead of markdown section boundaries.

    Graph embeddings use node-centric context from the NetworkX KnowledgeGraph:
    each significant entity gets a multi-hop neighborhood description embedded
    as a rich ``graph_context`` chunk (same approach as the main Embedder).
    """
    def __init__(self, ollama_manager: OllamaManager, txt_dir: str, graph_dir: str, db_dir: str, model_name: str = "bge-m3"):
        self.ollama = ollama_manager
        self.txt_dir = txt_dir
        self.graph_dir = graph_dir
        self.db_dir = db_dir
        self.model_name = model_name

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        # Use same collection name as Pipeline 1 to reuse Generator
        self.collection = self.chroma_client.get_or_create_collection(
            name="qasper_graph_rag",
            metadata={"hnsw:space": "cosine"}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Load the NetworkX Knowledge Graph
        self.kg = KnowledgeGraph(self.graph_dir)
        self.kg.load_all()
        self.kg.load_all_from_json()

    def process_all(self):
        print(f"Starting Pipeline 3 vectorization with {self.model_name}...")

        all_chunks = []
        all_metadata = []
        all_ids = []

        # 1. Read Baseline Text (Fixed-size chunks)
        txt_files = [f for f in os.listdir(self.txt_dir) if f.endswith('.txt')]
        for file in txt_files:
            paper_id = file.replace('.txt', '')
            with open(os.path.join(self.txt_dir, file), 'r', encoding='utf-8') as f:
                text = f.read()

            chunks = self.text_splitter.split_text(text)

            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({"paper_id": paper_id, "type": "baseline_chunk"})
                all_ids.append(f"{paper_id}_base_{idx}")

        # 2. Graph: node-centric context from NetworkX Knowledge Graph
        kg_stats = self.kg.stats()
        print(f"Knowledge Graph loaded: {kg_stats['nodes']} nodes, {kg_stats['edges']} edges")

        processed_papers = set()
        graph_files = [f for f in os.listdir(self.graph_dir)
                       if f.endswith('_graph.json') or f.endswith('_graph.graphml')]
        for file in graph_files:
            if file.endswith('_graph.json'):
                paper_id = file.replace('_graph.json', '')
            else:
                paper_id = file.replace('_graph.graphml', '')
            if paper_id in processed_papers:
                continue
            processed_papers.add(paper_id)

            # Get all entities for this paper
            entities = self.kg.get_entities_for_paper(paper_id)
            if not entities:
                continue

            for idx, entity in enumerate(entities):
                # Generate rich context for this entity (2-hop neighborhood)
                context = self.kg.get_entity_context(entity, hops=2)
                if not context or len(context) < 20:
                    continue

                all_chunks.append(context)
                all_metadata.append({
                    "paper_id": paper_id,
                    "type": "graph_context",
                    "entity": entity
                })
                all_ids.append(f"{paper_id}_entity_{idx}")

        # Filter already-embedded chunks to allow resume
        existing_data = self.collection.get(include=[])
        existing_ids = set(existing_data['ids']) if existing_data and 'ids' in existing_data else set()

        filtered_chunks = []
        filtered_metadata = []
        filtered_ids = []

        for ch, meta, ch_id in zip(all_chunks, all_metadata, all_ids):
            if ch_id not in existing_ids:
                filtered_chunks.append(ch)
                filtered_metadata.append(meta)
                filtered_ids.append(ch_id)

        total_chunks = len(filtered_chunks)
        print(f"Total {len(all_chunks)} chunks, already embedded {len(existing_ids)}, {total_chunks} remaining (P3 Mixed).")

        if total_chunks == 0:
            print("All chunks already embedded for P3.")
            return

        batch_size = 50
        try:
            for i in tqdm(range(0, total_chunks, batch_size), desc="P3 Batch Embed"):
                batch_text = filtered_chunks[i:i+batch_size]
                batch_meta = filtered_metadata[i:i+batch_size]
                batch_id = filtered_ids[i:i+batch_size]

                valid_texts = []
                valid_metas = []
                valid_ids = []
                embeddings = []

                for idx, text in enumerate(batch_text):
                    try:
                        safe_text = text[:15000]
                        emb = self.ollama.get_embeddings(model=self.model_name, prompt=safe_text, keep_alive=300)
                        embeddings.append(emb)
                        valid_texts.append(safe_text)
                        valid_metas.append(batch_meta[idx])
                        valid_ids.append(batch_id[idx])
                    except Exception as inner_e:
                        pass  # Skip failed chunk

                if embeddings:
                    self.collection.upsert(
                        documents=valid_texts,
                        embeddings=embeddings,
                        metadatas=valid_metas,
                        ids=valid_ids
                    )
        except Exception as e:
            print(f"P3 embedder outer loop error: {e}")
        finally:
            self.ollama.unload_model(model=self.model_name)

if __name__ == "__main__":
    pass
