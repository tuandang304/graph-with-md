import os
import sys
import chromadb
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.ollama_manager import OllamaManager

class BaselineEmbedder:
    """
    Pipeline 2 (Baseline):
    Uses Langchain RecursiveCharacterTextSplitter to split text into fixed-size chunks.
    This is the Naive RAG standard — intentionally ignores Semantic Section structure
    to serve as the baseline demonstrating Pipeline 1's superiority.
    """
    def __init__(self, ollama_manager: OllamaManager, input_dir: str, db_dir: str, model_name: str = "bge-m3"):
        self.ollama = ollama_manager
        self.input_dir = input_dir
        self.db_dir = db_dir
        self.model_name = model_name

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="baseline_rag",
            metadata={"hnsw:space": "cosine"}
        )

        # Fragmented Splitting standard: 1000 chars, 200 char overlap.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def process_all(self):
        print(f"Starting Baseline vectorization with {self.model_name}...")

        txt_files = [f for f in os.listdir(self.input_dir) if f.endswith('.txt')]
        all_chunks = []
        all_metadata = []
        all_ids = []

        for file in txt_files:
            paper_id = file.replace('.txt', '')
            with open(os.path.join(self.input_dir, file), 'r', encoding='utf-8') as f:
                text = f.read()

            chunks = self.text_splitter.split_text(text)

            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({"paper_id": paper_id, "type": "baseline_chunk"})
                all_ids.append(f"{paper_id}_base_{idx}")

        total_chunks = len(all_chunks)
        print(f"Total {total_chunks} chunks to embed (Baseline Fragmented).")

        batch_size = 50
        try:
            for i in tqdm(range(0, total_chunks, batch_size), desc="Baseline Batch Embed"):
                batch_text = all_chunks[i:i+batch_size]
                batch_meta = all_metadata[i:i+batch_size]
                batch_id = all_ids[i:i+batch_size]

                embeddings = []
                for text in batch_text:
                    emb = self.ollama.get_embeddings(model=self.model_name, prompt=text, keep_alive=300)
                    embeddings.append(emb)

                if embeddings:
                    self.collection.upsert(
                        documents=batch_text,
                        embeddings=embeddings,
                        metadatas=batch_meta,
                        ids=batch_id
                    )
        except Exception as e:
            print(f"Baseline embedder error: {e}")
        finally:
            self.ollama.unload_model(model=self.model_name)
