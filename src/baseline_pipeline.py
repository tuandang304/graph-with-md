import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(__file__))
from src.core.ollama_manager import OllamaManager
from src.baseline.loader import BaselineLoader
from src.baseline.embedder import BaselineEmbedder
from src.baseline.generator import BaselineGenerator
from src.baseline.evaluator import IREvaluator

class BaselinePipeline:
    def __init__(self, data_root: str = None, source_json: str = "qasper-dev-v0.3.json"):
        if data_root is None:
            data_root = os.path.join(_REPO_ROOT, "data")
        self.data_root = data_root
        self.source_json = source_json

        self.original_json_dir = os.path.join(data_root, "raw")

        # Baseline output domains
        self.baseline_parsed_dir = os.path.join(data_root, "baseline_parsed")
        self.db_dir = os.path.join(data_root, "baseline_embeddings")

        self.ollama_manager = OllamaManager()

    def run_ingestion(self):
        print("\n=== BASELINE PIPELINE INGESTION ===")
        # 1. Flatten JSON structure into simulated PDF text (Loader)
        loader = BaselineLoader(self.original_json_dir, self.baseline_parsed_dir)
        loader.process_file(self.source_json)

        # 2. Vectorize with fixed Chunk Size (Embedder)
        self.ollama_manager.unload_model("bge-m3")
        embedder = BaselineEmbedder(self.ollama_manager, self.baseline_parsed_dir, self.db_dir, model_name="bge-m3")
        embedder.process_all()
        print("\n[Baseline Ingestion Complete]")

    def run_qa_and_evaluate(self, test_data: list):
        print("\n=== BASELINE QA & EVALUATION ===")
        generator = BaselineGenerator(self.ollama_manager, self.db_dir, embed_model="bge-m3", llm_model="qwen2.5:7b-instruct-q4_K_M")

        results = []
        for index, q in enumerate(test_data):
            res = generator.query(question=q["question"], top_k=5)
            record = {
                "question": q["question"],
                "answer": res["answer"],
                "contexts": res["context"],
            }
            if "ground_truth" in q:
                record["ground_truth"] = q["ground_truth"]
            results.append(record)

        print("\n[Benchmarking with RAGAS (GPT-4)...]")
        from src.components.evaluator import Evaluator
        eval_ragas = Evaluator(use_local_model=False)
        df_ragas = eval_ragas.evaluate_dataframe(results)
        return df_ragas
