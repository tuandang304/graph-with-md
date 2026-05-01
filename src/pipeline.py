import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(__file__))
from core.ollama_manager import OllamaManager
from components.loader import QasperLoader
from components.graph_builder import GraphBuilder
from components.embedder import Embedder
from components.generator import Generator
from components.evaluator import Evaluator

class RAGPipeline:
    """
    Main Controller:
    Directly orchestrates 5 sequential steps following the modular architecture.
    Enforces 16GB VRAM rule: VRAM is cleared between each phase transition.
    """
    def __init__(self, data_root: str = None, source_json: str = "qasper-dev-v0.3.json"):
        if data_root is None:
            data_root = os.path.join(_REPO_ROOT, "data")
        self.data_root = data_root
        self.source_json = source_json

        self.raw_dir = os.path.join(data_root, "raw")
        self.parsed_dir = os.path.join(data_root, "parsed")
        self.graph_dir = os.path.join(data_root, "graph")
        self.db_dir = os.path.join(data_root, "embeddings")
        self.original_json_dir = os.path.join(data_root, "raw")

        self.ollama_manager = OllamaManager()

    def step_1_load_data(self):
        print("\n=== STEP 1: QASPER JSON DATA PROCESSING ===")
        loader = QasperLoader(self.original_json_dir, self.parsed_dir)
        loader.process_file(self.source_json)

    def step_2_build_graph(self):
        print("\n=== STEP 2: GRAPH KNOWLEDGE EXTRACTION (QWEN 14B) ===")
        self.ollama_manager.unload_model("qwen2.5:7b")  # Force clear before start
        builder = GraphBuilder(self.ollama_manager, self.parsed_dir, self.graph_dir, model_name="qwen2.5:7b")
        builder.process_all()
        self.ollama_manager.unload_model("qwen2.5:7b")  # Clear after finish

    def step_3_embed_data(self):
        print("\n=== STEP 3: CREATING VECTOR EMBEDDINGS (BGE-M3) ===")
        self.ollama_manager.unload_model("bge-m3")
        embedder = Embedder(self.ollama_manager, self.parsed_dir, self.graph_dir, self.db_dir, model_name="bge-m3")
        embedder.process_all()

    def step_4_query(self, questions: list):
        print("\n=== STEP 4: GENERATION (LLAMA 3) ===")
        # Generator auto-manages keep_alive=0 for both bge-m3 and llama3.1
        generator = Generator(self.ollama_manager, self.db_dir, embed_model="bge-m3", llm_model="llama3.1:8b")

        results = []
        for index, q in enumerate(questions):
            res = generator.query(question=q["question"], top_k=10)
            record = {
                "question": q["question"],
                "answer": res["answer"],
                "contexts": res["context"],
            }
            if "ground_truth" in q:
                record["ground_truth"] = q["ground_truth"]
            results.append(record)

            print(f"\nQ: {q['question']}")
            print(f"A: {res['answer']}")
            print("-" * 50)

        return results

    def step_5_eval(self, results: list):
        print("\n=== STEP 5: EVALUATING WITH RAGAS ===")
        evaluator = Evaluator(use_local_model=False)  # Uses OpenAI gpt-4o-mini
        df = evaluator.evaluate_dataframe(results)
        return df

    def run_full_pipeline(self):
        self.step_1_load_data()
        self.step_2_build_graph()
        self.step_3_embed_data()
        print("\nSystem Built! Knowledge base is stored in local Disk successfully.")
