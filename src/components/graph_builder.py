import os
import sys
import json
from tqdm import tqdm

# Ensure internal imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.ollama_manager import OllamaManager
from src.components.knowledge_graph import KnowledgeGraph

class GraphBuilder:
    """
    Component 2: Extracts vertices and edges from Markdown documents.
    Uses Ollama Manager with keep_alive=0 via Qwen model.

    After LLM extraction, triplets are:
    1. Saved as JSON (backward compatible)
    2. Inserted into a NetworkX KnowledgeGraph and persisted as .graphml
    """
    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert NLP AI extracting knowledge graph mappings. "
        "Given the markdown text (which may contain text and references to figure captions), "
        "extract the most important entities and their direct relationships. "
        "Output EXCLUSIVELY a JSON array format, where each object is: "
        '{"source": "Entity A", "target": "Entity B", "relation": "relationship type"}. '
        "Do not include any wrapping markdown formatting like ```json. Limit to top 20 relationships."
    )

    def __init__(self, ollama_manager: OllamaManager, input_dir: str, output_dir: str, model_name: str = "qwen2.5:7b", system_prompt: str = None):
        self.ollama = ollama_manager
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.chunk_char_limit = 8000  # Limit to 8000 chars to avoid context length overflow

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.system_prompt = system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT

        # NetworkX Knowledge Graph — accumulates triplets across all papers
        self.kg = KnowledgeGraph(self.output_dir)

    def build_graph_for_file(self, filename: str, keep_alive: int = 0):
        filepath = os.path.join(self.input_dir, filename)
        paper_id = filename.replace('.md', '')
        output_filepath = os.path.join(self.output_dir, f"{paper_id}_graph.json")

        if os.path.exists(output_filepath):
            # JSON already exists — still load into NetworkX for downstream use
            self.kg.load_from_json(paper_id)
            # Persist .graphml if not already present
            graphml_path = os.path.join(self.output_dir, f"{paper_id}_graph.graphml")
            if not os.path.exists(graphml_path):
                self.kg.save(paper_id)
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Truncate content to fit LLM context window
        text_chunk = text[:self.chunk_char_limit]
        prompt = f"Text:\n{text_chunk}\n\nStrictly reply with the JSON array only."

        try:
            response = self.ollama.generate(
                model=self.model_name,
                prompt=prompt,
                system=self.system_prompt,
                keep_alive=keep_alive
            )

            # Strip excess markdown from LLM output
            cleaned_resp = response.strip()
            if cleaned_resp.startswith("```json"):
                cleaned_resp = cleaned_resp[7:]
            if cleaned_resp.startswith("```"):
                cleaned_resp = cleaned_resp[3:]
            if cleaned_resp.endswith("```"):
                cleaned_resp = cleaned_resp[:-3]

            cleaned_resp = cleaned_resp.strip()

            try:
                graph_data = json.loads(cleaned_resp)
            except json.JSONDecodeError:
                # If LLM fails JSON, log raw text for dev inspection
                print(f"[{paper_id}] JSONDecodeError. Saving log file.")
                graph_data = {"error": "Invalid output format", "raw_output": cleaned_resp}

            # 1. Save JSON (backward compatible)
            with open(output_filepath, 'w', encoding='utf-8') as out_f:
                json.dump(graph_data, out_f, ensure_ascii=False, indent=2)

            # 2. Build NetworkX graph and persist .graphml
            if isinstance(graph_data, list):
                self.kg.add_triplets(paper_id, graph_data)
                self.kg.save(paper_id)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    def process_all(self):
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.md')]
        print(f"Starting Knowledge Graph construction with {self.model_name}...")
        try:
            for file in tqdm(files):
                # keep_alive=300: hold model in VRAM across batch, unload once at end
                self.build_graph_for_file(file, keep_alive=300)
        finally:
            self.ollama.unload_model(self.model_name)
        stats = self.kg.stats()
        print(f"Knowledge Graph built: {stats['nodes']} nodes, {stats['edges']} edges, {stats['components']} components")

if __name__ == "__main__":
    _root = os.path.join(os.path.dirname(__file__), '..', '..')
    manager = OllamaManager()
    builder = GraphBuilder(
        manager,
        input_dir=os.path.join(_root, "data", "parsed"),
        output_dir=os.path.join(_root, "data", "graph"),
        model_name="qwen2.5:7b"
    )

    files = os.listdir(os.path.join(_root, "data", "parsed"))
    if len(files) > 0:
        test_file = files[0]
        print(f"Testing Graph Builder on {test_file}...")
        # NOTE: Will make an HTTP call via Ollama
        # builder.build_graph_for_file(test_file)
        print("Graph Builder component 2 test done.")
