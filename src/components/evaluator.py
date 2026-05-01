import os
import sys
import pandas as pd
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    print("WARNING: 'ragas' or 'langchain_openai' library not installed.")

class Evaluator:
    """
    Component 5: Quantify and measure RAG quality using the RAGAS Framework.
    Uses gpt-4o-mini as LLM Judge (future option: swap to Gemma-2-9B).
    """
    def __init__(self, use_local_model: bool = False):
        self.use_local_model = use_local_model

        if not self.use_local_model:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("NOTE: OPENAI_API_KEY not found. Evaluation will fail.")
            else:
                self.eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                self.eval_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            # Future: replace with ChatOllama for local inference
            # from langchain_community.chat_models import ChatOllama
            print("Initializing Ragas Judge with Local LLM (Gemma).")
            self.eval_llm = None
            self.eval_embeddings = None

    def evaluate_dataframe(self, results: list) -> pd.DataFrame:
        """
        results: List[Dict] containing question, answer, contexts, ground_truth (optional)
        """
        print(f"\n[Evaluator] Running RAGAS Benchmarking on {len(results)} queries...")

        metrics = [faithfulness, answer_relevancy]

        formatted_data = {
            "question": [d.get("question", "") for d in results],
            "answer": [d.get("answer", "") for d in results],
            "contexts": [d.get("contexts", []) for d in results],
        }

        if "ground_truth" in results[0]:
            formatted_data["ground_truth"] = [d.get("ground_truth", "") for d in results]
            metrics.extend([context_precision, context_recall])

        dataset = Dataset.from_dict(formatted_data)

        # Call RAGAS engine
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=getattr(self, 'eval_llm', None),
            embeddings=getattr(self, 'eval_embeddings', None)
        )

        print("\n--- RAGAS EVALUATION RESULTS ---")
        print(evaluation_result)

        return evaluation_result.to_pandas()

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # df = Evaluator().evaluate_dataframe([{"question": "test", "answer": "test", "contexts": ["test"]}])
