import os
import pandas as pd

def print_metrics(file_path, name):
    if not os.path.exists(file_path):
        print(f"{name}: File not found ({file_path})")
        return
    df = pd.read_csv(file_path)
    metrics = df.mean(numeric_only=True)
    print(f"--- {name} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print()

def main():
    print("========================================")
    print(" OLD RESULTS (Vietnamese Prompt)")
    print("========================================")
    print_metrics("D:/paper/rag_pipeline/data/mass_baseline_ragas.csv", "Baseline")
    print_metrics("D:/paper/rag_pipeline/data/qasper_p3_metrics.csv", "Graph no markdown")
    print_metrics("D:/paper/rag_pipeline/data/mass_graphrag_ragas.csv", "Graph with markdown")
    
    print("========================================")
    print(" NEW RESULTS (English Prompt)")
    print("========================================")
    print_metrics("D:/paper/graph-with-md/data/qasper/results/baseline_metrics.csv", "Baseline")
    print_metrics("D:/paper/graph-with-md/data/qasper/results/graphnomd_metrics.csv", "Graph no markdown")
    print_metrics("D:/paper/graph-with-md/data/qasper/results/graphmd_metrics.csv", "Graph with markdown")

if __name__ == "__main__":
    main()
