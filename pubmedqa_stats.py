from datasets import load_dataset
import numpy as np

print("Downloading PubMedQA (pqa_labeled) from HuggingFace...")
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

print("\n--- PUBMEDQA (PQA_LABELED) STATISTICS ---")
print(f"Total samples (QA pairs): {len(ds)}")

question_lens = []
context_lens = []
answer_lens = []

for item in ds:
    question = item['question']
    contexts = item['context']['contexts']
    answer = item['long_answer']
    
    question_lens.append(len(question.split()))
    full_context = " ".join(contexts)
    context_lens.append(len(full_context.split()))
    answer_lens.append(len(answer.split()))

print("\nLength statistics (in words):")
print(f"1. Question:")
print(f"   - Average: {np.mean(question_lens):.1f} words")
print(f"   - Min: {np.min(question_lens)} words")
print(f"   - Max: {np.max(question_lens)} words")

print(f"\n2. Context (Abstracts):")
print(f"   - Average: {np.mean(context_lens):.1f} words")
print(f"   - Min: {np.min(context_lens)} words")
print(f"   - Max: {np.max(context_lens)} words")

print(f"\n3. Long Answer (Ground truth):")
print(f"   - Average: {np.mean(answer_lens):.1f} words")
print(f"   - Min: {np.min(answer_lens)} words")
print(f"   - Max: {np.max(answer_lens)} words")
