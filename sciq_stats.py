"""
SciQ Statistics — Download and analyze the SciQ dataset (allenai/sciq).
Science textbook question answering — 13,679 QA pairs with support paragraphs.
"""
from datasets import load_dataset
import numpy as np

print("Downloading SciQ (allenai/sciq) from HuggingFace...")
ds = load_dataset("allenai/sciq")

train = ds["train"]
val = ds["validation"]
test = ds["test"]

print(f"\n{'='*55}")
print(f"  SCIQ DATASET STATISTICS")
print(f"{'='*55}")
print(f"  Train:      {len(train):,} samples")
print(f"  Validation: {len(val):,} samples")
print(f"  Test:       {len(test):,} samples")
print(f"  Total:      {len(train)+len(val)+len(test):,} samples")

# Analyze all splits combined
all_data = list(train) + list(val) + list(test)

question_lens = []
support_lens = []
answer_lens = []
empty_support = 0

for item in all_data:
    question_lens.append(len(item["question"].split()))
    
    support = item["support"].strip()
    if not support:
        empty_support += 1
    support_lens.append(len(support.split()) if support else 0)
    
    answer_lens.append(len(item["correct_answer"].split()))

print(f"\n--- Length Statistics (in words) ---")
print(f"1. Question:")
print(f"   - Average: {np.mean(question_lens):.1f}")
print(f"   - Min: {np.min(question_lens)}")
print(f"   - Max: {np.max(question_lens)}")

print(f"\n2. Support Paragraph (textbook context):")
print(f"   - Average: {np.mean(support_lens):.1f}")
print(f"   - Min: {np.min(support_lens)}")
print(f"   - Max: {np.max(support_lens)}")
print(f"   - Empty support: {empty_support} / {len(all_data)}")

print(f"\n3. Correct Answer:")
print(f"   - Average: {np.mean(answer_lens):.1f}")
print(f"   - Min: {np.min(answer_lens)}")
print(f"   - Max: {np.max(answer_lens)}")

# Show example
print(f"\n--- Example (first from test split) ---")
item = test[0]
print(f"  Q: {item['question']}")
print(f"  A: {item['correct_answer']}")
print(f"  Support: {item['support'][:300]}...")
print(f"  Distractors: {item['distractor1']}, {item['distractor2']}, {item['distractor3']}")
