# split_dev.py
import json
import random

# Load dev data
with open('../data/preprocessed_dev.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Shuffle and split
random.seed(42)
random.shuffle(data)

val_data = data[:100]   # 100 for validation
test_data = data[100:]  # Rest for testing

# Save
with open('../data/preprocessed_dev_val100.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

with open('../data/preprocessed_dev_test.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print(f"Validation: {len(val_data)} instances")
print(f"Test: {len(test_data)} instances")