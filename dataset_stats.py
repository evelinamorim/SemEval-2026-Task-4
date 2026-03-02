"""
Compute dataset statistics from a SemEval-2026 Task 4 JSONL file.

Usage:
    python dataset_stats.py <jsonl_file>
"""

import json
import sys
import re
from collections import Counter
import statistics


def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def split_sentences(text):
    """Split text into sentences using simple rules."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def compute_stats(jsonl_path):
    all_tokens = []
    all_unique_tokens = Counter()
    all_sentence_counts = []

    role_stats = {role: {"tokens": [], "sentences": []} for role in ("anchor_text", "text_a", "text_b")}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)

            for role in ("anchor_text", "text_a", "text_b"):
                text = instance.get(role, "")
                tokens = tokenize(text)
                sentences = split_sentences(text)

                all_tokens.extend(tokens)
                all_unique_tokens.update(tokens)
                all_sentence_counts.append(len(sentences))

                role_stats[role]["tokens"].append(len(tokens))
                role_stats[role]["sentences"].append(len(sentences))

    n_stories = len(all_sentence_counts)

    print("=" * 55)
    print("DATASET STATISTICS")
    print("=" * 55)
    print(f"{'Total stories (all roles):':<40} {n_stories}")
    print(f"{'Total tokens:':<40} {len(all_tokens)}")
    print(f"{'Total unique tokens:':<40} {len(all_unique_tokens)}")
    print(f"{'Average tokens per story:':<40} {len(all_tokens) / n_stories:.2f} (±{statistics.stdev(role_stats['anchor_text']['tokens'] + role_stats['text_a']['tokens'] + role_stats['text_b']['tokens']):.2f})")
    print(f"{'Average sentences per story:':<40} {sum(all_sentence_counts) / n_stories:.2f} (±{statistics.stdev(all_sentence_counts):.2f})")
    print(f"{'Total sentences:':<40} {sum(all_sentence_counts)}")


    print()
    print("-" * 55)
    print("PER ROLE BREAKDOWN")
    print("-" * 55)
    role_labels = {"anchor_text": "Anchor", "text_a": "Story A", "text_b": "Story B"}
    for role, label in role_labels.items():
        token_counts = role_stats[role]["tokens"]
        sent_counts = role_stats[role]["sentences"]
        n = len(token_counts)
        print(f"\n  {label} ({n} stories)")
        print(f"    {'Avg tokens:':<30} {sum(token_counts) / n:.2f} (±{statistics.stdev(token_counts):.2f})")
        print(f"    {'Min/Max tokens:':<30} {min(token_counts)} / {max(token_counts)}")
        print(f"    {'Avg sentences:':<30} {sum(sent_counts) / n:.2f} (±{statistics.stdev(sent_counts):.2f})")
        print(f"    {'Min/Max sentences:':<30} {min(sent_counts)} / {max(sent_counts)}")

    print()
    print("=" * 55)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dataset_stats.py <jsonl_file>")
        sys.exit(1)

    compute_stats(sys.argv[1])