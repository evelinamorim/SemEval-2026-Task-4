"""
Evaluate Track B Embeddings on Final Exam Triplet Ground Truth.

This script maps story embeddings to their corresponding wikidata_id from the
preprocessed Track B file, computes cosine similarities for each triplet in the
answers file (anchor vs. text_a and text_b), and calculates final accuracy.

Usage:
    # Option A: Using pre-generated embeddings (.npy or .jsonl) from generate_embeddings_drs.py
    python evaluate_final_exam.py \
        --preprocessed track_b_preprocessed.jsonl \
        --answers test_answers.jsonl \
        --embeddings track_b_embeddings.npy \
        --output final_exam_results.json

    # Option B: Generating embeddings on the fly using a model checkpoint
    python evaluate_final_exam.py \
        --preprocessed track_b_preprocessed.jsonl \
        --answers test_answers.jsonl \
        --checkpoint ./checkpoints_five \
        --output final_exam_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Import functions directly from your fixed generate_embeddings_drs.py
from generate_embeddings_drs import (
    load_preprocessed_data,
    load_model_and_config,
    generate_embeddings,
)


def load_json_or_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from either a JSON array file or a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return []
        if content.startswith('['):
            return json.loads(content)
        for line in content.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_precomputed_embeddings(embeddings_path: str) -> np.ndarray:
    """Load precomputed embeddings from a .npy or .jsonl file."""
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    if path.suffix == '.npy':
        return np.load(path)
    elif path.suffix == '.jsonl':
        emb_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    emb_list.append(obj['embeddings'])
        return np.array(emb_list, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported embedding file format: {path.suffix}. Use .npy or .jsonl")


def build_wikidata_embedding_map(
    preprocessed_stories: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Associate each story's embedding with its wikidata_id.
    """
    if len(preprocessed_stories) != len(embeddings):
        raise ValueError(
            f"Length mismatch: {len(preprocessed_stories)} preprocessed stories "
            f"vs {len(embeddings)} embeddings."
        )

    emb_map: Dict[str, torch.Tensor] = {}
    missing_id_count = 0

    for item, emb_vec in zip(preprocessed_stories, embeddings):
        wiki_id = item.get('wikidata_id')
        if wiki_id:
            emb_map[wiki_id] = torch.tensor(emb_vec, dtype=torch.float32)
        else:
            missing_id_count += 1

    if missing_id_count > 0:
        print(f"⚠ Warning: {missing_id_count} stories did not have a 'wikidata_id' field.")

    if len(emb_map) == 0:
        raise RuntimeError(
            "No 'wikidata_id' fields were found in the preprocessed file! "
            "Please re-run preprocess_trackb.py with the updated script."
        )

    return emb_map


def evaluate_triplets(
    answers_data: List[Dict[str, Any]],
    emb_map: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Evaluate triplet predictions by comparing cosine similarity between
    (anchor, text_a) and (anchor, text_b) using wikidata_id lookups.
    """
    results = []
    correct_count = 0
    skipped_count = 0

    # Breakdown by annotation categories if present in the test answers
    sub_metrics = {
        'abstract_theme': {'correct': 0, 'total': 0},
        'course_of_actions': {'correct': 0, 'total': 0},
        'outcomes': {'correct': 0, 'total': 0},
    }

    for idx, item in enumerate(answers_data):
        try:
            anchor_id = item['story_details_anchor']['wikidata_id']
            a_id = item['story_details_a']['wikidata_id']
            b_id = item['story_details_b']['wikidata_id']
        except KeyError as e:
            print(f"  [Row {idx}] Skipping triplet due to missing key: {e}")
            skipped_count += 1
            continue

        # Check that all 3 embeddings exist in our map
        missing_ids = [wid for wid in (anchor_id, a_id, b_id) if wid not in emb_map]
        if missing_ids:
            print(f"  [Row {idx}] Skipping triplet; wikidata_id not found in lookup: {missing_ids}")
            skipped_count += 1
            continue

        anchor_emb = emb_map[anchor_id]
        a_emb = emb_map[a_id]
        b_emb = emb_map[b_id]

        # Compute cosine similarity (higher means closer to anchor)
        sim_a = F.cosine_similarity(anchor_emb.unsqueeze(0), a_emb.unsqueeze(0)).item()
        sim_b = F.cosine_similarity(anchor_emb.unsqueeze(0), b_emb.unsqueeze(0)).item()

        logits = sim_a - sim_b
        pred_a_is_closer = bool(sim_a > sim_b)
        label_a_is_closer = bool(item['text_a_is_closer'])

        is_correct = (pred_a_is_closer == label_a_is_closer)
        if is_correct:
            correct_count += 1

        # Track sub-aspect agreement (majority vote of boolean lists if available)
        for aspect in sub_metrics:
            if aspect in item and isinstance(item[aspect], list) and len(item[aspect]) > 0:
                valid_votes = [v for v in item[aspect] if v is not None]
                if valid_votes:
                    aspect_label = sum(1 for v in valid_votes if v) > (len(valid_votes) / 2)
                    sub_metrics[aspect]['total'] += 1
                    if pred_a_is_closer == aspect_label:
                        sub_metrics[aspect]['correct'] += 1

        results.append({
            'idx': idx,
            'anchor_wikidata_id': anchor_id,
            'a_wikidata_id': a_id,
            'b_wikidata_id': b_id,
            'sim_a': sim_a,
            'sim_b': sim_b,
            'logits': logits,
            'pred_a_is_closer': pred_a_is_closer,
            'label_a_is_closer': label_a_is_closer,
            'correct': is_correct,
        })

    total_evaluated = len(results)
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0.0

    logits_list = [r['logits'] for r in results]
    confident_correct = sum(1 for r in results if r['correct'] and abs(r['logits']) > 0.1)
    confident_wrong = sum(1 for r in results if not r['correct'] and abs(r['logits']) > 0.1)
    uncertain = sum(1 for r in results if abs(r['logits']) <= 0.1)

    aspect_accuracies = {
        k: (v['correct'] / v['total'] if v['total'] > 0 else 0.0)
        for k, v in sub_metrics.items()
    }

    return {
        'accuracy': accuracy,
        'correct': correct_count,
        'total_evaluated': total_evaluated,
        'skipped': skipped_count,
        'confident_correct': confident_correct,
        'confident_wrong': confident_wrong,
        'uncertain': uncertain,
        'mean_logits': float(np.mean(logits_list)) if logits_list else 0.0,
        'std_logits': float(np.std(logits_list)) if logits_list else 0.0,
        'aspect_agreement_accuracy': aspect_accuracies,
        'per_example': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Track B embeddings against final exam triplet answers'
    )
    parser.add_argument(
        '--preprocessed',
        type=str,
        required=True,
        help='Path to preprocessed stories JSONL (must contain wikidata_id)',
    )
    parser.add_argument(
        '--answers',
        type=str,
        required=True,
        help='Path to test answers JSON/JSONL containing triplets and ground truth labels',
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to precomputed embeddings (.npy or .jsonl). If omitted, --checkpoint is used.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint directory to generate embeddings on the fly if --embeddings is not provided',
    )
    parser.add_argument(
        '--save_embeddings',
        type=str,
        default=None,
        help='Optional path to save generated .npy embeddings when using --checkpoint',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='final_exam_results.json',
        help='Path to save detailed evaluation results JSON',
    )
    parser.add_argument(
        '--show_errors',
        action='store_true',
        help='Print misclassified triplets to console',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device: cuda, cpu, or auto (default: auto)',
    )

    args = parser.parse_args()

    if args.embeddings is None and args.checkpoint is None:
        parser.error("You must provide either --embeddings (precomputed) or --checkpoint (to compute them).")

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device

    print("\n" + "=" * 70)
    print("FINAL EXAM EVALUATION (TRACK B EMBEDDINGS -> TRIPLET ANSWERS)")
    print("=" * 70)

    # 1. Load preprocessed stories (with wikidata_id)
    print(f"Loading preprocessed stories from: {args.preprocessed}")
    preprocessed_stories = load_preprocessed_data(args.preprocessed)
    print(f"✓ Loaded {len(preprocessed_stories)} stories")

    # 2. Load or generate embeddings
    if args.embeddings is not None:
        print(f"Loading precomputed embeddings from: {args.embeddings}")
        embeddings = load_precomputed_embeddings(args.embeddings)
        print(f"✓ Loaded embeddings with shape: {embeddings.shape}")
    else:
        inference_model = load_model_and_config(args.checkpoint, device)
        embeddings = generate_embeddings(inference_model, preprocessed_stories)
        if args.save_embeddings:
            out_npy = Path(args.save_embeddings).with_suffix('.npy')
            np.save(out_npy, embeddings)
            print(f"✓ Saved generated embeddings to: {out_npy}")

    # 3. Map wikidata_id -> embedding tensor
    emb_map = build_wikidata_embedding_map(preprocessed_stories, embeddings)
    print(f"✓ Indexed {len(emb_map)} unique wikidata_id embeddings")

    # 4. Load test answers and evaluate
    print(f"\nLoading test answers from: {args.answers}")
    answers_data = load_json_or_jsonl(args.answers)
    print(f"✓ Loaded {len(answers_data)} test triplets")

    metrics = evaluate_triplets(answers_data, emb_map)

    # 5. Print summary
    print("\n" + "-" * 70)
    print("FINAL EXAM RESULTS")
    print("-" * 70)
    print(f"Total Triplets Evaluated: {metrics['total_evaluated']} (Skipped: {metrics['skipped']})")
    print(f"Correct Predictions:      {metrics['correct']} / {metrics['total_evaluated']}")
    print(f"Final Accuracy:           {metrics['accuracy'] * 100:.2f}%")

    print("\nConfidence Analysis:")
    print(f"  Confident & correct (|logit| > 0.1): {metrics['confident_correct']}")
    print(f"  Confident & wrong   (|logit| > 0.1): {metrics['confident_wrong']}")
    print(f"  Uncertain           (|logit| <= 0.1): {metrics['uncertain']}")

    print("\nAgreement with Human Sub-Labels:")
    for aspect, acc in metrics['aspect_agreement_accuracy'].items():
        print(f"  {aspect:20s}: {acc * 100:.2f}%")

    if args.show_errors:
        errors = [r for r in metrics['per_example'] if not r['correct']]
        print("\n" + "-" * 70)
        print(f"MISCLASSIFIED EXAMPLES (Showing first 10 of {len(errors)})")
        print("-" * 70)
        for err in errors[:10]:
            print(
                f"  Triplet {err['idx']:4d} | Anchor: {err['anchor_wikidata_id']} | "
                f"Sim A ({err['a_wikidata_id']}): {err['sim_a']:.4f} | "
                f"Sim B ({err['b_wikidata_id']}): {err['sim_b']:.4f} | "
                f"Pred: {'A' if err['pred_a_is_closer'] else 'B'} "
                f"(Actual: {'A' if err['label_a_is_closer'] else 'B'})"
            )

    # 6. Save output JSON
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Detailed results saved to: {args.output}")

    print("=" * 70)


if __name__ == "__main__":
    main()