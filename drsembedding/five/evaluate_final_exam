"""
Evaluate Track B Embeddings on Final Exam Triplet Ground Truth via Exact Text Matching.

Follows the official competition logic (join_labels.py):
1. Generates/loads embeddings in the exact order of the 849 stories.
2. Associates each embedding with its full story text.
3. Matches each triplet slot (anchor_text, text_a, text_b) by exact text to its embedding.

Usage:
    # Option A: Using pre-generated embeddings (.npy or .jsonl)
    python evaluate_final_exam.py \
        --preprocessed NarrativeSimilarity/test/track_b_preprocessed.jsonl \
        --stories NarrativeSimilarity/test/test_track_b.jsonl \
        --answers NarrativeSimilarity/test/test_track_b_labels.jsonl \
        --embeddings track_b_embeddings.npy \
        --output final_exam_results.json

    # Option B: Generating embeddings on the fly using a model checkpoint
    python evaluate_final_exam.py \
        --preprocessed NarrativeSimilarity/test/track_b_preprocessed.jsonl \
        --stories NarrativeSimilarity/test/test_track_b.jsonl \
        --answers NarrativeSimilarity/test/test_track_b_labels.jsonl \
        --checkpoint ./checkpoints_five \
        --output final_exam_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from drsembedding.five.generate_embeddings_drs import (
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
    """Load precomputed embeddings from a .npy or .jsonl file in exact order."""
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


def build_text_embedding_map(
    stories_source: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Step 2: Associate each generated embedding (in order) with its full story text.
    """
    if len(stories_source) != len(embeddings):
        raise ValueError(
            f"Length mismatch: {len(stories_source)} stories "
            f"vs {len(embeddings)} embeddings."
        )

    text_to_emb: Dict[str, torch.Tensor] = {}
    missing_text_count = 0

    for idx, (item, emb_vec) in enumerate(zip(stories_source, embeddings)):
        text = item.get('text')
        if text is not None and text != "":
            # Map both exact string and stripped string for safety
            tensor_emb = torch.tensor(emb_vec, dtype=torch.float32)
            text_to_emb[text] = tensor_emb
            text_to_emb[text.strip()] = tensor_emb
        else:
            missing_text_count += 1

    if missing_text_count > 0:
        print(f"⚠ Warning: {missing_text_count} stories did not have a valid 'text' field.")

    if len(text_to_emb) == 0:
        raise RuntimeError(
            "No 'text' fields were found in the stories file! "
            "Pass the raw Track B stories JSONL via --stories or ensure 'text' is in --preprocessed."
        )

    return text_to_emb


def lookup_embedding_by_text(text: str, emb_map: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """Find embedding by exact text match (with fallback to stripped text)."""
    if text in emb_map:
        return emb_map[text]
    stripped = text.strip()
    if stripped in emb_map:
        return emb_map[stripped]
    return None


def evaluate_triplets(
    answers_data: List[Dict[str, Any]],
    emb_map: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Step 3: Match each triplet slot (anchor_text, text_a, text_b) by exact text
    to its embedding and evaluate cosine similarity.
    """
    results = []
    correct_count = 0
    skipped_count = 0

    sub_metrics = {
        'abstract_theme': {'correct': 0, 'total': 0},
        'course_of_actions': {'correct': 0, 'total': 0},
        'outcomes': {'correct': 0, 'total': 0},
    }

    for idx, item in enumerate(answers_data):
        try:
            anchor_text = item['anchor_text']
            text_a = item['text_a']
            text_b = item['text_b']
        except KeyError as e:
            print(f"  [Row {idx}] Skipping triplet due to missing text key: {e}")
            skipped_count += 1
            continue

        anchor_emb = lookup_embedding_by_text(anchor_text, emb_map)
        a_emb = lookup_embedding_by_text(text_a, emb_map)
        b_emb = lookup_embedding_by_text(text_b, emb_map)

        missing_slots = []
        if anchor_emb is None:
            missing_slots.append('anchor_text')
        if a_emb is None:
            missing_slots.append('text_a')
        if b_emb is None:
            missing_slots.append('text_b')

        if missing_slots:
            print(f"  [Row {idx}] Skipping triplet; exact text not found in bank for: {missing_slots}")
            skipped_count += 1
            continue

        # Compute cosine similarity (higher means closer to anchor)
        sim_a = F.cosine_similarity(anchor_emb.unsqueeze(0), a_emb.unsqueeze(0)).item()
        sim_b = F.cosine_similarity(anchor_emb.unsqueeze(0), b_emb.unsqueeze(0)).item()

        logits = sim_a - sim_b
        pred_a_is_closer = bool(sim_a > sim_b)
        label_a_is_closer = bool(item['text_a_is_closer'])

        is_correct = (pred_a_is_closer == label_a_is_closer)
        if is_correct:
            correct_count += 1

        # Track sub-aspect agreement if available
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
            'anchor_preview': anchor_text[:60].replace('\n', ' ') + '...',
            'a_preview': text_a[:60].replace('\n', ' ') + '...',
            'b_preview': text_b[:60].replace('\n', ' ') + '...',
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
        description='Evaluate Track B embeddings against final exam triplet answers using exact text matching'
    )
    parser.add_argument(
        '--preprocessed',
        type=str,
        required=True,
        help='Path to preprocessed stories JSONL (in the exact order of Track B stories)',
    )
    parser.add_argument(
        '--stories',
        type=str,
        default=None,
        help='Optional path to raw Track B plain stories JSONL (e.g., test_track_b.jsonl). '
             'If provided, texts are read from here in order; otherwise read from --preprocessed.',
    )
    parser.add_argument(
        '--answers',
        type=str,
        required=True,
        help='Path to triplets JSON/JSONL containing anchor_text, text_a, text_b, and text_a_is_closer',
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
    print("FINAL EXAM EVALUATION (EXACT TEXT MATCHING)")
    print("=" * 70)

    # 1. Load preprocessed stories (and raw stories if provided)
    print(f"Loading preprocessed stories from: {args.preprocessed}")
    preprocessed_stories = load_preprocessed_data(args.preprocessed)
    print(f"✓ Loaded {len(preprocessed_stories)} preprocessed stories")

    if args.stories is not None:
        print(f"Loading plain story texts from: {args.stories}")
        stories_for_text = load_json_or_jsonl(args.stories)
        print(f"✓ Loaded {len(stories_for_text)} plain story texts")
    else:
        stories_for_text = preprocessed_stories

    # 2. Step 1: Generate or load embeddings in exact story order
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

    # 3. Step 2: Associate embeddings with full story texts
    emb_map = build_text_embedding_map(stories_for_text, embeddings)
    print(f"✓ Indexed embeddings for {len(stories_for_text)} stories by exact text")

    # 4. Step 3: Load test triplets and match each element by exact text
    print(f"\nLoading test triplets from: {args.answers}")
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
                f"  Triplet {err['idx']:4d} | "
                f"Sim A: {err['sim_a']:.4f} | Sim B: {err['sim_b']:.4f} | "
                f"Pred: {'A' if err['pred_a_is_closer'] else 'B'} "
                f"(Actual: {'A' if err['label_a_is_closer'] else 'B'})\n"
                f"    Anchor: {err['anchor_preview']}"
            )

    # 6. Save output JSON
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Detailed results saved to: {args.output}")

    print("=" * 70)


if __name__ == "__main__":
    main()