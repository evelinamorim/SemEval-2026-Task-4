"""
Evaluation Script for Triplet Similarity Model

Evaluates a trained model on a test/dev set.

Usage:
    python evaluate.py <test_data.jsonl> --checkpoint ./checkpoints/best_model.pt
    python evaluate.py ../data/dev/preprocessed_dev.jsonl --checkpoint ./checkpoints/best_model.pt
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
import json

# Import our modules
from data_loader import DRSDataset, Triplet
from feature_encoder import EventFeatureEncoder
from simple_encoder import TripletSimilarityModel, compute_triplet_loss, compute_accuracy


class Evaluator:
    """
    Evaluator for triplet similarity model.
    """

    def __init__(
            self,
            model: nn.Module,
            feature_encoder: EventFeatureEncoder,
            device: str = 'cpu',
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained TripletSimilarityModel
            feature_encoder: EventFeatureEncoder (must match training)
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_encoder = feature_encoder
        self.device = device

    @torch.no_grad()
    def evaluate_triplet(self, triplet: Triplet) -> Dict:
        """
        Evaluate a single triplet.

        Returns:
            Dict with prediction details
        """
        # Encode features
        encoded = self.feature_encoder.encode_triplet(triplet)

        # Move to device
        anchor = encoded['anchor'].to(self.device)
        story_a = encoded['story_a'].to(self.device)
        story_b = encoded['story_b'].to(self.device)

        # Forward pass
        output = self.model(anchor, story_a, story_b)

        # Get prediction
        logits = output['logits'].item()
        pred_a_closer = logits > 0
        label_a_closer = triplet.label
        correct = pred_a_closer == label_a_closer

        return {
            'idx': triplet.idx,
            'logits': logits,
            'sim_a': output['sim_a'].item(),
            'sim_b': output['sim_b'].item(),
            'pred_a_closer': pred_a_closer,
            'label_a_closer': label_a_closer,
            'correct': correct,
        }

    @torch.no_grad()
    def evaluate_dataset(
            self,
            dataset: DRSDataset,
            verbose: bool = True,
    ) -> Dict:
        """
        Evaluate on entire dataset.

        Returns:
            Dict with overall metrics and per-example results
        """
        results = []
        correct_count = 0
        total_count = 0

        for i, triplet in enumerate(dataset.triplets):
            result = self.evaluate_triplet(triplet)
            results.append(result)

            if result['correct']:
                correct_count += 1
            total_count += 1

            if verbose and (i + 1) % 100 == 0:
                print(f"  Evaluated {i + 1}/{len(dataset)} triplets...")

        accuracy = correct_count / total_count if total_count > 0 else 0

        # Compute additional metrics
        logits = [r['logits'] for r in results]
        labels = [1 if r['label_a_closer'] else 0 for r in results]

        # Confidence analysis
        confident_correct = sum(1 for r in results if r['correct'] and abs(r['logits']) > 0.1)
        confident_wrong = sum(1 for r in results if not r['correct'] and abs(r['logits']) > 0.1)
        uncertain = sum(1 for r in results if abs(r['logits']) <= 0.1)

        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': total_count,
            'results': results,
            'confident_correct': confident_correct,
            'confident_wrong': confident_wrong,
            'uncertain': uncertain,
            'mean_logits': np.mean(logits),
            'std_logits': np.std(logits),
        }


def load_model_and_encoder(
        checkpoint_path: str,
        train_data_path: str,
        device: str = 'cpu',
) -> Tuple[TripletSimilarityModel, EventFeatureEncoder]:
    """
    Load trained model and create matching feature encoder.

    Args:
        checkpoint_path: Path to model checkpoint
        train_data_path: Path to training data (for vocabulary)
        device: Device for model

    Returns:
        (model, feature_encoder)
    """
    # Load training data for vocabulary
    print(f"Loading vocabulary from training data: {train_data_path}")
    train_dataset = DRSDataset(train_data_path)
    feature_encoder = EventFeatureEncoder(dataset=train_dataset)

    # Create model with same architecture
    model = TripletSimilarityModel(
        input_dim=feature_encoder.feature_dim,
        hidden_dim=128,  # Must match training
        output_dim=64,  # Must match training
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val accuracy: {checkpoint.get('best_val_acc', 'unknown')}")

    return model, feature_encoder


def main():
    parser = argparse.ArgumentParser(description='Evaluate triplet similarity model')
    parser.add_argument('test_data', type=str, help='Path to test/dev preprocessed JSONL')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--train_data', type=str, default=None,
                        help='Path to training data (for vocabulary). If not provided, uses test_data.')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON file')
    parser.add_argument('--show_errors', action='store_true', help='Show misclassified examples')
    parser.add_argument('--num_errors', type=int, default=10, help='Number of errors to show')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Test data: {args.test_data}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model and encoder
    train_data = args.train_data if args.train_data else args.test_data
    model, feature_encoder = load_model_and_encoder(
        checkpoint_path=args.checkpoint,
        train_data_path=train_data,
        device=device,
    )

    # Load test data
    print(f"\nLoading test data: {args.test_data}")
    test_dataset = DRSDataset(args.test_data)

    # Create evaluator
    evaluator = Evaluator(model, feature_encoder, device)

    # Run evaluation
    print("\n" + "-" * 40)
    print("RUNNING EVALUATION")
    print("-" * 40)

    metrics = evaluator.evaluate_dataset(test_dataset, verbose=True)

    # Print results
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)

    print(f"\nAccuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Correct: {metrics['correct']} / {metrics['total']}")

    print(f"\nConfidence analysis:")
    print(f"  Confident & correct: {metrics['confident_correct']}")
    print(f"  Confident & wrong:   {metrics['confident_wrong']}")
    print(f"  Uncertain (|logit| <= 0.1): {metrics['uncertain']}")

    print(f"\nLogits statistics:")
    print(f"  Mean: {metrics['mean_logits']:.4f}")
    print(f"  Std:  {metrics['std_logits']:.4f}")

    # Show errors if requested
    if args.show_errors:
        print("\n" + "-" * 40)
        print(f"MISCLASSIFIED EXAMPLES (first {args.num_errors})")
        print("-" * 40)

        errors = [r for r in metrics['results'] if not r['correct']]

        for i, error in enumerate(errors[:args.num_errors]):
            triplet = test_dataset[error['idx']] if error['idx'] < len(test_dataset) else None

            print(f"\n[{i + 1}] Triplet {error['idx']}:")
            print(f"  Predicted: {'A closer' if error['pred_a_closer'] else 'B closer'}")
            print(f"  Actual:    {'A closer' if error['label_a_closer'] else 'B closer'}")
            print(f"  Logits: {error['logits']:.4f}")
            print(f"  Sim A: {error['sim_a']:.4f}, Sim B: {error['sim_b']:.4f}")

            if triplet:
                print(f"  Anchor events: {triplet.anchor.num_events}")
                print(f"  Story A events: {triplet.story_a.num_events}")
                print(f"  Story B events: {triplet.story_b.num_events}")

    # Save results if requested
    if args.output:
        output_data = {
            'accuracy': metrics['accuracy'],
            'correct': metrics['correct'],
            'total': metrics['total'],
            'confident_correct': metrics['confident_correct'],
            'confident_wrong': metrics['confident_wrong'],
            'uncertain': metrics['uncertain'],
            'mean_logits': metrics['mean_logits'],
            'std_logits': metrics['std_logits'],
            'checkpoint': args.checkpoint,
            'test_data': args.test_data,
            'per_example': metrics['results'],
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {args.output}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()