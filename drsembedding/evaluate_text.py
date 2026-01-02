"""
Evaluation Script with Text-Enhanced Feature Support

Evaluates a trained model on a test/dev set.
Automatically detects whether to use text-enhanced features based on checkpoint config.

Usage:
    python evaluate_text.py ../data/dev/preprocessed_dev.jsonl \
        --checkpoint ./checkpoints_text/best_model.pt \
        --train_data ../data/synthetic/preprocessed_synth.jsonl
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import argparse
from pathlib import Path
import json

# Import our modules
from data_loader import DRSDataset, Triplet
from feature_encoder import EventFeatureEncoder
from text_feature_encoder import TextEnhancedFeatureEncoder
from simple_encoder import TripletSimilarityModel, compute_triplet_loss


class Evaluator:
    """Evaluator for triplet similarity model."""

    def __init__(
            self,
            model: nn.Module,
            feature_encoder: Union[EventFeatureEncoder, TextEnhancedFeatureEncoder],
            device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.feature_encoder = feature_encoder
        self.device = device

    @torch.no_grad()
    def evaluate_triplet(self, triplet: Triplet) -> Dict:
        """Evaluate a single triplet."""
        encoded = self.feature_encoder.encode_triplet(triplet)

        anchor = encoded['anchor'].to(self.device)
        story_a = encoded['story_a'].to(self.device)
        story_b = encoded['story_b'].to(self.device)

        output = self.model(anchor, story_a, story_b)

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
    def evaluate_dataset(self, dataset: DRSDataset, verbose: bool = True) -> Dict:
        """Evaluate on entire dataset."""
        results = []
        correct_count = 0
        total_count = 0

        for i, triplet in enumerate(dataset.triplets):
            result = self.evaluate_triplet(triplet)
            results.append(result)

            if result['correct']:
                correct_count += 1
            total_count += 1

            if verbose and (i + 1) % 50 == 0:
                print(f"  Evaluated {i + 1}/{len(dataset)} triplets...")

        accuracy = correct_count / total_count if total_count > 0 else 0

        logits = [r['logits'] for r in results]

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
) -> Tuple[TripletSimilarityModel, Union[EventFeatureEncoder, TextEnhancedFeatureEncoder], Dict]:
    """
    Load trained model and create matching feature encoder.
    Auto-detects whether to use text features based on config.
    """
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'

    # Try to load config
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded config from {config_path}")
    else:
        print(f"⚠ No config.json found, using defaults")

    use_text = config.get('use_text', False)
    text_model = config.get('text_model', 'all-MiniLM-L6-v2')
    hidden_dim = config.get('hidden_dim', 128)
    output_dim = config.get('output_dim', 64)

    # Load training data for vocabulary
    print(f"Loading vocabulary from: {train_data_path}")
    train_dataset = DRSDataset(train_data_path)

    # Create appropriate encoder
    if use_text:
        print(f"Using TEXT-ENHANCED features with model: {text_model}")
        feature_encoder = TextEnhancedFeatureEncoder(
            dataset=train_dataset,
            model_name=text_model,
            device=device,
        )
    else:
        print(f"Using BASE features only")
        feature_encoder = EventFeatureEncoder(dataset=train_dataset)

    # Create model
    model = TripletSimilarityModel(
        input_dim=feature_encoder.feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}")

    return model, feature_encoder, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate triplet similarity model')
    parser.add_argument('test_data', type=str, help='Path to test/dev preprocessed JSONL')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON')
    parser.add_argument('--show_errors', action='store_true', help='Show misclassified examples')
    parser.add_argument('--num_errors', type=int, default=10, help='Number of errors to show')

    args = parser.parse_args()

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
    model, feature_encoder, config = load_model_and_encoder(
        checkpoint_path=args.checkpoint,
        train_data_path=args.train_data,
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

    print(f"\nFeature type: {'TEXT-ENHANCED' if config.get('use_text', False) else 'BASE'}")
    print(f"Input dimension: {feature_encoder.feature_dim}")

    print(f"\nAccuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Correct: {metrics['correct']} / {metrics['total']}")

    print(f"\nConfidence analysis:")
    print(f"  Confident & correct: {metrics['confident_correct']}")
    print(f"  Confident & wrong:   {metrics['confident_wrong']}")
    print(f"  Uncertain (|logit| <= 0.1): {metrics['uncertain']}")

    print(f"\nLogits statistics:")
    print(f"  Mean: {metrics['mean_logits']:.4f}")
    print(f"  Std:  {metrics['std_logits']:.4f}")

    # Show errors
    if args.show_errors:
        print("\n" + "-" * 40)
        print(f"MISCLASSIFIED EXAMPLES (first {args.num_errors})")
        print("-" * 40)

        errors = [r for r in metrics['results'] if not r['correct']]

        for i, error in enumerate(errors[:args.num_errors]):
            # Find the triplet by matching idx
            triplet = None
            for t in test_dataset.triplets:
                if t.idx == error['idx']:
                    triplet = t
                    break

            print(f"\n[{i + 1}] Triplet {error['idx']}:")
            print(f"  Predicted: {'A closer' if error['pred_a_closer'] else 'B closer'}")
            print(f"  Actual:    {'A closer' if error['label_a_closer'] else 'B closer'}")
            print(f"  Logits: {error['logits']:.4f}")
            print(f"  Sim A: {error['sim_a']:.4f}, Sim B: {error['sim_b']:.4f}")

            if triplet:
                print(f"  Anchor events: {triplet.anchor.num_events}")
                print(f"  Story A events: {triplet.story_a.num_events}")
                print(f"  Story B events: {triplet.story_b.num_events}")

    # Save results
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
            'config': config,
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