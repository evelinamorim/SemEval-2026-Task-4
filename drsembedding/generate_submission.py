#!/usr/bin/env python3
"""
Generate Track A submission for SemEval-2026 Task 4
Reads test data and produces predictions using trained model checkpoint.
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from five_component_encoder import (
    FiveComponentModel,
    FiveComponentConfig
)
from train_five import StoryDataProcessor
from data_loader import DRSDataset


def load_model(checkpoint_path, config_path, device='cpu'):
    """Load trained model from checkpoint."""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = FiveComponentConfig(**config_dict)

    print(f"Initializing model...")
    model = FiveComponentModel(config, device=device)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint.get('val_accuracy', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"✓ Loaded model from epoch {epoch}, val accuracy: {best_acc}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model state dict")

    model.eval()
    return model, config


def predict_batch(model, batch_data, device='cpu'):
    """Make predictions for a batch of triplets."""
    with torch.no_grad():
        outputs = model(
            batch_data['anchor'],
            batch_data['story_a'],
            batch_data['story_b']
        )
        logits = outputs['logits']  # Shape: [batch_size, 2]

        # Get predictions: class 0 = A closer, class 1 = B closer
        predictions = torch.argmax(logits, dim=1)

        # Convert to boolean: True if A is closer (class 0)
        text_a_is_closer = (predictions == 0).cpu().tolist()

    return text_a_is_closer

def generate_submission(
    model,
    test_file,
    output_file,
    text_model_name='sentence-transformers/all-MiniLM-L6-v2',
    device='cpu',
    batch_size=8
):
    """
    Generate submission file from test data.

    Args:
        model: Trained FiveComponentModel
        test_file: Path to preprocessed test JSONL file
        output_file: Path to output submission JSONL
        device: 'cpu' or 'cuda'
        batch_size: Number of instances per batch
    """
    print(f"\n{'=' * 60}")
    print(f"GENERATING SUBMISSION")
    print(f"{'=' * 60}")
    print(f"Test file: {test_file}")
    print(f"Output: {output_file}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")

    # Load test data
    print(f"\nLoading test data...")
    drs_dataset = DRSDataset(test_file)
    processor = StoryDataProcessor(drs_dataset, text_model_name=text_model_name)
    print(f"✓ Loaded {len(drs_dataset)} test instances")

    # Process in batches
    all_predictions = []
    num_batches = (len(drs_dataset) + batch_size - 1) // batch_size

    model.eval()

    print(f"\nGenerating predictions...")
    num_batches = (len(drs_dataset) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(drs_dataset), batch_size), total=num_batches):
        batch_indices = list(range(i, min(i + batch_size, len(drs_dataset))))

        # Get batch data using processor
        batch_data = processor.collate_fn([processor[idx] for idx in batch_indices])

        # Move to device
        batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch_data.items()}

        # Get predictions
        predictions = predict_batch(model, batch_data, device)
        all_predictions.extend(predictions)

    # Write submission file
    print(f"\nWriting submission to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in all_predictions:
            json.dump({"text_a_is_closer": pred}, f)
            f.write('\n')

    print(f"✓ Wrote {len(all_predictions)} predictions")
    print(f"\n{'=' * 60}")
    print(f"SUBMISSION GENERATED SUCCESSFULLY")
    print(f"{'=' * 60}")

    # Show distribution
    num_a = sum(all_predictions)
    num_b = len(all_predictions) - num_a
    print(f"\nPrediction distribution:")
    print(f"  Story A closer: {num_a} ({100 * num_a / len(all_predictions):.1f}%)")
    print(f"  Story B closer: {num_b} ({100 * num_b / len(all_predictions):.1f}%)")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate Track A submission for SemEval-2026 Task 4'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config (.json file)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        required=True,
        help='Path to preprocessed test data (.jsonl file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='track_a.jsonl',
        help='Output submission file (default: track_a.jsonl)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: cpu)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for inference (default: 8)'
    )

    parser.add_argument(
        '--text-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Text model for sentence embeddings (default: all-MiniLM-L6-v2)'
    )

    args = parser.parse_args()

    # Validate inputs
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    test_file = Path(args.test_file)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # Load model
    model, config = load_model(checkpoint_path, config_path, device)
    model = model.to(device)

    # Generate submission
    output_file = generate_submission(
        model=model,
        test_file=test_file,
        output_file=args.output,
        text_model_name=args.text_model,
        device=device,
        batch_size=args.batch_size
    )

    print(f"\n✅ Submission ready: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Create a zip file containing track_a.jsonl:")
    print(f"   zip submission.zip {output_file}")
    print(f"2. Upload submission.zip to CodaBench")
    print(f"   https://www.codabench.org/competitions/10273/")


if __name__ == '__main__':
    main()