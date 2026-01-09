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

def generate_submission(
    model,
    test_file,
    output_file,
    vocab_file,
    text_model_name='sentence-transformers/all-MiniLM-L6-v2',
    device='cpu'
):
    """
    Generate submission file from test data.

    Args:
        model: Trained FiveComponentModel
        test_file: Path to preprocessed test JSONL file
        output_file: Path to output submission JSONL
        device: 'cpu' or 'cuda'
    """
    print(f"\n{'=' * 60}")
    print(f"GENERATING SUBMISSION")
    print(f"{'=' * 60}")
    print(f"Test file: {test_file}")
    print(f"Output: {output_file}")
    print(f"Device: {device}")


    # Load test data
    print(f"\nLoading test data...")
    drs_dataset = DRSDataset(test_file)
    processor = StoryDataProcessor(drs_dataset, text_model_name=text_model_name)
    print(f"✓ Loaded {len(drs_dataset)} test instances")

    print(f"Loading vocabularies from {vocab_file}...")
    with open(vocab_file, 'r') as f:
        vocabs = json.load(f)
    processor.verbnet_vocab = vocabs['verbnet_vocab']
    processor.predicate_vocab = vocabs['predicate_vocab']
    print(f"✓ Loaded vocabularies:")
    print(f"  VerbNet classes: {len(processor.verbnet_vocab)}")
    print(f"  Logic predicates: {len(processor.predicate_vocab)}")
    print(f"✓ Loaded {len(drs_dataset)} test instances")

    # Process in batches
    all_predictions = []

    model.eval()

    print(f"\nGenerating predictions...")

    for i in tqdm(range(len(drs_dataset)), desc="Predicting"):
        triplet = drs_dataset[i]

        # Process triplet (same as training)
        anchor_data, story_a_data, story_b_data = processor.process_triplet(triplet, device)

        # Get prediction (single instance)
        with torch.no_grad():
            output = model(anchor_data, story_a_data, story_b_data)
            logits = output['logits']  # Shape: [1, 2] or [2]

            # Get prediction: class 0 = A closer, class 1 = B closer
            pred = torch.argmax(logits, dim=-1)
            text_a_is_closer = (pred == 0).item()

        all_predictions.append(text_a_is_closer)
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
        '--text-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Text model for sentence embeddings (default: all-MiniLM-L6-v2)'
    )

    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to vocabularies.json file from training'
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
        vocab_file=args.vocab,
        device=device
    )

    print(f"\n✅ Submission ready: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Create a zip file containing track_a.jsonl:")
    print(f"   zip submission.zip {output_file}")
    print(f"2. Upload submission.zip to CodaBench")
    print(f"   https://www.codabench.org/competitions/10273/")


if __name__ == '__main__':
    main()