"""
Generate submission file for SemEval-2026 Task 4 (Track A)

This script creates the track_a.jsonl file required for CodaBench submission.
"""

import json
import os
from drs_text_baseline import HybridEvaluator


def create_submission_file(drs_dir, jsonl_path, output_path='track_a.jsonl', method='hybrid_simple'):
    """
    Create submission file for CodaBench.

    Args:
        drs_dir: Directory with DRS files
        jsonl_path: Path to dev data (for getting total instances)
        output_path: Where to save submission file
        method: Which method to use for predictions
            - 'hybrid_simple': Simple average of DRS + text (unsupervised)
            - 'hybrid_logistic': ML-based (requires training)
            - 'text_only': Just SBERT
            - 'drs_cosine': Just DRS cosine similarity

    Returns:
        Path to created submission file
    """
    print(f"Creating submission using method: {method}")
    print("=" * 60)

    # Create evaluator
    evaluator = HybridEvaluator(drs_dir, jsonl_path)

    predictions = []

    # Get predictions for each instance
    for idx in range(len(evaluator.data)):
        print(f"Processing instance {idx}/{len(evaluator.data)}", end='\r')

        if method == 'text_only':
            # Text-only prediction
            text_feat = evaluator.extract_text_features(idx)
            prediction = text_feat['sim_a'] > text_feat['sim_b']

        elif method == 'drs_cosine':
            # DRS-only prediction
            drs_feat = evaluator.extract_drs_features(idx)
            if drs_feat is None:
                # Fallback to text if DRS unavailable
                text_feat = evaluator.extract_text_features(idx)
                prediction = text_feat['sim_a'] > text_feat['sim_b']
            else:
                prediction = drs_feat['sims_a']['cosine'] > drs_feat['sims_b']['cosine']

        elif method == 'hybrid_simple':
            # Simple hybrid (unsupervised)
            hybrid_feat = evaluator.extract_hybrid_features(idx)

            if hybrid_feat is None:
                # Fallback to text if DRS unavailable
                text_feat = evaluator.extract_text_features(idx)
                prediction = text_feat['sim_a'] > text_feat['sim_b']
            else:
                drs_a = hybrid_feat['drs_features']['sims_a']['cosine']
                drs_b = hybrid_feat['drs_features']['sims_b']['cosine']
                text_a = hybrid_feat['text_features']['sim_a']
                text_b = hybrid_feat['text_features']['sim_b']

                combined_a = (drs_a + text_a) / 2
                combined_b = (drs_b + text_b) / 2

                prediction = combined_a > combined_b

        elif method == 'hybrid_logistic':
            # This would require trained model - placeholder
            raise NotImplementedError("hybrid_logistic requires trained model - use hybrid_simple instead")

        else:
            raise ValueError(f"Unknown method: {method}")

        predictions.append({
            'idx': idx,
            'prediction': bool(prediction)  # Ensure it's a boolean
        })

    print()  # New line after progress

    # Write submission file
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    print(f"\n✓ Submission file created: {output_path}")
    print(f"  Total instances: {len(predictions)}")
    print(f"  Predictions: {sum(p['prediction'] for p in predictions)} A-closer, "
          f"{len(predictions) - sum(p['prediction'] for p in predictions)} B-closer")

    return output_path


def create_submission_zip(track_a_file, output_zip='submission.zip'):
    """
    Create ZIP file for CodaBench submission.

    CodaBench expects: submission.zip containing track_a.jsonl at root level

    Args:
        track_a_file: Path to track_a.jsonl
        output_zip: Name of output zip file
    """
    import zipfile

    with zipfile.ZipFile(output_zip, 'w') as zipf:
        # Add file at root level (not in subdirectory)
        zipf.write(track_a_file, arcname='track_a.jsonl')

    print(f"\n✓ Submission ZIP created: {output_zip}")
    print(f"\n📤 Ready to upload to CodaBench!")

    return output_zip


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Create CodaBench submission')
    parser.add_argument('drs_dir', help='Directory with DRS files')
    parser.add_argument('jsonl_path', help='Path to dev_track_a.jsonl')
    parser.add_argument('--method', default='hybrid_simple',
                        choices=['text_only', 'drs_cosine', 'hybrid_simple'],
                        help='Prediction method to use')
    parser.add_argument('--output', default='track_a.jsonl',
                        help='Output file path')
    parser.add_argument('--zip', action='store_true',
                        help='Also create submission.zip')

    args = parser.parse_args()

    # Create submission file
    submission_file = create_submission_file(
        args.drs_dir,
        args.jsonl_path,
        args.output,
        args.method
    )

    # Optionally create ZIP
    if args.zip:
        create_submission_zip(submission_file)

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Go to: https://www.codabench.org/competitions/10273/")
    print("2. Click 'Participate' tab")
    print("3. Click 'Submit' button")
    print("4. Upload submission.zip" if args.zip else "4. Create ZIP: zip -j submission.zip track_a.jsonl")
    print("5. Wait for evaluation results!")
    print("=" * 60)