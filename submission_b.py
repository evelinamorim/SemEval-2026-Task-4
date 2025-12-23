"""
Track B Submission Generator for SemEval-2026 Task 4

Track B: Create embeddings for individual stories (not triplets!)
Input: dev_track_b.jsonl with 480 individual stories
Output: Embeddings file for submission
"""

import json
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from drs_parser import parse_drs_file


def load_track_b_data(jsonl_path):
    """Load Track B data (individual stories)."""
    stories = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            stories.append(json.loads(line))
    return stories


def create_drs_embeddings_track_b(drs_dir, jsonl_path, output_path='track_b_drs_embeddings.pkl'):
    """
    Create DRS embeddings for Track B.

    Args:
        drs_dir: Directory containing DRS files (named 0_drs.txt, 1_drs.txt, ...)
        jsonl_path: Path to dev_track_b.jsonl
        output_path: Where to save embeddings

    Returns:
        dict: {story_idx: embedding_vector}
    """
    print("=" * 60)
    print("CREATING DRS EMBEDDINGS FOR TRACK B")
    print("=" * 60)

    stories = load_track_b_data(jsonl_path)
    print(f"Total stories: {len(stories)}")

    embeddings = {}
    missing_drs = []

    for idx in range(len(stories)):
        drs_file = os.path.join(drs_dir, f"{idx}_drs.txt")

        if not os.path.exists(drs_file):
            missing_drs.append(idx)
            print(f"Warning: DRS file not found: {drs_file}")
            # Create zero vector as placeholder
            embeddings[idx] = np.zeros(30, dtype=np.float32)
            continue

        try:
            parser = parse_drs_file(drs_file)
            features = parser.get_feature_vector()

            # Convert to numpy array (values only, in consistent order)
            feature_keys = sorted(features.keys())
            embedding = np.array([features[k] for k in feature_keys], dtype=np.float32)

            embeddings[idx] = embedding

            if (idx + 1) % 50 == 0:
                print(f"Processed: {idx + 1}/{len(stories)}")

        except Exception as e:
            print(f"Error processing {drs_file}: {e}")
            embeddings[idx] = np.zeros(30, dtype=np.float32)
            missing_drs.append(idx)

    # Save embeddings
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"\n✓ DRS embeddings saved: {output_path}")
    print(f"  Successfully processed: {len(embeddings) - len(missing_drs)}/{len(stories)}")
    print(f"  Missing DRS files: {len(missing_drs)}")
    print(f"  Embedding dimension: {embeddings[0].shape[0]}")

    if missing_drs:
        print(f"\nWarning: Missing DRS for indices: {missing_drs[:10]}{'...' if len(missing_drs) > 10 else ''}")

    return embeddings


def create_text_embeddings_track_b(jsonl_path, output_path='track_b_text_embeddings.pkl',
                                   model_name='all-MiniLM-L6-v2'):
    """
    Create text embeddings for Track B using SBERT.

    Args:
        jsonl_path: Path to dev_track_b.jsonl
        output_path: Where to save embeddings
        model_name: Sentence transformer model to use

    Returns:
        dict: {story_idx: embedding_vector}
    """
    print("=" * 60)
    print("CREATING TEXT EMBEDDINGS FOR TRACK B")
    print("=" * 60)

    stories = load_track_b_data(jsonl_path)
    print(f"Total stories: {len(stories)}")
    print(f"Model: {model_name}")

    # Load model
    model = SentenceTransformer(model_name)

    # Extract texts
    texts = [story['text'] for story in stories]

    # Batch encode
    print("Encoding stories...")
    embeddings_array = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Create dictionary
    embeddings = {
        idx: emb.astype(np.float32)
        for idx, emb in enumerate(embeddings_array)
    }

    # Save embeddings
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"\n✓ Text embeddings saved: {output_path}")
    print(f"  Total stories: {len(embeddings)}")
    print(f"  Embedding dimension: {embeddings[0].shape[0]}")

    return embeddings


def create_hybrid_embeddings_track_b(drs_dir, jsonl_path, output_path='track_b_hybrid_embeddings.pkl',
                                     model_name='all-MiniLM-L6-v2'):
    """
    Create hybrid (DRS + Text) embeddings for Track B.

    Args:
        drs_dir: Directory with DRS files
        jsonl_path: Path to dev_track_b.jsonl
        output_path: Where to save embeddings

    Returns:
        dict: {story_idx: embedding_vector}
    """
    print("=" * 60)
    print("CREATING HYBRID EMBEDDINGS FOR TRACK B")
    print("=" * 60)

    stories = load_track_b_data(jsonl_path)
    print(f"Total stories: {len(stories)}")

    # Get text embeddings
    print("\n1. Creating text embeddings...")
    model = SentenceTransformer(model_name)
    texts = [story['text'] for story in stories]
    text_embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Get DRS embeddings
    print("\n2. Creating DRS embeddings...")
    embeddings = {}
    missing_drs = []

    for idx in range(len(stories)):
        drs_file = os.path.join(drs_dir, f"{idx}_drs.txt")

        # Get text embedding
        text_emb = text_embeddings[idx]

        # Get DRS embedding
        if os.path.exists(drs_file):
            try:
                parser = parse_drs_file(drs_file)
                features = parser.get_feature_vector()

                feature_keys = sorted(features.keys())
                drs_emb = np.array([features[k] for k in feature_keys], dtype=np.float32)
            except Exception as e:
                print(f"Error processing {drs_file}: {e}")
                drs_emb = np.zeros(30, dtype=np.float32)
                missing_drs.append(idx)
        else:
            drs_emb = np.zeros(30, dtype=np.float32)
            missing_drs.append(idx)

        # Concatenate DRS + Text
        hybrid_emb = np.concatenate([drs_emb, text_emb]).astype(np.float32)
        embeddings[idx] = hybrid_emb

        if (idx + 1) % 50 == 0:
            print(f"Processed: {idx + 1}/{len(stories)}")

    # Save embeddings
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"\n✓ Hybrid embeddings saved: {output_path}")
    print(f"  Successfully processed: {len(embeddings) - len(missing_drs)}/{len(stories)}")
    print(f"  Missing DRS files: {len(missing_drs)}")
    print(f"  Embedding dimension: {embeddings[0].shape[0]} (DRS: 30 + Text: {text_embeddings.shape[1]})")

    return embeddings


def convert_to_submission_format(embeddings, output_file='track_b.pkl'):
    """
    Convert embeddings to submission format.

    Check organizers' baseline for exact format requirements!
    Common formats:
    - Pickle dictionary: {idx: embedding}
    - NumPy array: (n_stories, embedding_dim)
    - NPZ file: np.savez('embeddings.npz', embeddings=array)
    """
    # Save as pickle (most flexible)
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)

    # Also save as numpy array (common format)
    n_stories = len(embeddings)
    embedding_dim = embeddings[0].shape[0]
    embeddings_array = np.zeros((n_stories, embedding_dim), dtype=np.float32)

    for idx, emb in embeddings.items():
        embeddings_array[idx] = emb

    np.save(output_file.replace('.pkl', '.npy'), embeddings_array)

    print(f"\n✓ Submission files created:")
    print(f"  - {output_file} (pickle)")
    print(f"  - {output_file.replace('.pkl', '.npy')} (numpy)")

    return output_file


def create_submission_zip(embedding_file, output_zip='track_b_submission.zip'):
    """Create ZIP file for Track B submission."""
    import zipfile

    with zipfile.ZipFile(output_zip, 'w') as zipf:
        # Check organizers' requirements for exact filename!
        zipf.write(embedding_file, arcname=os.path.basename(embedding_file))

    print(f"\n✓ Submission ZIP created: {output_zip}")
    print(f"\n📤 Ready to upload to CodaBench Track B!")

    return output_zip


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create Track B embeddings')
    parser.add_argument('jsonl_path', help='Path to dev_track_b.jsonl')
    parser.add_argument('--drs_dir', help='Directory with DRS files (for DRS/hybrid methods)')
    parser.add_argument('--method', default='text',
                        choices=['drs', 'text', 'hybrid'],
                        help='Embedding method')
    parser.add_argument('--output', default='track_b',
                        help='Output filename prefix')
    parser.add_argument('--zip', action='store_true',
                        help='Create submission ZIP')

    args = parser.parse_args()

    # Create embeddings based on method
    if args.method == 'drs':
        if not args.drs_dir:
            print("Error: --drs_dir required for DRS embeddings")
            exit(1)
        embeddings = create_drs_embeddings_track_b(
            args.drs_dir,
            args.jsonl_path,
            f"{args.output}_drs_embeddings.pkl"
        )

    elif args.method == 'text':
        embeddings = create_text_embeddings_track_b(
            args.jsonl_path,
            f"{args.output}_text_embeddings.pkl"
        )

    else:  # hybrid
        if not args.drs_dir:
            print("Error: --drs_dir required for hybrid embeddings")
            exit(1)
        embeddings = create_hybrid_embeddings_track_b(
            args.drs_dir,
            args.jsonl_path,
            f"{args.output}_hybrid_embeddings.pkl"
        )

    # Convert to submission format
    submission_file = convert_to_submission_format(
        embeddings,
        f"{args.output}_{args.method}.pkl"
    )

    # Optionally create ZIP
    if args.zip:
        create_submission_zip(submission_file)

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Check organizers' Track B baseline for exact format requirements:")
    print("   https://github.com/narrative-similarity-task/semeval-2026-task-4-baselines/blob/main/track_b.py")
    print("2. Adjust format if needed")
    print("3. Create submission ZIP")
    print("4. Upload to CodaBench Track B")
    print("=" * 60)