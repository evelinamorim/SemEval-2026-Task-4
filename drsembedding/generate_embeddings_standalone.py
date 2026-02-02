"""
Generate Track B Embeddings for SemEval-2026 Task 4 - Complete Standalone Version

This script loads a trained five-component model and generates embeddings
for the Track B test set. Works with text-only input (no DRS annotations needed).

Usage:
    python generate_embeddings_standalone.py \
        --checkpoint checkpoints_five \
        --test_data track_b_test.jsonl \
        --output track_b_embeddings \
        --create_zip

Example test data format:
    {"text": "Following the surrender of the great leader Geronimo..."}
    {"text": "Another story text here..."}
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List
import time
import sys

# Import model components
try:
    from five_component_encoder import (
        FiveComponentConfig,
        FiveComponentModel,
        StoryDataProcessor,
    )
    from data_loader import StoryGraph
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure five_component_encoder.py and data_loader.py are in the same directory")
    sys.exit(1)


class InferenceModel:
    """Wrapper for the five-component model for inference."""

    def __init__(self, model: FiveComponentModel, processor: StoryDataProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def encode_single_story(self, text: str) -> np.ndarray:
        """
        Encode a single story text into an embedding.

        Args:
            text: Story text

        Returns:
            embedding: numpy array of shape [output_dim]
        """
        # Create minimal story graph (empty DRS structure)
        story_graph = StoryGraph(
            events=[],
            actors=[],
            temporal_edges=[],
            coreference_edges=[],
            semantic_edges=[],
            event_types=[],
            verbnet_classes=[],
            logic_predicates=[],
        )

        # Process story
        story_data = self.processor.process_story(story_graph, text, self.device)

        # Extract embeddings from each component
        component_embeddings = []

        # Component 1: Temporal
        if self.model.temporal_encoder:
            temp_emb, _ = self.model.temporal_encoder(
                story_data['event_features'],
                story_data['temporal_adjacency'],
                story_data['temporal_edge_types'],
            )
            component_embeddings.append(temp_emb)

        # Component 2: Logical
        if self.model.logical_encoder:
            log_emb = self.model.logical_encoder(
                story_data['event_type_dist'],
                story_data['verbnet_indices'],
                story_data['predicate_multihot'],
            )
            component_embeddings.append(log_emb)

        # Component 3: Participant
        if self.model.participant_encoder:
            part_emb, _ = self.model.participant_encoder(
                story_data['actor_features'],
                story_data['coref_adjacency'],
            )
            component_embeddings.append(part_emb)

        # Component 4: Semantic Role
        if self.model.semantic_encoder:
            # Get hidden representations
            event_hidden = self.model.temporal_encoder.node_encoder(story_data['event_features']) \
                if self.model.temporal_encoder else None
            actor_hidden = self.model.participant_encoder.node_encoder(story_data['actor_features']) \
                if self.model.participant_encoder else None

            if event_hidden is not None and actor_hidden is not None:
                sem_emb, _ = self.model.semantic_encoder(
                    event_hidden,
                    actor_hidden,
                    story_data['semantic_edges'],
                    story_data['event_id_to_idx'],
                    story_data['actor_id_to_idx'],
                )
                component_embeddings.append(sem_emb)

        # Component 5: Text (most important when we don't have DRS)
        if self.model.text_encoder:
            text_emb = self.model.text_encoder(story_data['text'])
            component_embeddings.append(text_emb)

        # Fuse all components
        if component_embeddings:
            fused = torch.cat(component_embeddings, dim=-1)
            embedding = self.model.fusion_layer(fused)
        else:
            # Fallback if no components enabled
            embedding = torch.zeros(self.model.config.output_dim, device=self.device)

        # Convert to numpy
        return embedding.cpu().numpy()


def load_model_and_config(checkpoint_dir: str, device: str = 'cpu') -> InferenceModel:
    """
    Load trained model, config, and vocabularies.

    Returns:
        InferenceModel wrapper
    """
    checkpoint_dir = Path(checkpoint_dir)

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    # Load config
    config_path = checkpoint_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    print(f"\n✓ Loaded config from {config_path}")
    print(f"  Output dimension: {config_dict['output_dim']}")
    print(f"  Components: T={config_dict['use_temporal']}, "
          f"L={config_dict['use_logical']}, "
          f"P={config_dict['use_participant']}, "
          f"S={config_dict['use_semantic']}, "
          f"Txt={config_dict['use_text']}")

    # Create config object
    config = FiveComponentConfig(
        temporal_dim=config_dict['temporal_dim'],
        logical_dim=config_dict['logical_dim'],
        participant_dim=config_dict['participant_dim'],
        semantic_dim=config_dict['semantic_dim'],
        text_dim=config_dict['text_dim'],
        hidden_dim=config_dict['hidden_dim'],
        output_dim=config_dict['output_dim'],
        dropout=config_dict['dropout'],
        text_model_name=config_dict['text_model_name'],
        num_verbnet_classes=config_dict['num_verbnet_classes'],
        num_predicates=config_dict['num_predicates'],
        num_gnn_layers=config_dict.get('num_gnn_layers', 2),
        use_temporal=config_dict['use_temporal'],
        use_logical=config_dict['use_logical'],
        use_participant=config_dict['use_participant'],
        use_semantic=config_dict['use_semantic'],
        use_text=config_dict['use_text'],
    )

    # Load vocabularies
    vocab_path = checkpoint_dir / 'vocabularies.json'
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabularies not found: {vocab_path}")

    with open(vocab_path, 'r') as f:
        vocabs = json.load(f)

    print(f"✓ Loaded vocabularies from {vocab_path}")
    print(f"  VerbNet classes: {len(vocabs['verbnet_vocab'])}")
    print(f"  Predicates: {len(vocabs['predicate_vocab'])}")

    # Create processor with loaded vocabularies
    processor = StoryDataProcessor(
        dataset=None,
        text_model_name=config.text_model_name,
    )
    processor.verbnet_vocab = vocabs['verbnet_vocab']
    processor.predicate_vocab = vocabs['predicate_vocab']

    # Create model
    model = FiveComponentModel(config, device)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = checkpoint_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best validation accuracy: {checkpoint['best_val_acc'] * 100:.2f}%")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return InferenceModel(model, processor, device)


def load_test_data(test_path: str) -> List[str]:
    """
    Load test data from JSONL file.

    Expected format:
        {"text": "story text"}

    Returns:
        List of story texts
    """
    test_path = Path(test_path)

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    stories = []
    with open(test_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'text' not in data:
                        raise ValueError(f"Line {line_num}: Missing 'text' field")
                    stories.append(data['text'])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_num}: Invalid JSON - {e}")

    print(f"\n✓ Loaded {len(stories)} test stories from {test_path}")

    return stories


def generate_embeddings(
        inference_model: InferenceModel,
        stories: List[str],
) -> np.ndarray:
    """
    Generate embeddings for all test stories.

    Returns:
        embeddings: numpy array of shape [num_stories, embedding_dim]
    """
    print("\n" + "=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)
    print(f"Processing {len(stories)} stories...")

    all_embeddings = []
    start_time = time.time()

    for i, text in enumerate(stories):
        # Generate embedding
        embedding = inference_model.encode_single_story(text)
        all_embeddings.append(embedding)

        # Progress update
        if (i + 1) % 100 == 0 or (i + 1) == len(stories):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(stories) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i + 1}/{len(stories)}] {rate:.1f} stories/s, ETA: {eta:.0f}s")

    # Stack into single array
    embeddings = np.stack(all_embeddings, axis=0)

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")

    return embeddings


def save_embeddings_npy(embeddings: np.ndarray, output_path: str):
    """Save embeddings as numpy file."""
    output_path = Path(output_path).with_suffix('.npy')
    np.save(output_path, embeddings)
    print(f"\n✓ Saved embeddings to {output_path}")
    print(f"  Format: NumPy (.npy)")
    return output_path


def save_embeddings_jsonl(embeddings: np.ndarray, output_path: str):
    """Save embeddings as JSONL file."""
    output_path = Path(output_path).with_suffix('.jsonl')

    with open(output_path, 'w') as f:
        for emb in embeddings:
            json_obj = {"embeddings": emb.tolist()}
            f.write(json.dumps(json_obj) + '\n')

    print(f"\n✓ Saved embeddings to {output_path}")
    print(f"  Format: JSONL")
    return output_path


def create_submission_zip(
        embeddings: np.ndarray,
        output_dir: str,
        format: str = 'npy',
):
    """
    Create a submission-ready ZIP file.

    The ZIP should contain:
        track_b.npy  OR  track_b.jsonl
    """
    import zipfile
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory
    temp_dir = output_dir / 'temp_submission'
    temp_dir.mkdir(exist_ok=True)

    # Create embeddings file
    if format == 'npy':
        emb_path = temp_dir / 'track_b.npy'
        np.save(emb_path, embeddings)
    else:  # jsonl
        emb_path = temp_dir / 'track_b.jsonl'
        with open(emb_path, 'w') as f:
            for emb in embeddings:
                json_obj = {"embeddings": emb.tolist()}
                f.write(json.dumps(json_obj) + '\n')

    # Create ZIP
    zip_path = output_dir / 'track_b_submission.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(emb_path, emb_path.name)

    # Clean up temp directory
    shutil.rmtree(temp_dir)

    print(f"\n✓ Created submission ZIP: {zip_path}")
    print(f"  Contents: {emb_path.name}")
    print(f"  File size: {zip_path.stat().st_size / 1024:.1f} KB")
    print(f"\n  Ready to upload to CodaBench!")

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate Track B embeddings for SemEval-2026 Task 4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings and create submission ZIP
  python generate_embeddings_standalone.py \\
      --checkpoint checkpoints_five \\
      --test_data track_b_test.jsonl \\
      --output track_b_embeddings \\
      --create_zip

  # Generate both formats
  python generate_embeddings_standalone.py \\
      --checkpoint checkpoints_five \\
      --test_data track_b_test.jsonl \\
      --format both
        """
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint directory (contains best_model.pt, config.json, vocabularies.json)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test JSONL file (one story per line: {"text": "..."})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='track_b_embeddings',
        help='Output path (without extension, default: track_b_embeddings)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['npy', 'jsonl', 'both'],
        default='npy',
        help='Output format: npy (recommended), jsonl, or both (default: npy)'
    )
    parser.add_argument(
        '--create_zip',
        action='store_true',
        help='Create submission-ready ZIP file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device: cuda, cpu, or auto (default: auto)'
    )

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("\n" + "=" * 70)
    print("TRACK B EMBEDDING GENERATION FOR SEMEVAL-2026 TASK 4")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")

    try:
        # Load model
        inference_model = load_model_and_config(args.checkpoint, device)

        # Load test data
        test_stories = load_test_data(args.test_data)

        # Generate embeddings
        embeddings = generate_embeddings(inference_model, test_stories)

        # Check embedding dimension constraints
        emb_dim = embeddings.shape[1]
        if not (10 <= emb_dim <= 8192):
            print(f"\n⚠ WARNING: Embedding dimension {emb_dim} is outside allowed range [10, 8192]")
        else:
            print(f"\n✓ Embedding dimension {emb_dim} is within allowed range [10, 8192]")

        # Save embeddings
        print("\n" + "=" * 70)
        print("SAVING EMBEDDINGS")
        print("=" * 70)

        saved_paths = []
        if args.format in ['npy', 'both']:
            path = save_embeddings_npy(embeddings, args.output)
            saved_paths.append(str(path))

        if args.format in ['jsonl', 'both']:
            path = save_embeddings_jsonl(embeddings, args.output)
            saved_paths.append(str(path))

        # Create submission ZIP if requested
        if args.create_zip:
            print("\n" + "=" * 70)
            print("CREATING SUBMISSION ZIP")
            print("=" * 70)

            output_dir = Path(args.output).parent if Path(args.output).parent != Path('.') else Path('.')
            zip_format = 'npy' if args.format in ['npy', 'both'] else 'jsonl'

            zip_path = create_submission_zip(
                embeddings=embeddings,
                output_dir=output_dir,
                format=zip_format,
            )

        # Summary
        print("\n" + "=" * 70)
        print("✓ COMPLETE")
        print("=" * 70)
        print(f"\nGenerated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Output files: {', '.join(saved_paths)}")

        if args.create_zip:
            print(f"\n📦 Submission file: {zip_path}")
            print(f"\n🚀 Next steps:")
            print(f"   1. Go to https://www.codabench.org/competitions/10273/")
            print(f"   2. Upload {zip_path.name}")
            print(f"   3. Select 'Track B: Embeddings' in the upload form")
            print(f"   4. Specify your method type")
        else:
            print(f"\n💡 To create submission ZIP, run again with --create_zip flag")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()