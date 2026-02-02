"""
Generate Track B Embeddings from Preprocessed DRS Data

This script loads preprocessed Track B data and generates embeddings.

Usage:
    # Step 1: Preprocess the data
    python preprocess_track_b.py \
        --test_json track_b_test.jsonl \
        --drs_dir track_b_drs \
        --output track_b_preprocessed.jsonl

    # Step 2: Generate embeddings
    python generate_embeddings_drs.py \
        --checkpoint checkpoints_five \
        --preprocessed track_b_preprocessed.jsonl \
        --output track_b_embeddings \
        --create_zip
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
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


def load_preprocessed_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load preprocessed Track B data."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item)
    return data


def item_to_story_graph(item: Dict[str, Any]) -> StoryGraph:
    """Convert preprocessed item to StoryGraph object."""
    return StoryGraph(
        events=item.get('events', []),
        actors=item.get('actors', []),
        temporal_edges=item.get('temporal_edges', []),
        coreference_edges=item.get('coreference_edges', []),
        semantic_edges=item.get('semantic_edges', []),
        event_types=item.get('event_types', {}),
        verbnet_classes=item.get('verbnet_classes', {}),
        logic_predicates=item.get('logic_predicates', []),
    )


class InferenceModel:
    """Wrapper for the five-component model for inference."""

    def __init__(self, model: FiveComponentModel, processor: StoryDataProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def encode_single_story(self, text: str, story_graph: StoryGraph) -> np.ndarray:
        """
        Encode a single story using the same logic as model.encode_story().
        """
        # Process story data
        story_data = self.processor.process_story(story_graph, text, self.device)

        components = []
        event_hidden = None
        actor_hidden = None

        # Component 1: Temporal
        if self.model.temporal_encoder is not None:
            temp_emb, event_hidden = self.model.temporal_encoder(
                story_data['event_features'],
                story_data['temporal_adjacency'],
                story_data['temporal_edge_types'],
            )
            components.append(temp_emb)

        # Component 2: Logical
        if self.model.logical_encoder is not None:
            log_emb = self.model.logical_encoder(
                story_data['event_type_dist'],
                story_data['verbnet_indices'],
                story_data['predicate_multihot'],
            )
            components.append(log_emb)

        # Component 3: Participant
        if self.model.participant_encoder is not None:
            part_emb, actor_hidden = self.model.participant_encoder(
                story_data['actor_features'],
                story_data['coref_adjacency'],
            )
            components.append(part_emb)

        # Component 4: Semantic
        if self.model.semantic_encoder is not None and event_hidden is not None and actor_hidden is not None:
            # Only compute if we have both event and actor representations
            if event_hidden.size(0) > 0 and actor_hidden.size(0) > 0:
                sem_emb, _ = self.model.semantic_encoder(
                    event_hidden,
                    actor_hidden,
                    story_data['semantic_edges'],
                    story_data['event_id_to_idx'],
                    story_data['actor_id_to_idx'],
                )
                components.append(sem_emb)

        # Component 5: Text
        if self.model.text_encoder is not None:
            text_emb = self.model.text_encoder(story_data['text'])
            components.append(text_emb)

        # Apply learnable component weights (same as encode_story)
        weights = F.softmax(self.model.component_weights, dim=0)
        weighted_components = []
        for i, comp in enumerate(components):
            # Clone inference tensors (e.g., from frozen SBERT)
            if comp.is_inference():
                comp = comp.clone()
            weighted_components.append(comp * weights[i])

        # Concatenate and fuse
        combined = torch.cat(weighted_components, dim=-1)
        embedding = self.model.fusion(combined)

        return embedding.cpu().numpy()


def load_model_and_config(checkpoint_dir: str, device: str = 'cpu') -> InferenceModel:
    """Load trained model, config, and vocabularies."""
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
    checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    if not checkpoint_files:
        # Try best_model.pt
        best_model_path = checkpoint_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint_path = best_model_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    else:
        # Use latest checkpoint
        checkpoint_path = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))

    print(f"\n✓ Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")

    return InferenceModel(model, processor, device)


def generate_embeddings(
    inference_model: InferenceModel,
    preprocessed_data: List[Dict[str, Any]]
) -> np.ndarray:
    """Generate embeddings for all preprocessed stories."""
    print("\n" + "=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)
    print(f"Processing {len(preprocessed_data)} stories...")

    all_embeddings = []
    start_time = time.time()

    with_drs = 0
    without_drs = 0

    for i, item in enumerate(preprocessed_data):
        text = item.get('text', '')
        story_graph = item_to_story_graph(item)

        # Track stats
        if item.get('has_drs', False) or len(item.get('events', [])) > 0:
            with_drs += 1
        else:
            without_drs += 1

        # Generate embedding
        embedding = inference_model.encode_single_story(text, story_graph)
        all_embeddings.append(embedding)

        # Progress update
        if (i + 1) % 100 == 0 or (i + 1) == len(preprocessed_data):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(preprocessed_data) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i + 1}/{len(preprocessed_data)}] {rate:.1f} stories/s, ETA: {eta:.0f}s")

    # Stack into single array
    embeddings = np.stack(all_embeddings, axis=0)

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
    print(f"\n  Statistics:")
    print(f"    - With DRS: {with_drs}")
    print(f"    - Text-only: {without_drs}")

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
    """Create submission-ready ZIP file."""
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
        description='Generate Track B embeddings from preprocessed DRS data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Preprocess (run preprocess_track_b.py first)
  python preprocess_track_b.py \\
      --test_json track_b_test.jsonl \\
      --drs_dir track_b_drs \\
      --output track_b_preprocessed.jsonl

  # Step 2: Generate embeddings
  python generate_embeddings_drs.py \\
      --checkpoint checkpoints_five \\
      --preprocessed track_b_preprocessed.jsonl \\
      --output track_b_embeddings \\
      --create_zip
        """
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--preprocessed',
        type=str,
        required=True,
        help='Path to preprocessed JSONL file (from preprocess_track_b.py)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='track_b_embeddings',
        help='Output path (without extension)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['npy', 'jsonl', 'both'],
        default='npy',
        help='Output format (default: npy)'
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
    print("TRACK B EMBEDDING GENERATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Preprocessed data: {args.preprocessed}")

    try:
        # Load model
        inference_model = load_model_and_config(args.checkpoint, device)

        # Load preprocessed data
        print("\n" + "=" * 70)
        print("LOADING PREPROCESSED DATA")
        print("=" * 70)
        preprocessed_data = load_preprocessed_data(args.preprocessed)
        print(f"✓ Loaded {len(preprocessed_data)} stories")

        # Generate embeddings
        embeddings = generate_embeddings(inference_model, preprocessed_data)

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
            print(f"   3. Select 'Track B: Embeddings'")
            print(f"   4. Specify your method type")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()