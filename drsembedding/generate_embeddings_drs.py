"""
Generate Track B Embeddings with DRS Annotations

This script loads test data from:
1. JSONL file with story texts (one per line)
2. Directory with DRS files (named {line_number}_drs.txt)

Usage:
    python generate_embeddings_with_drs.py \
        --checkpoint checkpoints_five \
        --test_json track_b_test.jsonl \
        --drs_dir track_b_drs \
        --output track_b_embeddings \
        --create_zip

Directory structure expected:
    track_b_test.jsonl          # Line 0: {"text": "..."}
                                # Line 1: {"text": "..."}
    track_b_drs/
        0_drs.txt              # DRS for line 0
        1_drs.txt              # DRS for line 1
        2_drs.txt              # DRS for line 2
        ...
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import time
import sys

# Import model components
try:
    from five_component_encoder import (
        FiveComponentConfig,
        FiveComponentModel,
        StoryDataProcessor,
    )
    from data_loader import StoryGraph, DRSDataset
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure five_component_encoder.py and data_loader.py are in the same directory")
    sys.exit(1)


def load_drs_from_file(drs_path: Path) -> Optional[StoryGraph]:
    """
    Load a DRS file and parse it into a StoryGraph.

    Args:
        drs_path: Path to the DRS file

    Returns:
        StoryGraph object or None if parsing fails
    """
    if not drs_path.exists():
        print(f"Warning: DRS file not found: {drs_path}")
        return None

    try:
        with open(drs_path, 'r', encoding='utf-8') as f:
            drs_content = f.read()

        # Parse DRS using the same logic as in data_loader.py
        # You'll need to use whatever parsing function you have
        # This is a placeholder - replace with your actual DRS parsing
        story_graph = parse_drs_content(drs_content)

        return story_graph
    except Exception as e:
        print(f"Error parsing DRS file {drs_path}: {e}")
        return None


def parse_drs_content(drs_text: str) -> StoryGraph:
    """
    Parse DRS text format into StoryGraph.

    This function needs to be customized based on your DRS format.

    IMPORTANT: Replace this with your actual DRS parsing logic!

    The DRS file should contain structured information about:
    - Events (with types, tenses, aspects, lemmas, positions)
    - Actors/Participants (with coreference information)
    - Temporal relations (occursBefore, occursAfter, overlaps, during)
    - Semantic roles (agent, patient, theme, etc.)
    - VerbNet classes
    - Predicates

    Expected output format (StoryGraph):
        events: List[Dict] with:
            - id: str (e.g., "e1", "e2")
            - type: str ("State", "Process", "Transition", "Unknown")
            - text: str (original text)
            - lemma: str (lemmatized form)
            - tense: str ("Past", "Present", "Future", "Unknown")
            - aspect: str ("Perfective", "Progressive", "Unknown")
            - vform: str ("Infinitive", "Participle", "Unknown")
            - position: float (0.0 to 1.0, relative position in text)
            - verbnet_class: str (optional, e.g., "run-51.3.2")
            - predicates: List[str] (optional, e.g., ["motion", "manner"])

        actors: List[Dict] with:
            - id: str (e.g., "x1", "x2")
            - text: str (mention text)
            - is_event_reference: bool
            - position: float (0.0 to 1.0)

        temporal_edges: List[Dict] with:
            - source: str (event id)
            - target: str (event id)
            - relation: str ("occursBefore", "occursAfter", "overlaps", "during")

        coreference_edges: List[Dict] with:
            - source: str (actor id)
            - target: str (actor id)

        semantic_edges: List[Dict] with:
            - event_id: str
            - actor_id: str
            - role: str (e.g., "Agent", "Patient", "Theme")
    """

    # ========================================================================
    # OPTION 1: If you have a DRS parsing library (recommended)
    # ========================================================================
    # Uncomment and modify this if you have a DRS parser:
    """
    try:
        from your_drs_parser import parse_drs  # Replace with your parser
        parsed_drs = parse_drs(drs_text)

        # Extract components from parsed DRS
        events = parsed_drs.get_events()
        actors = parsed_drs.get_actors()
        temporal_edges = parsed_drs.get_temporal_relations()
        coreference_edges = parsed_drs.get_coreference_chains()
        semantic_edges = parsed_drs.get_semantic_roles()

        return StoryGraph(
            story_id="parsed",
            events=events,
            actors=actors,
            temporal_edges=temporal_edges,
            coreference_edges=coreference_edges,
            semantic_edges=semantic_edges,
        )
    except Exception as e:
        print(f"Warning: DRS parsing failed: {e}")
        return None
    """

    # ========================================================================
    # OPTION 2: If your DRS is in JSON format
    # ========================================================================
    # Uncomment and modify if your DRS files are JSON:
    """
    try:
        import json
        drs_data = json.loads(drs_text)

        return StoryGraph(
            story_id=drs_data.get("story_id", "parsed"),
            events=drs_data.get("events", []),
            actors=drs_data.get("actors", []),
            temporal_edges=drs_data.get("temporal_edges", []),
            coreference_edges=drs_data.get("coreference_edges", []),
            semantic_edges=drs_data.get("semantic_edges", []),
        )
    except Exception as e:
        print(f"Warning: JSON parsing failed: {e}")
        return None
    """

    # ========================================================================
    # OPTION 3: If you use text2story or similar library
    # ========================================================================
    # Uncomment if you use text2story:
    """
    try:
        from text2story import parse_narrative

        parsed = parse_narrative(drs_text)

        # Convert to StoryGraph format
        events = [
            {
                'id': e.id,
                'type': e.event_type,
                'text': e.text,
                'lemma': e.lemma,
                'tense': e.tense,
                'aspect': e.aspect,
                'vform': e.vform,
                'position': e.position,
                'verbnet_class': e.verbnet_class,
                'predicates': e.predicates,
            }
            for e in parsed.events
        ]

        # Similar for actors, edges, etc.

        return StoryGraph(
            story_id="parsed",
            events=events,
            actors=...,
            temporal_edges=...,
            coreference_edges=...,
            semantic_edges=...,
        )
    except Exception as e:
        print(f"Warning: text2story parsing failed: {e}")
        return None
    """

    # ========================================================================
    # FALLBACK: Return empty StoryGraph (text-only mode)
    # ========================================================================
    print("Warning: Using text-only mode (no DRS parsing implemented)")
    return StoryGraph(
        story_id="parsed",
        events=[],
        actors=[],
        temporal_edges=[],
        coreference_edges=[],
        semantic_edges=[],
    )


class InferenceModel:
    """Wrapper for the five-component model for inference with DRS."""

    def __init__(self, model: FiveComponentModel, processor: StoryDataProcessor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def encode_single_story(self, text: str, story_graph: Optional[StoryGraph]) -> np.ndarray:
        """
        Encode a single story with text and optional DRS.

        Args:
            text: Story text
            story_graph: Parsed DRS (StoryGraph), or None for text-only

        Returns:
            embedding: numpy array of shape [output_dim]
        """
        # Use provided story_graph or create empty one
        if story_graph is None:
            story_graph = StoryGraph(
                story_id="inference",
                events=[],
                actors=[],
                temporal_edges=[],
                coreference_edges=[],
                semantic_edges=[],
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

        # Component 5: Text
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


def load_test_data(test_json_path: str, drs_dir: str) -> List[Dict]:
    """
    Load test data from JSON and DRS files.

    Returns:
        List of dicts with 'text', 'drs_path', 'line_num'
    """
    test_json_path = Path(test_json_path)
    drs_dir = Path(drs_dir)

    if not test_json_path.exists():
        raise FileNotFoundError(f"Test JSON not found: {test_json_path}")

    if not drs_dir.exists():
        raise FileNotFoundError(f"DRS directory not found: {drs_dir}")

    print("\n" + "=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    test_data = []
    with open(test_json_path, 'r') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'text' not in data:
                        raise ValueError(f"Line {line_num}: Missing 'text' field")

                    # Find corresponding DRS file
                    drs_path = drs_dir / f"{line_num}_drs.txt"

                    test_data.append({
                        'text': data['text'],
                        'drs_path': drs_path,
                        'line_num': line_num,
                    })
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_num}: Invalid JSON - {e}")

    print(f"✓ Loaded {len(test_data)} stories from {test_json_path}")

    # Check how many DRS files exist
    drs_found = sum(1 for item in test_data if item['drs_path'].exists())
    drs_missing = len(test_data) - drs_found

    print(f"✓ Found {drs_found} DRS files in {drs_dir}")
    if drs_missing > 0:
        print(f"⚠ Warning: {drs_missing} DRS files missing (will use text-only for these)")

    return test_data


def generate_embeddings(
        inference_model: InferenceModel,
        test_data: List[Dict],
) -> np.ndarray:
    """
    Generate embeddings for all test stories using text + DRS.

    Returns:
        embeddings: numpy array of shape [num_stories, embedding_dim]
    """
    print("\n" + "=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)
    print(f"Processing {len(test_data)} stories...")

    all_embeddings = []
    drs_loaded = 0
    drs_failed = 0
    text_only = 0

    start_time = time.time()

    for i, item in enumerate(test_data):
        text = item['text']
        drs_path = item['drs_path']

        # Try to load DRS
        story_graph = None
        if drs_path.exists():
            story_graph = load_drs_from_file(drs_path)
            if story_graph is not None:
                drs_loaded += 1
            else:
                drs_failed += 1
        else:
            text_only += 1

        # Generate embedding
        embedding = inference_model.encode_single_story(text, story_graph)
        all_embeddings.append(embedding)

        # Progress update
        if (i + 1) % 100 == 0 or (i + 1) == len(test_data):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test_data) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i + 1}/{len(test_data)}] {rate:.1f} stories/s, ETA: {eta:.0f}s "
                  f"(DRS: {drs_loaded}, Text-only: {text_only + drs_failed})")

    # Stack into single array
    embeddings = np.stack(all_embeddings, axis=0)

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
    print(f"\n  Statistics:")
    print(f"    - With DRS: {drs_loaded}")
    print(f"    - Text-only: {text_only}")
    print(f"    - DRS parse failed: {drs_failed}")

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
        description='Generate Track B embeddings with DRS annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings with DRS
  python generate_embeddings_with_drs.py \\
      --checkpoint checkpoints_five \\
      --test_json track_b_test.jsonl \\
      --drs_dir track_b_drs \\
      --output track_b_embeddings \\
      --create_zip

Directory structure:
  track_b_test.jsonl     # Line 0: {"text": "..."}
  track_b_drs/
      0_drs.txt          # DRS for line 0
      1_drs.txt          # DRS for line 1
      ...
        """
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--test_json',
        type=str,
        required=True,
        help='Path to test JSONL file'
    )
    parser.add_argument(
        '--drs_dir',
        type=str,
        required=True,
        help='Path to directory with DRS files (named {line_num}_drs.txt)'
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
    print("TRACK B EMBEDDING GENERATION WITH DRS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test JSON: {args.test_json}")
    print(f"DRS directory: {args.drs_dir}")

    try:
        # Load model
        inference_model = load_model_and_config(args.checkpoint, device)

        # Load test data
        test_data = load_test_data(args.test_json, args.drs_dir)

        # Generate embeddings
        embeddings = generate_embeddings(inference_model, test_data)

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