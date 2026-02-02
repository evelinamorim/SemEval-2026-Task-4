"""
Preprocess Track B data for DRS-based embedding generation.

Track B provides individual stories (not triplets). This script:
1. Reads the test JSONL with story texts
2. Parses corresponding DRS files
3. Outputs a preprocessed JSONL ready for the embedding model

Usage:
    python preprocess_track_b.py \
        --test_json track_b_test.jsonl \
        --drs_dir track_b_drs \
        --output track_b_preprocessed.jsonl

Expected input structure:
    track_b_test.jsonl:
        {"text": "Story 1 text..."}
        {"text": "Story 2 text..."}
        ...

    track_b_drs/
        0_drs.txt    # DRS for story at line 0
        1_drs.txt    # DRS for story at line 1
        ...

Output format (one JSON per line):
    {
        "idx": 0,
        "text": "Story text...",
        "events": [...],
        "actors": [...],
        "temporal_edges": [...],
        "coreference_edges": [...],
        "semantic_edges": [...],
        "event_types": {...},
        "verbnet_classes": {...},
        "logic_predicates": [...]
    }
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import sys

# Import the DRS extractor from preprocessing.py
from preprocessing import DRSGraphExtractor, DRSGraph


def load_test_stories(jsonl_path: str) -> List[Dict[str, str]]:
    """Load test stories from JSONL file."""
    stories = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                # Handle different possible field names
                text = item.get('text', item.get('story', item.get('content', '')))
                stories.append({'text': text})
    return stories


def find_drs_file(drs_dir: Path, idx: int) -> Optional[Path]:
    """
    Find the DRS file for a given story index.

    Tries multiple naming conventions:
    - {idx}_drs.txt
    - {idx}.txt
    - story_{idx}_drs.txt
    - {idx}_anchor_drs.txt (for compatibility)
    """
    patterns = [
        f"{idx}_drs.txt",
        f"{idx}.txt",
        f"story_{idx}_drs.txt",
        f"{idx}_anchor_drs.txt",
    ]

    for pattern in patterns:
        path = drs_dir / pattern
        if path.exists():
            return path

    return None


def preprocess_single_story(
    extractor: DRSGraphExtractor,
    text: str,
    drs_path: Optional[Path],
    idx: int
) -> Dict[str, Any]:
    """
    Preprocess a single story with its DRS.

    Returns a dictionary ready for the embedding model.
    """
    result = {
        'idx': idx,
        'text': text,
    }

    # Parse DRS if available
    if drs_path and drs_path.exists():
        graph = extractor.parse_drs_file(str(drs_path))

        if graph is not None:
            result['events'] = graph.events
            result['actors'] = graph.actors
            result['temporal_edges'] = graph.temporal_edges
            result['coreference_edges'] = graph.coreference_edges
            result['semantic_edges'] = graph.semantic_edges
            result['event_types'] = graph.event_types
            result['verbnet_classes'] = graph.verbnet_classes
            result['logic_predicates'] = graph.logic_predicates
            result['has_drs'] = True
        else:
            # DRS parsing failed - use empty structures
            result.update(get_empty_drs_fields())
            result['has_drs'] = False
    else:
        # No DRS file - use empty structures
        result.update(get_empty_drs_fields())
        result['has_drs'] = False

    return result


def get_empty_drs_fields() -> Dict[str, Any]:
    """Return empty DRS fields for stories without DRS."""
    return {
        'events': [],
        'actors': [],
        'temporal_edges': [],
        'coreference_edges': [],
        'semantic_edges': [],
        'event_types': {},
        'verbnet_classes': {},
        'logic_predicates': [],
    }


def preprocess_track_b(
    test_jsonl: str,
    drs_dir: str,
    output_path: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Preprocess Track B test data.

    Args:
        test_jsonl: Path to test JSONL with story texts
        drs_dir: Directory containing DRS files
        output_path: Where to save preprocessed JSONL
        verbose: Print progress

    Returns:
        Statistics about preprocessing
    """
    extractor = DRSGraphExtractor()
    drs_dir = Path(drs_dir)

    # Load stories
    stories = load_test_stories(test_jsonl)

    if verbose:
        print(f"\n{'='*60}")
        print("TRACK B PREPROCESSING")
        print(f"{'='*60}")
        print(f"Test JSONL: {test_jsonl}")
        print(f"DRS directory: {drs_dir}")
        print(f"Output: {output_path}")
        print(f"Stories to process: {len(stories)}")

    # Statistics
    stats = {
        'total': len(stories),
        'with_drs': 0,
        'without_drs': 0,
        'total_events': 0,
        'total_actors': 0,
        'total_temporal_edges': 0,
        'total_semantic_edges': 0,
    }

    # Process each story
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for idx, story in enumerate(stories):
            # Find DRS file
            drs_path = find_drs_file(drs_dir, idx)

            # Preprocess
            result = preprocess_single_story(
                extractor,
                story['text'],
                drs_path,
                idx
            )

            # Update stats
            if result.get('has_drs', False):
                stats['with_drs'] += 1
                stats['total_events'] += len(result.get('events', []))
                stats['total_actors'] += len(result.get('actors', []))
                stats['total_temporal_edges'] += len(result.get('temporal_edges', []))
                stats['total_semantic_edges'] += len(result.get('semantic_edges', []))
            else:
                stats['without_drs'] += 1

            # Write to output
            f_out.write(json.dumps(result) + '\n')

            # Progress
            if verbose and (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(stories)}...")

    # Compute averages
    if stats['with_drs'] > 0:
        stats['avg_events'] = stats['total_events'] / stats['with_drs']
        stats['avg_actors'] = stats['total_actors'] / stats['with_drs']
        stats['avg_temporal_edges'] = stats['total_temporal_edges'] / stats['with_drs']
        stats['avg_semantic_edges'] = stats['total_semantic_edges'] / stats['with_drs']
    else:
        stats['avg_events'] = 0
        stats['avg_actors'] = 0
        stats['avg_temporal_edges'] = 0
        stats['avg_semantic_edges'] = 0

    if verbose:
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total stories: {stats['total']}")
        print(f"With DRS: {stats['with_drs']}")
        print(f"Without DRS (text-only): {stats['without_drs']}")
        print(f"\nAverage per story (with DRS):")
        print(f"  Events: {stats['avg_events']:.1f}")
        print(f"  Actors: {stats['avg_actors']:.1f}")
        print(f"  Temporal edges: {stats['avg_temporal_edges']:.1f}")
        print(f"  Semantic edges: {stats['avg_semantic_edges']:.1f}")
        print(f"\nOutput saved to: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Track B data for DRS embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python preprocess_track_b.py \\
        --test_json track_b_test.jsonl \\
        --drs_dir track_b_drs \\
        --output track_b_preprocessed.jsonl

    # If DRS files use different naming
    # Edit find_drs_file() function to match your naming convention
        """
    )

    parser.add_argument(
        '--test_json',
        type=str,
        required=True,
        help='Path to test JSONL with story texts'
    )
    parser.add_argument(
        '--drs_dir',
        type=str,
        required=True,
        help='Directory containing DRS files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for preprocessed JSONL'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.test_json).exists():
        print(f"Error: Test JSONL not found: {args.test_json}")
        sys.exit(1)

    if not Path(args.drs_dir).exists():
        print(f"Error: DRS directory not found: {args.drs_dir}")
        sys.exit(1)

    # Run preprocessing
    stats = preprocess_track_b(
        test_jsonl=args.test_json,
        drs_dir=args.drs_dir,
        output_path=args.output,
        verbose=not args.quiet
    )

    # Exit with error if no stories were processed
    if stats['total'] == 0:
        print("Error: No stories were processed!")
        sys.exit(1)


if __name__ == "__main__":
    main()