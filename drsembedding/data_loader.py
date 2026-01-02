"""
Step 2.1: Data Loader for Preprocessed DRS Graphs

Loads preprocessed JSONL files and provides easy access to:
- Events with attributes
- Temporal edges (graph structure)
- VerbNet classes and logic predicates
- Labels for training

Usage:
    dataset = DRSDataset('preprocessed_synth.jsonl')
    sample = dataset[0]
    print(sample['anchor']['events'])
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StoryGraph:
    """Represents a single story's DRS graph."""
    events: List[Dict]  # List of event nodes
    temporal_edges: List[Dict]  # List of {source, target, relation}
    event_types: Dict[str, str]  # event_id -> type
    verbnet_classes: Dict[str, str]  # event_id -> VerbNet class
    logic_predicates: List[str]  # Semantic predicates
    actors: List[Dict]  # Participant nodes
    coreference_edges: List[Dict]  # Coreference links
    semantic_edges: List[Dict]  # Actor-event links

    @property
    def num_events(self) -> int:
        return len(self.events)

    @property
    def num_temporal_edges(self) -> int:
        return len(self.temporal_edges)

    @property
    def event_ids(self) -> List[str]:
        """Get ordered list of event IDs (a, b, c, ...)."""
        return [e['id'] for e in self.events]

    def get_event_by_id(self, event_id: str) -> Optional[Dict]:
        """Get event by its ID."""
        for e in self.events:
            if e['id'] == event_id:
                return e
        return None

    def get_temporal_adjacency(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Get temporal adjacency as dict: {source: [(target, relation), ...]}
        """
        adj = {}
        for edge in self.temporal_edges:
            src = edge['source']
            tgt = edge['target']
            rel = edge['relation']
            if src not in adj:
                adj[src] = []
            adj[src].append((tgt, rel))
        return adj


@dataclass
class Triplet:
    """A single training triplet (anchor, story_a, story_b)."""
    idx: int
    anchor: StoryGraph
    story_a: StoryGraph
    story_b: StoryGraph
    label: bool  # True = A is closer, False = B is closer

    # Optional: raw text for debugging
    anchor_text: str = ""
    text_a: str = ""
    text_b: str = ""


class DRSDataset:
    """Dataset of preprocessed DRS triplets."""

    def __init__(self, jsonl_path: str):
        """
        Load preprocessed JSONL file.

        Args:
            jsonl_path: Path to preprocessed_*.jsonl file
        """
        self.path = Path(jsonl_path)
        self.triplets: List[Triplet] = []

        self._load_data()

        print(f"✓ Loaded {len(self.triplets)} triplets from {self.path.name}")
        self._print_stats()

    def _load_data(self):
        """Load all triplets from JSONL."""
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                triplet = self._parse_triplet(data)
                self.triplets.append(triplet)

    def _parse_triplet(self, data: Dict) -> Triplet:
        """Parse a single triplet from JSON."""
        return Triplet(
            idx=data.get('idx', 0),
            anchor=self._parse_story(data['anchor']),
            story_a=self._parse_story(data['story_a']),
            story_b=self._parse_story(data['story_b']),
            label=data.get('label', True),
            anchor_text=data.get('anchor_text', ''),
            text_a=data.get('text_a', ''),
            text_b=data.get('text_b', ''),
        )

    def _parse_story(self, data: Dict) -> StoryGraph:
        """Parse a single story graph from JSON."""
        return StoryGraph(
            events=data.get('events', []),
            temporal_edges=data.get('temporal_edges', []),
            event_types=data.get('event_types', {}),
            verbnet_classes=data.get('verbnet_classes', {}),
            logic_predicates=data.get('logic_predicates', []),
            actors=data.get('actors', []),
            coreference_edges=data.get('coreference_edges', []),
            semantic_edges=data.get('semantic_edges', []),
        )

    def _print_stats(self):
        """Print dataset statistics."""
        if not self.triplets:
            return

        # Collect stats
        anchor_events = [t.anchor.num_events for t in self.triplets]
        anchor_temp_edges = [t.anchor.num_temporal_edges for t in self.triplets]
        labels = [t.label for t in self.triplets]

        print(f"  Events per story: {sum(anchor_events) / len(anchor_events):.1f} avg")
        print(f"  Temporal edges: {sum(anchor_temp_edges) / len(anchor_temp_edges):.1f} avg")
        print(f"  Labels: {sum(labels)} A-closer, {len(labels) - sum(labels)} B-closer")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Triplet:
        return self.triplets[idx]

    def get_all_event_types(self) -> set:
        """Get all unique event types across dataset."""
        types = set()
        for t in self.triplets:
            for story in [t.anchor, t.story_a, t.story_b]:
                types.update(story.event_types.values())
        return types

    def get_all_tenses(self) -> set:
        """Get all unique tenses across dataset."""
        tenses = set()
        for t in self.triplets:
            for story in [t.anchor, t.story_a, t.story_b]:
                for event in story.events:
                    if event.get('tense'):
                        tenses.add(event['tense'])
        return tenses

    def get_all_aspects(self) -> set:
        """Get all unique aspects across dataset."""
        aspects = set()
        for t in self.triplets:
            for story in [t.anchor, t.story_a, t.story_b]:
                for event in story.events:
                    if event.get('aspect'):
                        aspects.add(event['aspect'])
        return aspects

    def get_all_vforms(self) -> set:
        """Get all unique verb forms across dataset."""
        vforms = set()
        for t in self.triplets:
            for story in [t.anchor, t.story_a, t.story_b]:
                for event in story.events:
                    if event.get('vform'):
                        vforms.add(event['vform'])
        return vforms

    def get_all_temporal_relations(self) -> set:
        """Get all unique temporal relation types."""
        rels = set()
        for t in self.triplets:
            for story in [t.anchor, t.story_a, t.story_b]:
                for edge in story.temporal_edges:
                    rels.add(edge['relation'])
        return rels

    def get_all_verbnet_classes(self) -> set:
        """Get all unique VerbNet classes."""
        classes = set()
        for t in self.triplets:
            for story in [t.anchor, t.story_a, t.story_b]:
                classes.update(story.verbnet_classes.values())
        return classes

    def get_all_logic_predicates(self) -> set:
        """Get all unique logic predicates."""
        preds = set()
        for t in self.triplets:
            for story in [t.anchor, t.story_a, t.story_b]:
                preds.update(story.logic_predicates)
        return preds


def inspect_triplet(triplet: Triplet, verbose: bool = True):
    """Pretty-print a triplet for inspection."""
    print("=" * 60)
    print(f"TRIPLET {triplet.idx}")
    print(f"Label: {'A is closer' if triplet.label else 'B is closer'}")
    print("=" * 60)

    for name, story in [('ANCHOR', triplet.anchor),
                        ('STORY_A', triplet.story_a),
                        ('STORY_B', triplet.story_b)]:
        print(f"\n--- {name} ---")
        print(f"Events: {story.num_events}")
        print(f"Temporal edges: {story.num_temporal_edges}")

        if verbose:
            print(f"\nEvents (first 5):")
            for e in story.events[:5]:
                print(f"  {e['id']}: {e['text'][:30]}... "
                      f"(type={e['type']}, tense={e['tense']}, aspect={e['aspect']})")

            print(f"\nTemporal edges (first 5):")
            for edge in story.temporal_edges[:5]:
                print(f"  {edge['source']} --{edge['relation']}--> {edge['target']}")

            print(f"\nVerbNet classes (first 5):")
            for i, (k, v) in enumerate(story.verbnet_classes.items()):
                if i >= 5:
                    break
                print(f"  {k}: {v}")

            print(f"\nLogic predicates: {story.logic_predicates[:10]}")


# ============================================================
# VERIFICATION SCRIPT
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <preprocessed.jsonl>")
        print("\nExample:")
        print("  python data_loader.py preprocessed_synth.jsonl")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    # Load dataset
    print("\n" + "=" * 60)
    print("STEP 2.1: DATA LOADER VERIFICATION")
    print("=" * 60 + "\n")

    dataset = DRSDataset(jsonl_path)

    # Print vocabulary statistics
    print("\n" + "-" * 40)
    print("VOCABULARY STATISTICS")
    print("-" * 40)

    event_types = dataset.get_all_event_types()
    tenses = dataset.get_all_tenses()
    aspects = dataset.get_all_aspects()
    vforms = dataset.get_all_vforms()
    temp_rels = dataset.get_all_temporal_relations()
    vn_classes = dataset.get_all_verbnet_classes()
    logic_preds = dataset.get_all_logic_predicates()

    print(f"Event types ({len(event_types)}): {sorted(event_types)}")
    print(f"Tenses ({len(tenses)}): {sorted(tenses)}")
    print(f"Aspects ({len(aspects)}): {sorted(aspects)}")
    print(f"Verb forms ({len(vforms)}): {sorted(vforms)}")
    print(f"Temporal relations ({len(temp_rels)}): {sorted(temp_rels)}")
    print(f"VerbNet classes: {len(vn_classes)} unique")
    print(f"Logic predicates: {len(logic_preds)} unique")

    # Inspect first triplet
    print("\n" + "-" * 40)
    print("SAMPLE TRIPLET")
    print("-" * 40)

    inspect_triplet(dataset[0], verbose=True)

    # Test adjacency
    print("\n" + "-" * 40)
    print("TEMPORAL ADJACENCY TEST")
    print("-" * 40)

    anchor = dataset[0].anchor
    adj = anchor.get_temporal_adjacency()
    print(f"Nodes with outgoing edges: {len(adj)}")
    print("Sample adjacency (first 3 nodes):")
    for i, (src, targets) in enumerate(adj.items()):
        if i >= 3:
            break
        print(f"  {src} -> {targets[:3]}{'...' if len(targets) > 3 else ''}")

    print("\n" + "=" * 60)
    print("✓ DATA LOADER VERIFICATION COMPLETE")
    print("=" * 60)