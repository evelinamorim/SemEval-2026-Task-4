"""
Step 2.2: Event Node Feature Encoding

Converts event attributes to numerical feature vectors:
- Event type (one-hot): State, Process, Transition, Unknown
- Tense (one-hot): Past, Present, Future, Unknown
- Aspect (one-hot): Perfective, Progressive, Unknown
- Verb form (one-hot): Infinitive, Participle, Unknown
- Position (normalized): 0.0 to 1.0

Also encodes:
- VerbNet class embeddings (learned or hashed)
- Logic predicates (multi-hot)

Usage:
    encoder = EventFeatureEncoder(dataset)
    features = encoder.encode_story(story_graph)  # [num_events, feature_dim]
"""

import numpy as np
from typing import Dict, List, Set, Optional
from collections import Counter
import math

# Try to import torch, fall back to numpy
try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Note: PyTorch not available, using NumPy arrays")

# Import from data_loader (assumes same directory)
try:
    from data_loader import DRSDataset, StoryGraph, Triplet
except ImportError:
    pass  # Will be imported when needed


class EventFeatureEncoder:
    """
    Encodes event attributes as numerical feature vectors.

    Feature vector structure:
    - [0:4]   Event type one-hot (State, Process, Transition, Unknown)
    - [4:8]   Tense one-hot (Past, Present, Future, Unknown)
    - [8:11]  Aspect one-hot (Perfective, Progressive, Unknown)
    - [11:14] Vform one-hot (Infinitive, Participle, Unknown)
    - [14]    Position (normalized 0-1)

    Total base features: 15

    Optional additions:
    - VerbNet class embedding
    - Logic predicate multi-hot
    """

    # Fixed vocabularies (based on DRS schema)
    EVENT_TYPES = ['State', 'Process', 'Transition', 'Unknown']
    TENSES = ['Past', 'Present', 'Future', 'Unknown']
    ASPECTS = ['Perfective', 'Progressive', 'Unknown']
    VFORMS = ['Infinitive', 'Participle', 'Unknown']

    # Normalization mappings (handle inconsistencies)
    TENSE_NORMALIZE = {
        'Pres': 'Present',
        'past': 'Past',
        'present': 'Present',
        'future': 'Future',
    }

    def __init__(
            self,
            dataset: Optional['DRSDataset'] = None,
            include_verbnet: bool = True,
            include_predicates: bool = True,
            verbnet_dim: int = 32,
            predicate_dim: int = 64,
            max_predicates: int = 150,
    ):
        """
        Initialize encoder.

        Args:
            dataset: DRSDataset to extract vocabularies from
            include_verbnet: Whether to include VerbNet class features
            include_predicates: Whether to include logic predicate features
            verbnet_dim: Embedding dimension for VerbNet classes
            predicate_dim: Dimension for predicate encoding (or max predicates for multi-hot)
            max_predicates: Maximum number of predicates for multi-hot encoding
        """
        self.include_verbnet = include_verbnet
        self.include_predicates = include_predicates
        self.verbnet_dim = verbnet_dim
        self.predicate_dim = predicate_dim
        self.max_predicates = max_predicates

        # Base feature dimension
        self.base_dim = (
                len(self.EVENT_TYPES) +  # 4
                len(self.TENSES) +  # 4
                len(self.ASPECTS) +  # 3
                len(self.VFORMS) +  # 3
                1  # position
        )  # = 15

        # Build vocabularies from dataset
        self.verbnet_vocab: Dict[str, int] = {}
        self.predicate_vocab: Dict[str, int] = {}

        if dataset is not None:
            self._build_vocabularies(dataset)

        # Calculate total feature dimension
        self.feature_dim = self.base_dim
        if self.include_verbnet:
            self.feature_dim += self.verbnet_dim
        if self.include_predicates:
            self.feature_dim += min(len(self.predicate_vocab), self.max_predicates)

        print(f"✓ EventFeatureEncoder initialized")
        print(f"  Base features: {self.base_dim}")
        print(
            f"  VerbNet classes: {len(self.verbnet_vocab)} (dim={self.verbnet_dim if self.include_verbnet else 'disabled'})")
        print(
            f"  Logic predicates: {len(self.predicate_vocab)} (dim={min(len(self.predicate_vocab), self.max_predicates) if self.include_predicates else 'disabled'})")
        print(f"  Total feature dim: {self.feature_dim}")

    def _build_vocabularies(self, dataset: 'DRSDataset'):
        """Build VerbNet and predicate vocabularies from dataset."""
        # VerbNet classes
        vn_classes = dataset.get_all_verbnet_classes()
        self.verbnet_vocab = {cls: i for i, cls in enumerate(sorted(vn_classes))}

        # Logic predicates
        predicates = dataset.get_all_logic_predicates()
        # Sort by frequency if we have access to counts, otherwise alphabetically
        self.predicate_vocab = {pred: i for i, pred in enumerate(sorted(predicates))}

    def _normalize_tense(self, tense: Optional[str]) -> str:
        """Normalize tense values to handle inconsistencies."""
        if tense is None:
            return 'Unknown'
        # Check normalization mapping
        normalized = self.TENSE_NORMALIZE.get(tense, tense)
        # Validate
        if normalized not in self.TENSES:
            return 'Unknown'
        return normalized

    def _one_hot(self, value: Optional[str], vocab: List[str]) -> List[float]:
        """Create one-hot encoding for a categorical value."""
        if value is None or value not in vocab:
            # Use 'Unknown' if available, otherwise last position
            if 'Unknown' in vocab:
                idx = vocab.index('Unknown')
            else:
                idx = len(vocab) - 1
        else:
            idx = vocab.index(value)

        one_hot = [0.0] * len(vocab)
        one_hot[idx] = 1.0
        return one_hot

    def _hash_verbnet(self, vn_class: str) -> List[float]:
        """
        Create a pseudo-embedding for VerbNet class using feature hashing.

        This is a simple approach that doesn't require learning.
        For better results, use learned embeddings (see get_verbnet_embedding_layer).
        """
        # Use multiple hash functions for diversity
        embedding = [0.0] * self.verbnet_dim

        if vn_class in self.verbnet_vocab:
            idx = self.verbnet_vocab[vn_class]
            # Distribute the index across dimensions
            for i in range(self.verbnet_dim):
                # Different hash for each dimension
                h = hash(f"{vn_class}_{i}") % 1000 / 1000.0
                embedding[i] = h
        else:
            # Unknown class - use hash of the string
            for i in range(self.verbnet_dim):
                h = hash(f"UNK_{vn_class}_{i}") % 1000 / 1000.0
                embedding[i] = h

        return embedding

    def _encode_predicates(self, predicates: List[str]) -> List[float]:
        """
        Create multi-hot encoding for logic predicates.
        """
        dim = min(len(self.predicate_vocab), self.max_predicates)
        multi_hot = [0.0] * dim

        for pred in predicates:
            if pred in self.predicate_vocab:
                idx = self.predicate_vocab[pred]
                if idx < dim:
                    multi_hot[idx] = 1.0

        return multi_hot

    def encode_event(
            self,
            event: Dict,
            vn_class: Optional[str] = None,
            predicates: Optional[List[str]] = None,
            num_events: int = 1,
    ) -> List[float]:
        """
        Encode a single event as a feature vector.

        Args:
            event: Event dict with keys: type, tense, aspect, vform, position
            vn_class: VerbNet class for this event
            predicates: Logic predicates (shared across story, applied to all events)
            num_events: Total events in story (for position normalization)

        Returns:
            List of floats representing the feature vector
        """
        features = []

        # Event type (one-hot, 4 dim)
        event_type = event.get('type') or 'Unknown'
        features.extend(self._one_hot(event_type, self.EVENT_TYPES))

        # Tense (one-hot, 4 dim)
        tense = self._normalize_tense(event.get('tense'))
        features.extend(self._one_hot(tense, self.TENSES))

        # Aspect (one-hot, 3 dim)
        aspect = event.get('aspect') or 'Unknown'
        features.extend(self._one_hot(aspect, self.ASPECTS))

        # Verb form (one-hot, 3 dim)
        vform = event.get('vform') or 'Unknown'
        features.extend(self._one_hot(vform, self.VFORMS))

        # Position (normalized, 1 dim)
        position = event.get('position', 0)
        norm_position = position / max(num_events - 1, 1)
        features.append(norm_position)

        # VerbNet class embedding
        if self.include_verbnet and vn_class:
            features.extend(self._hash_verbnet(vn_class))
        elif self.include_verbnet:
            features.extend([0.0] * self.verbnet_dim)

        # Logic predicates (multi-hot, shared across all events in story)
        if self.include_predicates and predicates:
            features.extend(self._encode_predicates(predicates))
        elif self.include_predicates:
            dim = min(len(self.predicate_vocab), self.max_predicates)
            features.extend([0.0] * dim)

        return features

    def encode_story(self, story: 'StoryGraph'):
        """
        Encode all events in a story as a feature matrix.

        Args:
            story: StoryGraph object

        Returns:
            Tensor/array of shape [num_events, feature_dim]
        """
        features = []
        num_events = story.num_events
        predicates = story.logic_predicates

        for event in story.events:
            event_id = event['id']
            vn_class = story.verbnet_classes.get(event_id)

            event_features = self.encode_event(
                event=event,
                vn_class=vn_class,
                predicates=predicates,
                num_events=num_events,
            )
            features.append(event_features)

        if HAS_TORCH:
            return torch.tensor(features, dtype=torch.float32)
        else:
            return np.array(features, dtype=np.float32)

    def encode_triplet(self, triplet: 'Triplet') -> Dict[str, any]:
        """
        Encode all three stories in a triplet.

        Returns:
            Dict with keys 'anchor', 'story_a', 'story_b', each a tensor/array
        """
        return {
            'anchor': self.encode_story(triplet.anchor),
            'story_a': self.encode_story(triplet.story_a),
            'story_b': self.encode_story(triplet.story_b),
        }

    def get_verbnet_embedding_layer(self):
        """
        Create a learnable embedding layer for VerbNet classes.

        Use this instead of hash-based embeddings for better performance.
        Requires PyTorch.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for embedding layers")
        num_classes = len(self.verbnet_vocab) + 1  # +1 for unknown
        return nn.Embedding(num_classes, self.verbnet_dim, padding_idx=0)

    def get_verbnet_index(self, vn_class: str) -> int:
        """Get vocabulary index for a VerbNet class (0 = unknown)."""
        return self.verbnet_vocab.get(vn_class, 0) + 1  # +1 because 0 is padding


# ============================================================
# VERIFICATION SCRIPT
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_encoder.py <preprocessed.jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    print("\n" + "=" * 60)
    print("STEP 2.2: FEATURE ENCODER VERIFICATION")
    print("=" * 60 + "\n")

    # Load dataset
    from data_loader import DRSDataset, inspect_triplet

    dataset = DRSDataset(jsonl_path)

    # Create encoder
    print("\n" + "-" * 40)
    print("ENCODER INITIALIZATION")
    print("-" * 40)

    encoder = EventFeatureEncoder(
        dataset=dataset,
        include_verbnet=True,
        include_predicates=True,
        verbnet_dim=32,
    )

    # Test on first triplet
    print("\n" + "-" * 40)
    print("ENCODING TEST")
    print("-" * 40)

    triplet = dataset[0]
    encoded = encoder.encode_triplet(triplet)

    print(f"\nTriplet 0 shapes:")
    print(f"  Anchor:  {encoded['anchor'].shape}")
    print(f"  Story A: {encoded['story_a'].shape}")
    print(f"  Story B: {encoded['story_b'].shape}")

    # Show sample features
    print(f"\nSample feature vector (first event of anchor):")
    first_event_features = encoded['anchor'][0]
    print(f"  Shape: {first_event_features.shape}")
    print(f"  Values (first 20): {list(first_event_features[:20])}")

    # Decode back to verify
    print(f"\n  Decoded features:")
    idx = 0
    print(f"    Event type: {encoder.EVENT_TYPES[int(np.argmax(first_event_features[idx:idx + 4]))]}")
    idx += 4
    print(f"    Tense: {encoder.TENSES[int(np.argmax(first_event_features[idx:idx + 4]))]}")
    idx += 4
    print(f"    Aspect: {encoder.ASPECTS[int(np.argmax(first_event_features[idx:idx + 3]))]}")
    idx += 3
    print(f"    Vform: {encoder.VFORMS[int(np.argmax(first_event_features[idx:idx + 3]))]}")
    idx += 3
    print(f"    Position: {float(first_event_features[idx]):.3f}")

    # Test batch encoding
    print("\n" + "-" * 40)
    print("BATCH ENCODING TEST")
    print("-" * 40)

    batch_size = 10
    all_anchors = []
    for i in range(min(batch_size, len(dataset))):
        encoded_story = encoder.encode_story(dataset[i].anchor)
        all_anchors.append(encoded_story)

    print(f"Encoded {len(all_anchors)} stories")
    print(f"Shapes: {[t.shape for t in all_anchors[:3]]}...")

    # Statistics
    print("\n" + "-" * 40)
    print("FEATURE STATISTICS")
    print("-" * 40)

    # Stack first 100 for stats
    all_features = np.concatenate([encoder.encode_story(dataset[i].anchor) for i in range(min(100, len(dataset)))],
                                  axis=0)

    print(f"Combined shape: {all_features.shape}")
    print(f"Mean: {all_features.mean():.4f}")
    print(f"Std: {all_features.std():.4f}")
    print(f"Min: {all_features.min():.4f}")
    print(f"Max: {all_features.max():.4f}")

    # Check for NaN/Inf
    print(f"Has NaN: {np.isnan(all_features).any()}")
    print(f"Has Inf: {np.isinf(all_features).any()}")

    print("\n" + "=" * 60)
    print("✓ FEATURE ENCODER VERIFICATION COMPLETE")
    print("=" * 60)