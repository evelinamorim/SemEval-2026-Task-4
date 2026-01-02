"""
Step 2.2b: Text-Enhanced Feature Encoder

Adds sentence-transformer embeddings for event text to capture semantic meaning.

Features:
- Base features (type, tense, aspect, vform, position): 15 dim
- VerbNet hash embedding: 32 dim
- Logic predicates: 119 dim
- Event text embedding (sentence-transformer): 384 dim (or 768 for larger models)

Total: ~550 dim (vs 166 before)

Usage:
    encoder = TextEnhancedFeatureEncoder(dataset, model_name='all-MiniLM-L6-v2')
    features = encoder.encode_story(story_graph)  # [num_events, ~550]
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Import our modules
try:
    from data_loader import DRSDataset, StoryGraph, Triplet
    from feature_encoder import EventFeatureEncoder
except ImportError:
    pass


class TextEnhancedFeatureEncoder:
    """
    Feature encoder that combines structural features with text embeddings.

    Feature vector structure:
    - [0:15]      Base features (type, tense, aspect, vform, position)
    - [15:47]     VerbNet hash embedding (32 dim)
    - [47:166]    Logic predicates (119 dim, varies by dataset)
    - [166:550]   Event text embedding (384 dim for MiniLM)

    Options for text encoding:
    - 'event_only': Encode just the event text (e.g., "arrives")
    - 'event_context': Encode event + surrounding context
    - 'full_story': Also include full story embedding
    """

    # Recommended models (speed vs quality tradeoff)
    MODELS = {
        'fast': 'all-MiniLM-L6-v2',  # 384 dim, very fast
        'balanced': 'all-mpnet-base-v2',  # 768 dim, good balance
        'accurate': 'all-roberta-large-v1',  # 1024 dim, most accurate
    }

    def __init__(
            self,
            dataset: Optional['DRSDataset'] = None,
            model_name: str = 'all-MiniLM-L6-v2',
            include_base_features: bool = True,
            include_verbnet: bool = True,
            include_predicates: bool = True,
            text_mode: str = 'event_only',
            cache_embeddings: bool = True,
            device: str = 'auto',
            verbnet_dim: int = 32,
    ):
        """
        Initialize text-enhanced encoder.

        Args:
            dataset: DRSDataset to extract vocabularies from
            model_name: Sentence-transformer model name
            include_base_features: Include one-hot features (type, tense, etc.)
            include_verbnet: Include VerbNet hash embeddings
            include_predicates: Include logic predicate multi-hot
            text_mode: How to encode text ('event_only', 'event_context')
            cache_embeddings: Cache computed embeddings for speed
            device: Device for model ('auto', 'cpu', 'cuda')
            verbnet_dim: Dimension for VerbNet hash embedding
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")

        self.model_name = model_name
        self.include_base_features = include_base_features
        self.include_verbnet = include_verbnet
        self.include_predicates = include_predicates
        self.text_mode = text_mode
        self.cache_embeddings = cache_embeddings
        self.verbnet_dim = verbnet_dim

        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load sentence-transformer model
        print(f"Loading sentence-transformer model: {model_name}...")
        self.text_model = SentenceTransformer(model_name, device=self.device)
        self.text_dim = self.text_model.get_sentence_embedding_dimension()
        print(f"✓ Loaded model with embedding dim: {self.text_dim}")

        # Initialize base encoder for structural features
        self.base_encoder = EventFeatureEncoder(
            dataset=dataset,
            include_verbnet=include_verbnet,
            include_predicates=include_predicates,
            verbnet_dim=verbnet_dim,
        )

        self.base_dim = self.base_encoder.feature_dim

        # Calculate total feature dimension
        self.feature_dim = self.base_dim + self.text_dim

        # Cache for text embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}

        print(f"\n✓ TextEnhancedFeatureEncoder initialized")
        print(f"  Base features: {self.base_dim}")
        print(f"  Text embedding: {self.text_dim}")
        print(f"  Total feature dim: {self.feature_dim}")
        print(f"  Text mode: {self.text_mode}")
        print(f"  Device: {self.device}")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding, using cache if available.
        """
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]

        # Compute embedding
        embedding = self.text_model.encode(text, convert_to_numpy=True)

        if self.cache_embeddings:
            self._embedding_cache[text] = embedding

        return embedding

    def _get_event_text(self, event: Dict, story_text: str = "") -> str:
        """
        Prepare event text for embedding based on text_mode.
        """
        event_text = event.get('text', '')

        if self.text_mode == 'event_only':
            # Just the event text
            return event_text

        elif self.text_mode == 'event_context':
            # Event text with some context (if available)
            # This could be expanded to include surrounding events
            return event_text

        else:
            return event_text

    def encode_event(
            self,
            event: Dict,
            vn_class: Optional[str] = None,
            predicates: Optional[List[str]] = None,
            num_events: int = 1,
            story_text: str = "",
    ) -> np.ndarray:
        """
        Encode a single event with text embedding.

        Args:
            event: Event dict with keys: text, type, tense, aspect, vform, position
            vn_class: VerbNet class for this event
            predicates: Logic predicates
            num_events: Total events in story
            story_text: Full story text (for context)

        Returns:
            Feature vector as numpy array
        """
        # Get base features
        base_features = self.base_encoder.encode_event(
            event=event,
            vn_class=vn_class,
            predicates=predicates,
            num_events=num_events,
        )

        # Get text embedding
        event_text = self._get_event_text(event, story_text)
        text_embedding = self._get_text_embedding(event_text)

        # Combine features
        combined = np.concatenate([base_features, text_embedding])

        return combined

    def encode_story(self, story: 'StoryGraph', story_text: str = "") -> torch.Tensor:
        """
        Encode all events in a story.

        Args:
            story: StoryGraph object
            story_text: Optional full story text for context

        Returns:
            Tensor of shape [num_events, feature_dim]
        """
        features = []
        num_events = story.num_events
        predicates = story.logic_predicates

        # Batch encode all event texts for efficiency
        event_texts = [self._get_event_text(event, story_text) for event in story.events]

        # Get all text embeddings at once (much faster than one-by-one)
        if self.cache_embeddings:
            # Check cache first, encode only uncached
            text_embeddings = []
            texts_to_encode = []
            indices_to_encode = []

            for i, text in enumerate(event_texts):
                if text in self._embedding_cache:
                    text_embeddings.append(self._embedding_cache[text])
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
                    text_embeddings.append(None)  # Placeholder

            # Batch encode uncached texts
            if texts_to_encode:
                new_embeddings = self.text_model.encode(texts_to_encode, convert_to_numpy=True)
                for idx, text, emb in zip(indices_to_encode, texts_to_encode, new_embeddings):
                    self._embedding_cache[text] = emb
                    text_embeddings[idx] = emb
        else:
            # Batch encode all texts
            text_embeddings = self.text_model.encode(event_texts, convert_to_numpy=True)

        # Combine base features with text embeddings
        for i, event in enumerate(story.events):
            event_id = event['id']
            vn_class = story.verbnet_classes.get(event_id)

            # Get base features
            base_features = self.base_encoder.encode_event(
                event=event,
                vn_class=vn_class,
                predicates=predicates,
                num_events=num_events,
            )

            # Combine
            combined = np.concatenate([base_features, text_embeddings[i]])
            features.append(combined)

        return torch.tensor(np.array(features), dtype=torch.float32)

    def encode_triplet(self, triplet: 'Triplet') -> Dict[str, torch.Tensor]:
        """
        Encode all three stories in a triplet.

        Returns:
            Dict with keys 'anchor', 'story_a', 'story_b'
        """
        return {
            'anchor': self.encode_story(triplet.anchor, triplet.anchor_text),
            'story_a': self.encode_story(triplet.story_a, triplet.text_a),
            'story_b': self.encode_story(triplet.story_b, triplet.text_b),
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        print(f"✓ Cleared embedding cache")

    def cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cached_texts': len(self._embedding_cache),
            'memory_mb': sum(e.nbytes for e in self._embedding_cache.values()) / 1024 / 1024,
        }


# ============================================================
# VERIFICATION SCRIPT
# ============================================================

if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python text_feature_encoder.py <preprocessed.jsonl>")
        print("\nModels available:")
        for name, model in TextEnhancedFeatureEncoder.MODELS.items():
            print(f"  {name}: {model}")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'all-MiniLM-L6-v2'

    print("\n" + "=" * 60)
    print("STEP 2.2b: TEXT-ENHANCED FEATURE ENCODER VERIFICATION")
    print("=" * 60 + "\n")

    # Load dataset
    from data_loader import DRSDataset

    dataset = DRSDataset(jsonl_path)

    # Create encoder
    print("\n" + "-" * 40)
    print("ENCODER INITIALIZATION")
    print("-" * 40)

    encoder = TextEnhancedFeatureEncoder(
        dataset=dataset,
        model_name=model_name,
        include_base_features=True,
        include_verbnet=True,
        include_predicates=True,
        text_mode='event_only',
        cache_embeddings=True,
    )

    # Test on first triplet
    print("\n" + "-" * 40)
    print("ENCODING TEST")
    print("-" * 40)

    triplet = dataset[0]

    start_time = time.time()
    encoded = encoder.encode_triplet(triplet)
    encode_time = time.time() - start_time

    print(f"\nTriplet 0 shapes:")
    print(f"  Anchor:  {encoded['anchor'].shape}")
    print(f"  Story A: {encoded['story_a'].shape}")
    print(f"  Story B: {encoded['story_b'].shape}")
    print(f"  Encoding time: {encode_time:.3f}s")

    # Show sample features
    print(f"\nSample feature vector (first event of anchor):")
    first_event = encoded['anchor'][0]
    print(f"  Shape: {first_event.shape}")
    print(f"  Base features (first 15): {first_event[:15].tolist()}")
    print(f"  Text embedding (first 10): {first_event[-384:-374].tolist()}")

    # Decode base features
    print(f"\n  Decoded base features:")
    idx = 0
    print(f"    Event type: {encoder.base_encoder.EVENT_TYPES[int(first_event[idx:idx + 4].argmax())]}")
    idx += 4
    print(f"    Tense: {encoder.base_encoder.TENSES[int(first_event[idx:idx + 4].argmax())]}")
    idx += 4
    print(f"    Aspect: {encoder.base_encoder.ASPECTS[int(first_event[idx:idx + 3].argmax())]}")

    # Test batch encoding speed
    print("\n" + "-" * 40)
    print("BATCH ENCODING SPEED TEST")
    print("-" * 40)

    batch_size = 10
    start_time = time.time()

    for i in range(batch_size):
        triplet = dataset[i]
        encoded = encoder.encode_triplet(triplet)

    batch_time = time.time() - start_time

    print(f"Encoded {batch_size} triplets in {batch_time:.2f}s")
    print(f"Average: {batch_time / batch_size:.3f}s per triplet")
    print(f"Cache stats: {encoder.cache_stats()}")

    # Test semantic similarity
    print("\n" + "-" * 40)
    print("SEMANTIC SIMILARITY TEST")
    print("-" * 40)

    # Get embeddings for similar/different event texts
    test_texts = [
        "arrives",
        "comes",
        "departs",
        "leaves",
        "eats",
        "sleeping",
    ]

    embeddings = encoder.text_model.encode(test_texts, convert_to_numpy=True)

    print("\nCosine similarities between event texts:")
    from numpy.linalg import norm


    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))


    # Print similarity matrix
    print(f"{'':>12}", end="")
    for t in test_texts:
        print(f"{t:>10}", end="")
    print()

    for i, t1 in enumerate(test_texts):
        print(f"{t1:>12}", end="")
        for j, t2 in enumerate(test_texts):
            sim = cosine_sim(embeddings[i], embeddings[j])
            print(f"{sim:>10.3f}", end="")
        print()

    print("\nExpected patterns:")
    print("  - 'arrives' ~ 'comes' (high similarity)")
    print("  - 'departs' ~ 'leaves' (high similarity)")
    print("  - 'arrives' vs 'eats' (low similarity)")

    # Feature statistics
    print("\n" + "-" * 40)
    print("FEATURE STATISTICS")
    print("-" * 40)

    # Encode several stories
    all_features = []
    for i in range(min(50, len(dataset))):
        features = encoder.encode_story(dataset[i].anchor)
        all_features.append(features)

    all_features = torch.cat(all_features, dim=0)

    print(f"Combined shape: {all_features.shape}")
    print(f"Mean: {all_features.mean().item():.4f}")
    print(f"Std: {all_features.std().item():.4f}")
    print(f"Min: {all_features.min().item():.4f}")
    print(f"Max: {all_features.max().item():.4f}")

    # Check for NaN/Inf
    print(f"Has NaN: {torch.isnan(all_features).any().item()}")
    print(f"Has Inf: {torch.isinf(all_features).any().item()}")

    # Base vs text feature statistics
    base_dim = encoder.base_dim
    print(f"\nBase features (dim 0:{base_dim}):")
    print(f"  Mean: {all_features[:, :base_dim].mean().item():.4f}")
    print(f"  Std: {all_features[:, :base_dim].std().item():.4f}")

    print(f"\nText embeddings (dim {base_dim}:{encoder.feature_dim}):")
    print(f"  Mean: {all_features[:, base_dim:].mean().item():.4f}")
    print(f"  Std: {all_features[:, base_dim:].std().item():.4f}")

    print("\n" + "=" * 60)
    print("✓ TEXT-ENHANCED FEATURE ENCODER VERIFICATION COMPLETE")
    print("=" * 60)