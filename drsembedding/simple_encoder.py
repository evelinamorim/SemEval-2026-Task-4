"""
Step 2.4: Simple Encoder (Mean Pooling Baseline)

A simple baseline encoder that:
1. Projects event features to hidden dimension
2. Aggregates via mean pooling
3. Produces fixed-size story embeddings

This serves as a baseline before adding GNN components.

Usage:
    encoder = SimpleTemporalEncoder(input_dim=166, hidden_dim=128, output_dim=64)
    embedding = encoder(event_features)  # [batch, output_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

# Import our modules
try:
    from data_loader import DRSDataset, StoryGraph, Triplet
    from feature_encoder import EventFeatureEncoder
    from graph_encoder import GraphStructureEncoder, TemporalGraph
except ImportError:
    pass


class SimpleTemporalEncoder(nn.Module):
    """
    Simple encoder using mean pooling over event features.

    Architecture:
        Event Features [N, input_dim]
        -> Linear + ReLU [N, hidden_dim]
        -> Linear + ReLU [N, hidden_dim]
        -> Mean Pooling [1, hidden_dim]
        -> Linear [1, output_dim]
        -> Story Embedding
    """

    def __init__(
            self,
            input_dim: int = 166,
            hidden_dim: int = 128,
            output_dim: int = 64,
            dropout: float = 0.1,
            num_layers: int = 2,
    ):
        """
        Initialize encoder.

        Args:
            input_dim: Dimension of input event features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout probability
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Build MLP layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.event_encoder = nn.Sequential(*layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
            self,
            event_features: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a story's events into a fixed-size embedding.

        Args:
            event_features: [num_events, input_dim] or [batch, max_events, input_dim]
            mask: Optional [batch, max_events] mask for padded sequences

        Returns:
            Story embedding [output_dim] or [batch, output_dim]
        """
        # Handle both single story and batched input
        if event_features.dim() == 2:
            # Single story: [num_events, input_dim]
            encoded = self.event_encoder(event_features)  # [num_events, hidden_dim]
            pooled = encoded.mean(dim=0)  # [hidden_dim]
            embedding = self.output_projection(pooled)  # [output_dim]
            return embedding

        else:
            # Batched: [batch, max_events, input_dim]
            batch_size = event_features.size(0)

            # Encode all events
            encoded = self.event_encoder(event_features)  # [batch, max_events, hidden_dim]

            # Apply mask for mean pooling
            if mask is not None:
                # Expand mask for hidden dim
                mask_expanded = mask.unsqueeze(-1).float()  # [batch, max_events, 1]
                encoded = encoded * mask_expanded

                # Mean over non-padded events
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1]
                pooled = encoded.sum(dim=1) / lengths  # [batch, hidden_dim]
            else:
                pooled = encoded.mean(dim=1)  # [batch, hidden_dim]

            embedding = self.output_projection(pooled)  # [batch, output_dim]
            return embedding

    def encode_story(self, story_features: torch.Tensor) -> torch.Tensor:
        """Convenience method for single story encoding."""
        return self.forward(story_features)


class TripletSimilarityModel(nn.Module):
    """
    Full model for triplet similarity prediction.

    Takes three stories (anchor, A, B) and predicts which one is closer to anchor.

    Architecture:
        Anchor -> Encoder -> anchor_emb
        Story A -> Encoder -> a_emb
        Story B -> Encoder -> b_emb

        sim_A = cosine(anchor_emb, a_emb)
        sim_B = cosine(anchor_emb, b_emb)

        prediction = sim_A > sim_B
    """

    def __init__(
            self,
            input_dim: int = 166,
            hidden_dim: int = 128,
            output_dim: int = 64,
            dropout: float = 0.1,
            similarity: str = 'cosine',
    ):
        """
        Initialize triplet model.

        Args:
            input_dim: Event feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Embedding dimension
            dropout: Dropout probability
            similarity: Similarity function ('cosine', 'dot', 'euclidean')
        """
        super().__init__()

        self.encoder = SimpleTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

        self.similarity = similarity

        # Optional: learnable comparison MLP
        self.use_comparison_mlp = False
        if self.use_comparison_mlp:
            self.comparison_mlp = nn.Sequential(
                nn.Linear(output_dim * 3, hidden_dim),  # concat + diff
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def compute_similarity(
            self,
            emb1: torch.Tensor,
            emb2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between two embeddings.

        Args:
            emb1, emb2: Embeddings [batch, dim] or [dim]

        Returns:
            Similarity scores [batch] or scalar
        """
        if self.similarity == 'cosine':
            return F.cosine_similarity(emb1, emb2, dim=-1)
        elif self.similarity == 'dot':
            return (emb1 * emb2).sum(dim=-1)
        elif self.similarity == 'euclidean':
            return -torch.norm(emb1 - emb2, dim=-1)  # Negative distance
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")

    def forward(
            self,
            anchor_features: torch.Tensor,
            story_a_features: torch.Tensor,
            story_b_features: torch.Tensor,
            anchor_mask: Optional[torch.Tensor] = None,
            a_mask: Optional[torch.Tensor] = None,
            b_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for triplet comparison.

        Args:
            anchor_features: Anchor event features
            story_a_features: Story A event features
            story_b_features: Story B event features
            *_mask: Optional masks for padded batches

        Returns:
            Dict with:
                - anchor_emb: Anchor embedding
                - a_emb: Story A embedding
                - b_emb: Story B embedding
                - sim_a: Similarity anchor-A
                - sim_b: Similarity anchor-B
                - logits: sim_a - sim_b (positive = A closer)
        """
        # Encode all three stories
        anchor_emb = self.encoder(anchor_features, anchor_mask)
        a_emb = self.encoder(story_a_features, a_mask)
        b_emb = self.encoder(story_b_features, b_mask)

        # Compute similarities
        sim_a = self.compute_similarity(anchor_emb, a_emb)
        sim_b = self.compute_similarity(anchor_emb, b_emb)

        # Logits: positive means A is closer
        logits = sim_a - sim_b

        return {
            'anchor_emb': anchor_emb,
            'a_emb': a_emb,
            'b_emb': b_emb,
            'sim_a': sim_a,
            'sim_b': sim_b,
            'logits': logits,
        }

    def predict(
            self,
            anchor_features: torch.Tensor,
            story_a_features: torch.Tensor,
            story_b_features: torch.Tensor,
    ) -> bool:
        """
        Predict which story is closer to anchor.

        Returns:
            True if A is closer, False if B is closer
        """
        with torch.no_grad():
            output = self.forward(anchor_features, story_a_features, story_b_features)
            return output['logits'].item() > 0


def compute_triplet_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        margin: float = 0.0,
) -> torch.Tensor:
    """
    Compute loss for triplet similarity.

    We use binary cross-entropy on the sigmoid of logits.

    Args:
        logits: sim_a - sim_b scores [batch]
        labels: 1 if A is closer, 0 if B is closer [batch]
        margin: Optional margin (not used with BCE)

    Returns:
        Loss scalar
    """
    # BCE loss: predicts probability that A is closer
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def compute_accuracy(
        logits: torch.Tensor,
        labels: torch.Tensor,
) -> float:
    """
    Compute accuracy for triplet predictions.

    Args:
        logits: sim_a - sim_b scores [batch]
        labels: 1 if A is closer, 0 if B is closer [batch]

    Returns:
        Accuracy as float
    """
    predictions = (logits > 0).long()
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total


# ============================================================
# VERIFICATION SCRIPT
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python simple_encoder.py <preprocessed.jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    print("\n" + "=" * 60)
    print("STEP 2.4: SIMPLE ENCODER VERIFICATION")
    print("=" * 60 + "\n")

    # Load dataset
    from data_loader import DRSDataset
    from feature_encoder import EventFeatureEncoder

    dataset = DRSDataset(jsonl_path)
    feature_encoder = EventFeatureEncoder(dataset=dataset)

    # Create model
    print("\n" + "-" * 40)
    print("MODEL INITIALIZATION")
    print("-" * 40)

    input_dim = feature_encoder.feature_dim
    model = TripletSimilarityModel(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=64,
        dropout=0.1,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input dim: {input_dim}")
    print(f"Hidden dim: 128")
    print(f"Output dim: 64")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test on single triplet
    print("\n" + "-" * 40)
    print("SINGLE TRIPLET TEST")
    print("-" * 40)

    triplet = dataset[0]
    encoded = feature_encoder.encode_triplet(triplet)

    model.eval()
    with torch.no_grad():
        output = model(
            encoded['anchor'],
            encoded['story_a'],
            encoded['story_b'],
        )

    print(f"\nEmbedding shapes:")
    print(f"  Anchor: {output['anchor_emb'].shape}")
    print(f"  Story A: {output['a_emb'].shape}")
    print(f"  Story B: {output['b_emb'].shape}")

    print(f"\nSimilarities:")
    print(f"  Anchor-A: {output['sim_a'].item():.4f}")
    print(f"  Anchor-B: {output['sim_b'].item():.4f}")
    print(f"  Logits (A-B): {output['logits'].item():.4f}")

    print(f"\nPrediction: {'A is closer' if output['logits'].item() > 0 else 'B is closer'}")
    print(f"Ground truth: {'A is closer' if triplet.label else 'B is closer'}")
    print(f"Correct: {(output['logits'].item() > 0) == triplet.label}")

    # Test batch processing
    print("\n" + "-" * 40)
    print("BATCH PROCESSING TEST")
    print("-" * 40)

    batch_size = 8
    all_logits = []
    all_labels = []

    for i in range(batch_size):
        triplet = dataset[i]
        encoded = feature_encoder.encode_triplet(triplet)

        with torch.no_grad():
            output = model(
                encoded['anchor'],
                encoded['story_a'],
                encoded['story_b'],
            )

        all_logits.append(output['logits'])
        all_labels.append(1 if triplet.label else 0)

    logits_tensor = torch.stack(all_logits)
    labels_tensor = torch.tensor(all_labels)

    # Compute loss and accuracy
    loss = compute_triplet_loss(logits_tensor, labels_tensor)
    accuracy = compute_accuracy(logits_tensor, labels_tensor)

    print(f"Batch size: {batch_size}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy * 100:.1f}%")
    print(f"(Random baseline: 50%)")

    # Test gradient flow
    print("\n" + "-" * 40)
    print("GRADIENT FLOW TEST")
    print("-" * 40)

    model.train()
    triplet = dataset[0]
    encoded = feature_encoder.encode_triplet(triplet)

    output = model(
        encoded['anchor'],
        encoded['story_a'],
        encoded['story_b'],
    )

    label = torch.tensor([1.0 if triplet.label else 0.0])
    loss = compute_triplet_loss(output['logits'].unsqueeze(0), label)

    loss.backward()

    # Check gradients
    has_gradients = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  WARNING: No gradient for {name}")
            has_gradients = False
        elif param.grad.abs().sum() == 0:
            print(f"  WARNING: Zero gradient for {name}")

    if has_gradients:
        print("✓ All parameters have gradients")

    # Embedding statistics
    print("\n" + "-" * 40)
    print("EMBEDDING STATISTICS")
    print("-" * 40)

    model.eval()
    all_embeddings = []

    for i in range(min(100, len(dataset))):
        triplet = dataset[i]
        encoded = feature_encoder.encode_triplet(triplet)

        with torch.no_grad():
            anchor_emb = model.encoder(encoded['anchor'])
            all_embeddings.append(anchor_emb)

    embeddings = torch.stack(all_embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean().item():.4f}")
    print(f"Std: {embeddings.std().item():.4f}")
    print(f"Min: {embeddings.min().item():.4f}")
    print(f"Max: {embeddings.max().item():.4f}")

    # Check embedding variance per dimension
    dim_std = embeddings.std(dim=0)
    print(f"Per-dim std: min={dim_std.min().item():.4f}, max={dim_std.max().item():.4f}")

    # Check for degenerate embeddings (all same)
    pairwise_dist = torch.cdist(embeddings[:10], embeddings[:10])
    print(f"Pairwise distances (10x10): mean={pairwise_dist.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("✓ SIMPLE ENCODER VERIFICATION COMPLETE")
    print("=" * 60)