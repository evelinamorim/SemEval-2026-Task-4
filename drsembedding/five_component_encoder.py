"""
Five-Component DRS Encoder

A modular encoder with 5 distinct components:
1. Temporal Graph Encoder  - Events + temporal relations (GNN)
2. Logical Layer Encoder   - Event types + VerbNet + predicates (MLP)
3. Participant Graph Encoder - Actors + coreference (GNN)
4. Semantic Role Encoder   - Event-actor relations (Bipartite GNN)
5. Text Encoder            - Full story text (Sentence-Transformer)

Each component produces a fixed-size embedding, which are then fused
for the final story representation.

Usage:
    model = FiveComponentModel(config)
    story_embedding = model(story_data)  # [batch, output_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn("sentence-transformers not installed. Component 5 will be disabled.")

# Import our modules
try:
    from data_loader import DRSDataset, StoryGraph, Triplet
    from graph_encoder import GraphStructureEncoder, TemporalGraph
except ImportError:
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class FiveComponentConfig:
    """Configuration for five-component encoder."""

    # Component dimensions
    temporal_dim: int = 64          # Component 1 output
    logical_dim: int = 64           # Component 2 output
    participant_dim: int = 64       # Component 3 output
    semantic_dim: int = 64          # Component 4 output
    text_dim: int = 384             # Component 5 output (depends on model)

    # Model architecture
    hidden_dim: int = 128           # Hidden layer size
    output_dim: int = 128           # Final fused embedding size
    dropout: float = 0.1
    num_gnn_layers: int = 2

    # Text encoder
    text_model_name: str = 'all-MiniLM-L6-v2'

    # Feature sizes (set from dataset)
    num_event_types: int = 4        # State, Process, Transition, Unknown
    num_tenses: int = 4             # Past, Present, Future, Unknown
    num_aspects: int = 3            # Perfective, Progressive, Unknown
    num_vforms: int = 3             # Infinitive, Participle, Unknown
    num_temporal_relations: int = 4  # occursBefore, occursAfter, overlaps, during
    num_verbnet_classes: int = 3000  # Approximate
    num_predicates: int = 150        # Approximate

    # Which components to use
    use_temporal: bool = True
    use_logical: bool = True
    use_participant: bool = True
    use_semantic: bool = True
    use_text: bool = True

    def total_component_dim(self) -> int:
        """Calculate total dimension from enabled components."""
        dim = 0
        if self.use_temporal:
            dim += self.temporal_dim
        if self.use_logical:
            dim += self.logical_dim
        if self.use_participant:
            dim += self.participant_dim
        if self.use_semantic:
            dim += self.semantic_dim
        if self.use_text:
            dim += self.text_dim
        return dim


# ============================================================
# COMPONENT 1: TEMPORAL GRAPH ENCODER
# ============================================================

class TemporalGraphEncoder(nn.Module):
    """
    Component 1: Encodes events and their temporal relations.

    Input:
        - Event features (type, tense, aspect, vform, position)
        - Temporal edge graph (occursBefore, overlaps, etc.)

    Output:
        - Fixed-size temporal embedding [temporal_dim]

    Architecture:
        Event features -> Node embeddings
        GNN layers (message passing on temporal graph)
        -> Graph pooling -> Temporal embedding
    """

    def __init__(self, config: FiveComponentConfig):
        super().__init__()

        self.config = config

        # Event feature dimensions
        event_input_dim = (
            config.num_event_types +   # 4
            config.num_tenses +        # 4
            config.num_aspects +       # 3
            config.num_vforms +        # 3
            1                          # position (normalized)
        )  # = 15

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(event_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Edge embedding dimension
        edge_dim = config.hidden_dim // 4

        # GNN layers (with edge type awareness)
        self.gnn_layers = nn.ModuleList([
            TemporalGNNLayer(
                config.hidden_dim,
                config.dropout,
                edge_dim=edge_dim,
                num_edge_types=config.num_temporal_relations + 1,
            )
            for _ in range(config.num_gnn_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.temporal_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        event_features: torch.Tensor,      # [num_events, event_input_dim]
        adjacency: torch.Tensor,           # [num_events, num_events]
        edge_types: torch.Tensor,          # [num_events, num_events]
        mask: Optional[torch.Tensor] = None,  # [num_events] for batched
    ) -> torch.Tensor:
        """
        Forward pass for temporal graph encoding.

        Returns:
            Temporal embedding [temporal_dim] or [batch, temporal_dim]
        """
        # Encode node features
        h = self.node_encoder(event_features)  # [N, hidden_dim]

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adjacency, edge_types)

        # Graph pooling (mean over nodes)
        if mask is not None:
            # Masked mean for batched input
            mask_expanded = mask.unsqueeze(-1).float()
            h_masked = h * mask_expanded
            pooled = h_masked.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        else:
            pooled = h.mean(dim=0)  # [hidden_dim]

        # Project to output dimension
        output = self.output_projection(pooled)

        return output, h


class TemporalGNNLayer(nn.Module):
    """GNN layer for temporal graph with edge type awareness."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, edge_dim: int = 32, num_edge_types: int = 5):
        super().__init__()

        self.edge_dim = edge_dim

        # Edge type embedding
        self.edge_embedding = nn.Embedding(num_edge_types, edge_dim)

        # Message MLP now includes edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # [N, hidden_dim]
        adjacency: torch.Tensor,   # [N, N]
        edge_types: torch.Tensor,  # [N, N] edge type indices
    ) -> torch.Tensor:
        """Message passing on temporal graph with edge types."""

        N = h.size(0)

        # Get edge type embeddings for all pairs
        edge_emb = self.edge_embedding(edge_types)  # [N, N, edge_dim]

        # Compute messages: concat(h_i, h_j, edge_emb) for each edge
        h_i = h.unsqueeze(1).expand(N, N, -1)  # [N, N, hidden]
        h_j = h.unsqueeze(0).expand(N, N, -1)  # [N, N, hidden]

        messages = self.message_mlp(torch.cat([h_i, h_j, edge_emb], dim=-1))  # [N, N, hidden]

        # Aggregate messages weighted by adjacency
        adj_expanded = adjacency.unsqueeze(-1)  # [N, N, 1]
        aggregated = (messages * adj_expanded).sum(dim=1)  # [N, hidden]

        # Normalize by degree
        degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1)  # [N, 1]
        aggregated = aggregated / degree

        # Update node features
        h_new = self.update_mlp(torch.cat([h, aggregated], dim=-1))

        # Residual connection + layer norm
        h_out = self.layer_norm(h + h_new)

        return h_out


# ============================================================
# COMPONENT 2: LOGICAL LAYER ENCODER
# ============================================================

class LogicalLayerEncoder(nn.Module):
    """
    Component 2: Encodes logical/semantic features.

    Input:
        - Event types (State, Process, Transition)
        - VerbNet classes
        - Logic predicates (multi-hot)

    Output:
        - Fixed-size logical embedding [logical_dim]

    Architecture:
        Event type distribution + VerbNet embedding + Predicate encoding
        -> MLP -> Logical embedding
    """

    def __init__(self, config: FiveComponentConfig):
        super().__init__()

        self.config = config

        # Event type encoding (aggregate over all events)
        self.event_type_dim = config.num_event_types  # 4

        # VerbNet class embedding
        self.verbnet_embedding = nn.Embedding(
            config.num_verbnet_classes + 1,  # +1 for unknown
            config.hidden_dim // 4,
            padding_idx=0,
        )
        self.verbnet_dim = config.hidden_dim // 4

        # Predicate encoding (multi-hot -> dense)
        self.predicate_encoder = nn.Sequential(
            nn.Linear(config.num_predicates, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.predicate_dim = config.hidden_dim // 2

        # Combined MLP
        combined_dim = self.event_type_dim + self.verbnet_dim + self.predicate_dim

        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.logical_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        event_type_dist: torch.Tensor,   # [num_event_types] distribution
        verbnet_indices: torch.Tensor,   # [num_events] VerbNet class indices
        predicate_multihot: torch.Tensor,  # [num_predicates] multi-hot
    ) -> torch.Tensor:
        """
        Forward pass for logical layer encoding.

        Returns:
            Logical embedding [logical_dim]
        """
        # Event type distribution (already aggregated)
        type_features = event_type_dist  # [4]

        # VerbNet: embed and average
        vn_embedded = self.verbnet_embedding(verbnet_indices)  # [num_events, vn_dim]
        vn_pooled = vn_embedded.mean(dim=0)  # [vn_dim]

        # Predicate encoding
        pred_encoded = self.predicate_encoder(predicate_multihot)  # [pred_dim]

        # Combine and project
        combined = torch.cat([type_features, vn_pooled, pred_encoded], dim=-1)
        output = self.mlp(combined)

        return output


# ============================================================
# COMPONENT 3: PARTICIPANT GRAPH ENCODER
# ============================================================

class ParticipantGraphEncoder(nn.Module):
    """
    Component 3: Encodes actors and their coreference relations.

    Input:
        - Actor features (is_event_ref flag, text embedding)
        - Coreference graph (objIdentity relations)

    Output:
        - Fixed-size participant embedding [participant_dim]

    Architecture:
        Actor features -> Node embeddings
        GNN layers (message passing on coreference graph)
        -> Graph pooling -> Participant embedding
    """

    def __init__(self, config: FiveComponentConfig):
        super().__init__()

        self.text_reduction_dim = 64
        self.text_projection = nn.Linear(384, self.text_reduction_dim)

        self.config = config

        # Actor feature dimension (simple: just is_event_ref flag + position)
        actor_input_dim = 2 + config.text_dim # is_event_ref (1) + normalized position (1)

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(2 + self.text_reduction_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            ParticipantGNNLayer(config.hidden_dim, config.dropout)
            for _ in range(config.num_gnn_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.participant_dim),
            nn.ReLU(),
        )

    def forward(
            self,
            actor_features: torch.Tensor,  # [num_actors, 386]
            coref_adjacency: torch.Tensor, # [num_actors, num_actors]
            mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if actor_features.size(0) == 0:
            return torch.zeros(self.config.participant_dim, device=actor_features.device), \
                   torch.zeros((0, self.config.hidden_dim), device=actor_features.device)

        # 1. SPLIT the features
        # The first 2 are structural (is_event_ref, position)
        # The remaining 384 are the SBERT text embeddings
        struct_feat = actor_features[:, :2]
        text_feat = actor_features[:, 2:]

        # 2. PROJECT the text part from 384 down to 16
        # This prevents the text from overwhelming the structural signal
        projected_text = F.relu(self.text_projection(text_feat))

        # 3. COMBINE them back together
        # New shape will be [num_actors, 18] (which is 2 + 16)
        h_input = torch.cat([struct_feat, projected_text], dim=1)

        # 4. ENCODE using your Sequential node_encoder
        # Make sure you updated the input dimension in __init__ to 18!
        h = self.node_encoder(h_input)

        # 5. Apply GNN layers (unchanged)
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, coref_adjacency)

        # 6. Graph pooling (unchanged)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            h_masked = h * mask_expanded
            pooled = h_masked.sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        else:
            pooled = h.mean(dim=0)

        output = self.output_projection(pooled)

        return output, h


class ParticipantGNNLayer(nn.Module):
    """GNN layer for participant/coreference graph."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Message passing on coreference graph."""

        N = h.size(0)

        if N == 0:
            return h

        # Compute messages
        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)

        messages = self.message_mlp(torch.cat([h_i, h_j], dim=-1))

        # Aggregate
        adj_expanded = adjacency.unsqueeze(-1)
        aggregated = (messages * adj_expanded).sum(dim=1)

        degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1)
        aggregated = aggregated / degree

        # Update
        h_new = self.update_mlp(torch.cat([h, aggregated], dim=-1))
        h_out = self.layer_norm(h + h_new)

        return h_out


# ============================================================
# COMPONENT 4: SEMANTIC ROLE ENCODER
# ============================================================

class SemanticRoleEncoder(nn.Module):
    """
    Component 4: Encodes semantic relations between events and actors.

    Input:
        - Event nodes
        - Actor nodes
        - Semantic edges (agent, patient, theme, etc.)

    Output:
        - Fixed-size semantic embedding [semantic_dim]

    Architecture:
        Bipartite graph (events <-> actors)
        -> Bipartite GNN -> Graph pooling -> Semantic embedding
    """

    # Semantic role types
    ROLE_TYPES = ['agent', 'patient', 'theme', 'cause', 'other']

    def __init__(self, config: FiveComponentConfig):
        super().__init__()

        self.config = config

        # Role type embedding
        self.role_embedding = nn.Embedding(
            len(self.ROLE_TYPES) + 1,  # +1 for unknown
            config.hidden_dim // 4,
        )

        # Event-to-actor message passing
        self.event_to_actor = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Actor-to-event message passing
        self.actor_to_event = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.semantic_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        event_features: torch.Tensor,   # [num_events, hidden_dim]
        actor_features: torch.Tensor,   # [num_actors, hidden_dim]
        semantic_edges: List[Dict],     # [{actor_id, event_id, role}, ...]
        event_id_to_idx: Dict[str, int],
        actor_id_to_idx: Dict[str, int],
    ) -> torch.Tensor:
        """
        Forward pass for semantic role encoding.

        Returns:
            Semantic embedding [semantic_dim]
        """
        num_events = event_features.size(0)
        num_actors = actor_features.size(0)
        device = event_features.device

        if num_events == 0 or num_actors == 0 or len(semantic_edges) == 0:
            return torch.zeros(self.config.semantic_dim, device=device)

        # Build bipartite adjacency and role types
        # event_to_actor_adj[e, a] = 1 if edge exists
        event_to_actor_adj = torch.zeros(num_events, num_actors, device=device)
        role_indices = torch.zeros(num_events, num_actors, dtype=torch.long, device=device)

        for edge in semantic_edges:
            actor_id = edge.get('actor_id')
            event_id = edge.get('event_id')
            role = edge.get('role', 'other')

            if actor_id in actor_id_to_idx and event_id in event_id_to_idx:
                e_idx = event_id_to_idx[event_id]
                a_idx = actor_id_to_idx[actor_id]

                event_to_actor_adj[e_idx, a_idx] = 1.0
                role_idx = self.ROLE_TYPES.index(role) if role in self.ROLE_TYPES else len(self.ROLE_TYPES)
                role_indices[e_idx, a_idx] = role_idx

        # Get role embeddings
        role_emb = self.role_embedding(role_indices)  # [E, A, role_dim]

        # Event -> Actor message passing
        event_expanded = event_features.unsqueeze(1).expand(-1, num_actors, -1)  # [E, A, hidden]
        e2a_messages = torch.cat([event_expanded, role_emb], dim=-1)  # [E, A, hidden + role_dim]
        e2a_messages = self.event_to_actor(e2a_messages)  # [E, A, hidden]

        # Aggregate messages to actors
        adj_e2a = event_to_actor_adj.unsqueeze(-1)  # [E, A, 1]
        actor_aggregated = (e2a_messages * adj_e2a).sum(dim=0)  # [A, hidden]
        degree_a = event_to_actor_adj.sum(dim=0, keepdim=True).T.clamp(min=1)  # [A, 1]
        actor_aggregated = actor_aggregated / degree_a

        # Actor -> Event message passing
        actor_expanded = actor_features.unsqueeze(0).expand(num_events, -1, -1)  # [E, A, hidden]
        a2e_messages = torch.cat([actor_expanded, role_emb], dim=-1)
        a2e_messages = self.actor_to_event(a2e_messages)

        # Aggregate messages to events
        adj_a2e = event_to_actor_adj.unsqueeze(-1)
        event_aggregated = (a2e_messages * adj_a2e).sum(dim=1)  # [E, hidden]
        degree_e = event_to_actor_adj.sum(dim=1, keepdim=True).clamp(min=1)
        event_aggregated = event_aggregated / degree_e

        # Pool and combine
        event_pooled = event_aggregated.mean(dim=0)  # [hidden]
        actor_pooled = actor_aggregated.mean(dim=0)  # [hidden]

        combined = torch.cat([event_pooled, actor_pooled], dim=-1)
        output = self.output_projection(combined)

        return output


# ============================================================
# COMPONENT 5: TEXT ENCODER
# ============================================================

class TextEncoder(nn.Module):
    """
    Component 5: Encodes the full story text.

    Input:
        - Full story text string

    Output:
        - Text embedding [text_dim]

    Uses pre-trained sentence-transformer model.
    """

    def __init__(self, config: FiveComponentConfig, device: str = 'cpu'):
        super().__init__()

        self.config = config
        self.device = device

        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required for TextEncoder")

        # Load pre-trained model
        self.model = SentenceTransformer(config.text_model_name, device=device)
        self.text_dim = self.model.get_sentence_embedding_dimension()

        # Verify dimension matches config
        if self.text_dim != config.text_dim:
            print(f"Warning: Model text_dim ({self.text_dim}) != config ({config.text_dim})")
            print(f"Updating config.text_dim to {self.text_dim}")
            config.text_dim = self.text_dim

        # Optional projection layer
        self.projection = None

        # Freeze the sentence transformer by default
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text: str) -> torch.Tensor:
        """
        Encode story text.

        Args:
            text: Full story text string

        Returns:
            Text embedding [text_dim]
        """
        # Get embedding from sentence-transformer
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)

        if self.projection is not None:
            embedding = self.projection(embedding)

        return embedding

    def forward_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode batch of texts.

        Returns:
            Text embeddings [batch, text_dim]
        """
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings


# ============================================================
# MAIN MODEL: FIVE-COMPONENT FUSION
# ============================================================

class FiveComponentModel(nn.Module):
    """
    Main model combining all 5 components.

    Architecture:
        Story -> [Comp1, Comp2, Comp3, Comp4, Comp5] -> Concat -> Fusion MLP -> Embedding
    """

    def __init__(self, config: FiveComponentConfig, device: str = 'cpu'):
        super().__init__()

        self.config = config
        self.device = device

        # Initialize components
        if config.use_temporal:
            self.temporal_encoder = TemporalGraphEncoder(config)
        else:
            self.temporal_encoder = None

        if config.use_logical:
            self.logical_encoder = LogicalLayerEncoder(config)
        else:
            self.logical_encoder = None

        if config.use_participant:
            self.participant_encoder = ParticipantGraphEncoder(config)
        else:
            self.participant_encoder = None

        if config.use_semantic:
            self.semantic_encoder = SemanticRoleEncoder(config)
        else:
            self.semantic_encoder = None

        if config.use_text and HAS_SENTENCE_TRANSFORMERS:
            self.text_encoder = TextEncoder(config, device)
        else:
            self.text_encoder = None
            if config.use_text:
                print("Warning: Text encoder disabled (sentence-transformers not available)")
                config.use_text = False

        # Fusion layer
        total_dim = config.total_component_dim()

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.1))

        print(f"\n✓ FiveComponentModel initialized")
        print(f"  Component 1 (Temporal):    {'✓' if config.use_temporal else '✗'} -> {config.temporal_dim}d")
        print(f"  Component 2 (Logical):     {'✓' if config.use_logical else '✗'} -> {config.logical_dim}d")
        print(f"  Component 3 (Participant): {'✓' if config.use_participant else '✗'} -> {config.participant_dim}d")
        print(f"  Component 4 (Semantic):    {'✓' if config.use_semantic else '✗'} -> {config.semantic_dim}d")
        print(f"  Component 5 (Text):        {'✓' if config.use_text else '✗'} -> {config.text_dim}d")
        print(f"  Total input dim: {total_dim}")
        print(f"  Output dim: {config.output_dim}")

    def encode_story(self, story_data: Dict) -> torch.Tensor:
        """
        Encode a single story using all components.

        Args:
            story_data: Dict with all story features (from StoryDataProcessor)

        Returns:
            Story embedding [output_dim]
        """
        components = []

        # Component 1: Temporal
        if self.temporal_encoder is not None:
            temporal_emb, event_node_features = self.temporal_encoder(
                story_data['event_features'],
                story_data['temporal_adjacency'],
                story_data['temporal_edge_types'],
            )
            components.append(temporal_emb)

        # Component 2: Logical
        if self.logical_encoder is not None:
            logical_emb = self.logical_encoder(
                story_data['event_type_dist'],
                story_data['verbnet_indices'],
                story_data['predicate_multihot'],
            )
            components.append(logical_emb)

        # Component 3: Participant
        if self.participant_encoder is not None:
            #print(f"Actor feature shape: {story_data['actor_features'].shape}")
            participant_emb, actor_node_features = self.participant_encoder(
                story_data['actor_features'],
                story_data['coref_adjacency'],
            )
            components.append(participant_emb)

        # Component 4: Semantic
        if self.semantic_encoder is not None:
            # Need hidden representations from temporal encoder for events
            # For now, use event features directly
            #event_hidden = self.temporal_encoder.node_encoder(story_data['event_features'])
            #actor_hidden = self.participant_encoder.node_encoder(story_data['actor_features'])
            # Inside encode_story
            #print(f"Event node features shape: {event_node_features.shape}")  # Should be [num_events, hidden_dim]
            #print(f"Actor node features shape: {actor_node_features.shape}")  # Should be [num_actors, hidden_dim]

            semantic_emb = self.semantic_encoder(
                event_node_features,  # Contextualized by the temporal graph
                actor_node_features,  # Contextualized by character identity/coref
                story_data['semantic_edges'],
                story_data['event_id_to_idx'],
                story_data['actor_id_to_idx'],
            )
            components.append(semantic_emb)

        # Component 5: Text
        if self.text_encoder is not None:
            text_emb = self.text_encoder(story_data['text'])
            components.append(text_emb)

        # Concatenate all components
        combined = torch.cat(components, dim=-1)

        # Fusion
        output = self.fusion(combined)

        return output

    def forward(
        self,
        anchor_data: Dict,
        story_a_data: Dict,
        story_b_data: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for triplet comparison.

        Returns:
            Dict with embeddings and similarities
        """
        # Encode all three stories
        anchor_emb = self.encode_story(anchor_data)
        a_emb = self.encode_story(story_a_data)
        b_emb = self.encode_story(story_b_data)

        # Compute similarities (cosine)
        sim_a = F.cosine_similarity(anchor_emb.unsqueeze(0), a_emb.unsqueeze(0))
        sim_b = F.cosine_similarity(anchor_emb.unsqueeze(0), b_emb.unsqueeze(0))

        logits = (sim_a - sim_b) / self.temperature

        return {
            'anchor_emb': anchor_emb,
            'a_emb': a_emb,
            'b_emb': b_emb,
            'sim_a': sim_a,
            'sim_b': sim_b,
            'logits': logits,
        }


# ============================================================
# DATA PROCESSOR
# ============================================================

class StoryDataProcessor:
    """
    Processes StoryGraph into tensors for FiveComponentModel.

    Handles:
    - Event feature encoding
    - Graph construction
    - VerbNet/predicate vocabulary
    - Text extraction
    """

    # Fixed vocabularies
    EVENT_TYPES = ['State', 'Process', 'Transition', 'Unknown']
    TENSES = ['Past', 'Present', 'Future', 'Unknown']
    ASPECTS = ['Perfective', 'Progressive', 'Unknown']
    VFORMS = ['Infinitive', 'Participle', 'Unknown']

    TENSE_NORMALIZE = {'Pres': 'Present'}

    def __init__(self, dataset: Optional[DRSDataset] = None, text_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize processor.

        Args:
            dataset: DRSDataset to build vocabularies from
        """
        # Build vocabularies from dataset
        self.verbnet_vocab: Dict[str, int] = {'<UNK>': 0}
        self.predicate_vocab: Dict[str, int] = {}

        if dataset is not None:
            self._build_vocabularies(dataset)

        if HAS_SENTENCE_TRANSFORMERS:
            print(f"  Loading text model for processor: {text_model_name}")
            self.text_model = SentenceTransformer(text_model_name)
        else:
            self.text_model = None

        print(f"✓ StoryDataProcessor initialized")
        print(f"  VerbNet classes: {len(self.verbnet_vocab)}")
        print(f"  Logic predicates: {len(self.predicate_vocab)}")

    def _build_vocabularies(self, dataset: DRSDataset):
        """Build vocabularies from dataset."""
        # VerbNet classes
        vn_classes = dataset.get_all_verbnet_classes()
        for i, cls in enumerate(sorted(vn_classes)):
            self.verbnet_vocab[cls] = i + 1  # 0 is reserved for UNK

        # Predicates
        predicates = dataset.get_all_logic_predicates()
        for i, pred in enumerate(sorted(predicates)):
            self.predicate_vocab[pred] = i

    def _normalize_tense(self, tense: Optional[str]) -> str:
        if tense is None:
            return 'Unknown'
        return self.TENSE_NORMALIZE.get(tense, tense)

    def _one_hot(self, value: Optional[str], vocab: List[str]) -> torch.Tensor:
        if value is None or value not in vocab:
            idx = vocab.index('Unknown') if 'Unknown' in vocab else len(vocab) - 1
        else:
            idx = vocab.index(value)

        one_hot = torch.zeros(len(vocab))
        one_hot[idx] = 1.0
        return one_hot

    def process_story(
        self,
        story: 'StoryGraph',
        text: str = "",
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """
        Process a StoryGraph into tensors for the model.

        Returns:
            Dict with all required tensors
        """
        # ====== EVENT FEATURES ======
        event_features = []
        event_id_to_idx = {}

        for i, event in enumerate(story.events):
            event_id_to_idx[event['id']] = i

            # One-hot encodings
            type_oh = self._one_hot(event.get('type'), self.EVENT_TYPES)
            tense_oh = self._one_hot(self._normalize_tense(event.get('tense')), self.TENSES)
            aspect_oh = self._one_hot(event.get('aspect'), self.ASPECTS)
            vform_oh = self._one_hot(event.get('vform'), self.VFORMS)

            # Position (normalized)
            position = event.get('position', i) / max(len(story.events) - 1, 1)

            # Combine
            features = torch.cat([type_oh, tense_oh, aspect_oh, vform_oh, torch.tensor([position])])
            event_features.append(features)

        event_features = torch.stack(event_features) if event_features else torch.zeros(0, 15)

        # ====== TEMPORAL GRAPH ======
        num_events = len(story.events)
        temporal_adjacency = torch.zeros(num_events, num_events)
        temporal_edge_types = torch.zeros(num_events, num_events, dtype=torch.long)

        TEMPORAL_RELATIONS = ['occursBefore', 'occursAfter', 'overlaps', 'during']

        for edge in story.temporal_edges:
            src_id = edge['source']
            tgt_id = edge['target']
            rel = edge['relation']

            if src_id in event_id_to_idx and tgt_id in event_id_to_idx:
                src_idx = event_id_to_idx[src_id]
                tgt_idx = event_id_to_idx[tgt_id]

                temporal_adjacency[src_idx, tgt_idx] = 1.0
                rel_idx = TEMPORAL_RELATIONS.index(rel) + 1 if rel in TEMPORAL_RELATIONS else 0
                temporal_edge_types[src_idx, tgt_idx] = rel_idx

        # Add self-loops
        for i in range(num_events):
            temporal_adjacency[i, i] = 1.0

        # ====== EVENT TYPE DISTRIBUTION ======
        type_counts = torch.zeros(len(self.EVENT_TYPES))
        for event in story.events:
            event_type = event.get('type') or 'Unknown'
            if event_type in self.EVENT_TYPES:
                type_counts[self.EVENT_TYPES.index(event_type)] += 1

        event_type_dist = type_counts / max(type_counts.sum(), 1)

        # ====== VERBNET INDICES ======
        verbnet_indices = []
        for event in story.events:
            vn_class = story.verbnet_classes.get(event['id'], '<UNK>')
            idx = self.verbnet_vocab.get(vn_class, 0)
            verbnet_indices.append(idx)

        verbnet_indices = torch.tensor(verbnet_indices, dtype=torch.long)

        # ====== PREDICATE MULTI-HOT ======
        predicate_multihot = torch.zeros(len(self.predicate_vocab))
        for pred in story.logic_predicates:
            if pred in self.predicate_vocab:
                predicate_multihot[self.predicate_vocab[pred]] = 1.0

        # ====== ACTOR FEATURES ======
        actor_features = []
        actor_id_to_idx = {}
        for i, actor in enumerate(story.actors):
            actor_id_to_idx[actor['id']] = i
            is_event_ref = 1.0 if actor.get('is_event_ref', False) else 0.0
            position = i / max(len(story.actors) - 1, 1)

            if self.text_model is not None:
                actor_text_emb = self.text_model.encode(actor['text'], convert_to_tensor=True)
            else:
                # Fallback if transformer is missing
                actor_text_emb = torch.zeros(384)

            features = torch.cat([
                torch.tensor([is_event_ref, position], device=actor_text_emb.device),
                    actor_text_emb
            ])
            actor_features.append(features)
        text_dim = self.text_model.get_sentence_embedding_dimension()
        actor_features = torch.stack(actor_features) if actor_features else torch.zeros(0, 2 + 384)

        # ====== COREFERENCE GRAPH ======
        num_actors = len(story.actors)
        coref_adjacency = torch.zeros(num_actors, num_actors)

        for edge in story.coreference_edges:
            src_id = edge['source']
            tgt_id = edge['target']

            if src_id in actor_id_to_idx and tgt_id in actor_id_to_idx:
                src_idx = actor_id_to_idx[src_id]
                tgt_idx = actor_id_to_idx[tgt_id]

                coref_adjacency[src_idx, tgt_idx] = 1.0
                coref_adjacency[tgt_idx, src_idx] = 1.0  # Symmetric

        # Add self-loops
        for i in range(num_actors):
            coref_adjacency[i, i] = 1.0

        # ====== SEMANTIC EDGES ======
        semantic_edges = story.semantic_edges  # Already in correct format

        # Move to device
        return {
            'event_features': event_features.to(device),
            'temporal_adjacency': temporal_adjacency.to(device),
            'temporal_edge_types': temporal_edge_types.to(device),
            'event_type_dist': event_type_dist.to(device),
            'verbnet_indices': verbnet_indices.to(device),
            'predicate_multihot': predicate_multihot.to(device),
            'actor_features': actor_features.to(device),
            'coref_adjacency': coref_adjacency.to(device),
            'semantic_edges': semantic_edges,
            'event_id_to_idx': event_id_to_idx,
            'actor_id_to_idx': actor_id_to_idx,
            'text': text,
        }

    def process_triplet(
        self,
        triplet: 'Triplet',
        device: str = 'cpu',
    ) -> Tuple[Dict, Dict, Dict]:
        """Process a full triplet."""
        anchor_data = self.process_story(triplet.anchor, triplet.anchor_text, device)
        story_a_data = self.process_story(triplet.story_a, triplet.text_a, device)
        story_b_data = self.process_story(triplet.story_b, triplet.text_b, device)

        return anchor_data, story_a_data, story_b_data


# ============================================================
# VERIFICATION SCRIPT
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python five_component_encoder.py <preprocessed.jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "=" * 60)
    print("FIVE-COMPONENT ENCODER VERIFICATION")
    print("=" * 60)
    print(f"\nDevice: {device}")

    # Load dataset
    from data_loader import DRSDataset
    dataset = DRSDataset(jsonl_path)

    # Create data processor
    print("\n" + "-" * 40)
    print("DATA PROCESSOR")
    print("-" * 40)

    processor = StoryDataProcessor(dataset)

    # Create config
    config = FiveComponentConfig(
        num_verbnet_classes=len(processor.verbnet_vocab),
        num_predicates=len(processor.predicate_vocab),
    )

    # Create model
    print("\n" + "-" * 40)
    print("MODEL INITIALIZATION")
    print("-" * 40)

    model = FiveComponentModel(config, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test on first triplet
    print("\n" + "-" * 40)
    print("ENCODING TEST")
    print("-" * 40)

    triplet = dataset[0]
    anchor_data, story_a_data, story_b_data = processor.process_triplet(triplet, device)

    print(f"\nProcessed data shapes:")
    print(f"  Event features: {anchor_data['event_features'].shape}")
    print(f"  Temporal adjacency: {anchor_data['temporal_adjacency'].shape}")
    print(f"  Actor features: {anchor_data['actor_features'].shape}")
    print(f"  Coref adjacency: {anchor_data['coref_adjacency'].shape}")
    print(f"  VerbNet indices: {anchor_data['verbnet_indices'].shape}")
    print(f"  Predicate multi-hot: {anchor_data['predicate_multihot'].shape}")

    # Test individual components
    print("\n" + "-" * 40)
    print("COMPONENT-BY-COMPONENT TEST")
    print("-" * 40)

    model.eval()
    with torch.no_grad():
        # Component 1: Temporal
        if model.temporal_encoder:
            temp_emb = model.temporal_encoder(
                anchor_data['event_features'],
                anchor_data['temporal_adjacency'],
                anchor_data['temporal_edge_types'],
            )
            print(f"\n1. Temporal embedding: {temp_emb.shape}")
            print(f"   Values: mean={temp_emb.mean():.4f}, std={temp_emb.std():.4f}")

        # Component 2: Logical
        if model.logical_encoder:
            log_emb = model.logical_encoder(
                anchor_data['event_type_dist'],
                anchor_data['verbnet_indices'],
                anchor_data['predicate_multihot'],
            )
            print(f"\n2. Logical embedding: {log_emb.shape}")
            print(f"   Values: mean={log_emb.mean():.4f}, std={log_emb.std():.4f}")

        # Component 3: Participant
        if model.participant_encoder:
            part_emb = model.participant_encoder(
                anchor_data['actor_features'],
                anchor_data['coref_adjacency'],
            )
            print(f"\n3. Participant embedding: {part_emb.shape}")
            print(f"   Values: mean={part_emb.mean():.4f}, std={part_emb.std():.4f}")

        # Component 4: Semantic
        if model.semantic_encoder:
            event_hidden = model.temporal_encoder.node_encoder(anchor_data['event_features'])
            actor_hidden = model.participant_encoder.node_encoder(anchor_data['actor_features'])

            sem_emb = model.semantic_encoder(
                event_hidden,
                actor_hidden,
                anchor_data['semantic_edges'],
                anchor_data['event_id_to_idx'],
                anchor_data['actor_id_to_idx'],
            )
            print(f"\n4. Semantic embedding: {sem_emb.shape}")
            print(f"   Values: mean={sem_emb.mean():.4f}, std={sem_emb.std():.4f}")

        # Component 5: Text
        if model.text_encoder:
            text_emb = model.text_encoder(anchor_data['text'])
            print(f"\n5. Text embedding: {text_emb.shape}")
            print(f"   Values: mean={text_emb.mean():.4f}, std={text_emb.std():.4f}")

    # Test full forward pass
    print("\n" + "-" * 40)
    print("FULL FORWARD PASS TEST")
    print("-" * 40)

    with torch.no_grad():
        output = model(anchor_data, story_a_data, story_b_data)

    print(f"\nEmbedding shapes:")
    print(f"  Anchor: {output['anchor_emb'].shape}")
    print(f"  Story A: {output['a_emb'].shape}")
    print(f"  Story B: {output['b_emb'].shape}")

    print(f"\nSimilarities:")
    print(f"  Anchor-A: {output['sim_a'].item():.4f}")
    print(f"  Anchor-B: {output['sim_b'].item():.4f}")
    print(f"  Logits: {output['logits'].item():.4f}")

    print(f"\nPrediction: {'A closer' if output['logits'].item() > 0 else 'B closer'}")
    print(f"Ground truth: {'A closer' if triplet.label else 'B closer'}")

    # Test gradient flow
    print("\n" + "-" * 40)
    print("GRADIENT FLOW TEST")
    print("-" * 40)

    model.train()
    output = model(anchor_data, story_a_data, story_b_data)

    label = torch.tensor([1.0 if triplet.label else 0.0], device=device)
    loss = F.binary_cross_entropy_with_logits(output['logits'], label)

    loss.backward()

    print(f"Loss: {loss.item():.4f}")

    has_grads = True
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"  WARNING: No gradient for {name}")
            has_grads = False

    if has_grads:
        print("✓ All trainable parameters have gradients")

    print("\n" + "=" * 60)
    print("✓ FIVE-COMPONENT ENCODER VERIFICATION COMPLETE")
    print("=" * 60)