"""
Step 2.3: Graph Structure Encoding

Converts temporal edges to graph structures:
- Adjacency matrix (dense or sparse)
- Edge type matrix (for typed GNNs)
- Edge index format (for PyTorch Geometric)

Usage:
    graph_encoder = GraphStructureEncoder()
    adj_matrix, edge_types = graph_encoder.encode_temporal_graph(story_graph)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Try to import torch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import from data_loader
try:
    from data_loader import DRSDataset, StoryGraph, Triplet
except ImportError:
    pass


@dataclass
class TemporalGraph:
    """Encoded temporal graph structure."""
    num_nodes: int  # Number of events
    adjacency: np.ndarray  # [num_nodes, num_nodes] adjacency matrix
    edge_types: np.ndarray  # [num_nodes, num_nodes] edge type indices
    edge_index: np.ndarray  # [2, num_edges] COO format
    edge_type_list: List[int]  # [num_edges] edge types in COO order
    node_ids: List[str]  # Event IDs in order (a, b, c, ...)

    def to_torch(self):
        """Convert to PyTorch tensors."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        return {
            'adjacency': torch.tensor(self.adjacency, dtype=torch.float32),
            'edge_types': torch.tensor(self.edge_types, dtype=torch.long),
            'edge_index': torch.tensor(self.edge_index, dtype=torch.long),
            'edge_type_list': torch.tensor(self.edge_type_list, dtype=torch.long),
        }


class GraphStructureEncoder:
    """
    Encodes temporal graph structure for GNN processing.

    Supports:
    - Dense adjacency matrix (for small graphs)
    - Sparse COO format (for PyTorch Geometric)
    - Edge type encoding (occursBefore=1, overlaps=2, etc.)
    """

    # Temporal relation types
    TEMPORAL_RELATIONS = ['occursBefore', 'occursAfter', 'overlaps', 'during']

    def __init__(self, add_self_loops: bool = True, bidirectional: bool = False):
        """
        Initialize graph encoder.

        Args:
            add_self_loops: Add self-loops to adjacency matrix
            bidirectional: Make edges bidirectional (for undirected GNNs)
        """
        self.add_self_loops = add_self_loops
        self.bidirectional = bidirectional

        # Edge type vocabulary (0 = no edge, 1+ = relation types)
        self.edge_type_vocab = {rel: i + 1 for i, rel in enumerate(self.TEMPORAL_RELATIONS)}

        print(f"✓ GraphStructureEncoder initialized")
        print(f"  Self-loops: {self.add_self_loops}")
        print(f"  Bidirectional: {self.bidirectional}")
        print(f"  Edge types: {self.edge_type_vocab}")

    def encode_temporal_graph(self, story: 'StoryGraph') -> TemporalGraph:
        """
        Encode temporal edges as graph structure.

        Args:
            story: StoryGraph object

        Returns:
            TemporalGraph with adjacency matrix and edge info
        """
        # Get ordered event IDs
        node_ids = story.event_ids  # ['a', 'b', 'c', ...]
        num_nodes = len(node_ids)

        # Create node ID to index mapping
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        # Initialize adjacency and edge type matrices
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        edge_types = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        # Edge lists for COO format
        edge_src = []
        edge_dst = []
        edge_type_list = []

        # Process temporal edges
        for edge in story.temporal_edges:
            src_id = edge['source']
            dst_id = edge['target']
            rel_type = edge['relation']

            # Skip if nodes not in our list (shouldn't happen, but safety check)
            if src_id not in node_to_idx or dst_id not in node_to_idx:
                continue

            src_idx = node_to_idx[src_id]
            dst_idx = node_to_idx[dst_id]
            edge_type_idx = self.edge_type_vocab.get(rel_type, 0)

            # Add edge
            adjacency[src_idx, dst_idx] = 1.0
            edge_types[src_idx, dst_idx] = edge_type_idx

            edge_src.append(src_idx)
            edge_dst.append(dst_idx)
            edge_type_list.append(edge_type_idx)

            # Add reverse edge if bidirectional
            if self.bidirectional:
                adjacency[dst_idx, src_idx] = 1.0
                edge_types[dst_idx, src_idx] = edge_type_idx

                edge_src.append(dst_idx)
                edge_dst.append(src_idx)
                edge_type_list.append(edge_type_idx)

        # Add self-loops
        if self.add_self_loops:
            for i in range(num_nodes):
                adjacency[i, i] = 1.0
                # Self-loops have edge type 0 (or could use a special type)

        # Create edge index in COO format [2, num_edges]
        if edge_src:
            edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        return TemporalGraph(
            num_nodes=num_nodes,
            adjacency=adjacency,
            edge_types=edge_types,
            edge_index=edge_index,
            edge_type_list=edge_type_list,
            node_ids=node_ids,
        )

    def encode_triplet(self, triplet: 'Triplet') -> Dict[str, TemporalGraph]:
        """
        Encode all three stories in a triplet.

        Returns:
            Dict with keys 'anchor', 'story_a', 'story_b'
        """
        return {
            'anchor': self.encode_temporal_graph(triplet.anchor),
            'story_a': self.encode_temporal_graph(triplet.story_a),
            'story_b': self.encode_temporal_graph(triplet.story_b),
        }

    def get_normalized_adjacency(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Compute normalized adjacency matrix (for GCN).

        D^(-1/2) * A * D^(-1/2)
        """
        # Degree matrix
        degree = adjacency.sum(axis=1)
        degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
        degree_inv_sqrt[degree == 0] = 0

        # D^(-1/2) as diagonal matrix
        d_inv_sqrt = np.diag(degree_inv_sqrt)

        # Normalized adjacency
        return d_inv_sqrt @ adjacency @ d_inv_sqrt

    def visualize_graph(self, temporal_graph: TemporalGraph, title: str = "Temporal Graph"):
        """
        Create a text visualization of the graph.
        """
        print(f"\n{title}")
        print("=" * 40)
        print(f"Nodes ({temporal_graph.num_nodes}): {temporal_graph.node_ids}")
        print(f"Edges ({len(temporal_graph.edge_type_list)}):")

        # Group edges by source
        edges_by_src = {}
        for i, (src, dst) in enumerate(zip(temporal_graph.edge_index[0],
                                           temporal_graph.edge_index[1])):
            src_id = temporal_graph.node_ids[src]
            dst_id = temporal_graph.node_ids[dst]
            edge_type = temporal_graph.edge_type_list[i]
            rel_name = self.TEMPORAL_RELATIONS[edge_type - 1] if edge_type > 0 else "unknown"

            if src_id not in edges_by_src:
                edges_by_src[src_id] = []
            edges_by_src[src_id].append((dst_id, rel_name))

        for src_id in sorted(edges_by_src.keys()):
            targets = edges_by_src[src_id]
            if len(targets) <= 3:
                print(f"  {src_id} -> {targets}")
            else:
                print(f"  {src_id} -> {targets[:3]}... (+{len(targets) - 3} more)")

        print(f"\nAdjacency matrix (5x5 corner):")
        print(temporal_graph.adjacency[:5, :5])


# ============================================================
# VERIFICATION SCRIPT
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graph_encoder.py <preprocessed.jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    print("\n" + "=" * 60)
    print("STEP 2.3: GRAPH STRUCTURE VERIFICATION")
    print("=" * 60 + "\n")

    # Load dataset
    from data_loader import DRSDataset

    dataset = DRSDataset(jsonl_path)

    # Create encoder
    print("\n" + "-" * 40)
    print("GRAPH ENCODER INITIALIZATION")
    print("-" * 40)

    graph_encoder = GraphStructureEncoder(
        add_self_loops=True,
        bidirectional=False,
    )

    # Test on first triplet
    print("\n" + "-" * 40)
    print("GRAPH ENCODING TEST")
    print("-" * 40)

    triplet = dataset[0]
    graphs = graph_encoder.encode_triplet(triplet)

    print(f"\nTriplet 0 graph structures:")
    for name, graph in graphs.items():
        print(f"\n  {name.upper()}:")
        print(f"    Nodes: {graph.num_nodes}")
        print(f"    Edges: {len(graph.edge_type_list)}")
        print(f"    Adjacency shape: {graph.adjacency.shape}")
        print(f"    Edge index shape: {graph.edge_index.shape}")
        print(f"    Edge density: {graph.adjacency.sum() / (graph.num_nodes ** 2):.3f}")

    # Visualize anchor graph
    graph_encoder.visualize_graph(graphs['anchor'], "ANCHOR TEMPORAL GRAPH")

    # Test normalized adjacency
    print("\n" + "-" * 40)
    print("NORMALIZED ADJACENCY TEST")
    print("-" * 40)

    norm_adj = graph_encoder.get_normalized_adjacency(graphs['anchor'].adjacency)
    print(f"Normalized adjacency shape: {norm_adj.shape}")
    print(f"Row sums (should be ~1): {norm_adj.sum(axis=1)[:5]}")

    # Batch statistics
    print("\n" + "-" * 40)
    print("BATCH STATISTICS")
    print("-" * 40)

    node_counts = []
    edge_counts = []
    densities = []

    for i in range(min(100, len(dataset))):
        graph = graph_encoder.encode_temporal_graph(dataset[i].anchor)
        node_counts.append(graph.num_nodes)
        edge_counts.append(len(graph.edge_type_list))
        density = graph.adjacency.sum() / (graph.num_nodes ** 2) if graph.num_nodes > 0 else 0
        densities.append(density)

    print(f"Nodes:   min={min(node_counts)}, max={max(node_counts)}, avg={np.mean(node_counts):.1f}")
    print(f"Edges:   min={min(edge_counts)}, max={max(edge_counts)}, avg={np.mean(edge_counts):.1f}")
    print(f"Density: min={min(densities):.3f}, max={max(densities):.3f}, avg={np.mean(densities):.3f}")

    # Edge type distribution
    print("\n" + "-" * 40)
    print("EDGE TYPE DISTRIBUTION")
    print("-" * 40)

    edge_type_counts = {rel: 0 for rel in graph_encoder.TEMPORAL_RELATIONS}

    for i in range(min(100, len(dataset))):
        graph = graph_encoder.encode_temporal_graph(dataset[i].anchor)
        for edge_type in graph.edge_type_list:
            if edge_type > 0 and edge_type <= len(graph_encoder.TEMPORAL_RELATIONS):
                rel = graph_encoder.TEMPORAL_RELATIONS[edge_type - 1]
                edge_type_counts[rel] += 1

    total_edges = sum(edge_type_counts.values())
    for rel, count in edge_type_counts.items():
        pct = count / total_edges * 100 if total_edges > 0 else 0
        print(f"  {rel}: {count} ({pct:.1f}%)")

    # Test PyTorch conversion if available
    if HAS_TORCH:
        print("\n" + "-" * 40)
        print("PYTORCH CONVERSION TEST")
        print("-" * 40)

        torch_graph = graphs['anchor'].to_torch()
        print(f"Adjacency: {torch_graph['adjacency'].shape}, dtype={torch_graph['adjacency'].dtype}")
        print(f"Edge index: {torch_graph['edge_index'].shape}, dtype={torch_graph['edge_index'].dtype}")
        print(f"Edge types: {torch_graph['edge_type_list'].shape}, dtype={torch_graph['edge_type_list'].dtype}")

    print("\n" + "=" * 60)
    print("✓ GRAPH STRUCTURE VERIFICATION COMPLETE")
    print("=" * 60)