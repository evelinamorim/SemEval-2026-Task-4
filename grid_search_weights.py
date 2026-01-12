"""
Grid Search for Optimal Hybrid Weights

Tests different combinations of:
1. DRS aggregate feature weights
2. Text vs DRS balance weights
3. Different DRS features to include

Evaluates on dev set using cross-validation or held-out split.
"""

import json
import numpy as np
from itertools import product
from collections import defaultdict
import os
import sys

# Import from your existing code
from drs_parser import parse_drs_file
from drs_similarity import DRSSimilarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class GridSearchOptimizer:
    """Grid search for optimal hybrid weights."""

    def __init__(self, drs_dir, jsonl_path, text_model='all-mpnet-base-v2'):
        """
        Initialize optimizer.

        Args:
            drs_dir: Directory with DRS files
            jsonl_path: Path to dev_track_a.jsonl with ground truth
            text_model: Sentence transformer model
        """
        self.drs_dir = drs_dir
        self.jsonl_path = jsonl_path

        print(f"Loading text model: {text_model}")
        self.text_model = SentenceTransformer(text_model)
        print("✓ Model loaded")

        # Load data
        self.data = self._load_data()
        print(f"✓ Loaded {len(self.data)} instances")

        # Pre-compute all features (expensive, do once)
        print("\nPre-computing features for all instances...")
        self.all_features = self._precompute_all_features()
        print(f"✓ Pre-computed features for {len(self.all_features)} instances")

    def _load_data(self):
        """Load ground truth from jsonl."""
        data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                item['idx'] = idx
                data.append(item)
        return data

    def _precompute_all_features(self):
        """Pre-compute DRS and text features for all instances."""
        all_features = {}

        for idx, item in enumerate(self.data):
            features = self._extract_features(idx, item)
            if features is not None:
                all_features[idx] = features

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(self.data)}...")

        return all_features

    def _extract_features(self, idx, item):
        """Extract all DRS and text features for one instance."""
        # DRS paths
        anchor_path = os.path.join(self.drs_dir, f"{idx}_anchor_drs.txt")
        a_path = os.path.join(self.drs_dir, f"{idx}_a_drs.txt")
        b_path = os.path.join(self.drs_dir, f"{idx}_b_drs.txt")

        if not all(os.path.exists(p) for p in [anchor_path, a_path, b_path]):
            return None

        try:
            # Parse DRS
            anchor_parser = parse_drs_file(anchor_path)
            a_parser = parse_drs_file(a_path)
            b_parser = parse_drs_file(b_path)

            # Get feature vectors
            anchor_feat = anchor_parser.get_feature_vector()
            a_feat = a_parser.get_feature_vector()
            b_feat = b_parser.get_feature_vector()

            # Compute DRS similarities
            sim_anchor_a = DRSSimilarity(anchor_parser, a_parser)
            sim_anchor_b = DRSSimilarity(anchor_parser, b_parser)

            drs_sims_a = sim_anchor_a.compute_all_similarities()
            drs_sims_b = sim_anchor_b.compute_all_similarities()

            # Node counts for ratio adjustment
            nodes_anchor = max(1, anchor_feat.get('event_count', 1))
            nodes_a = max(1, a_feat.get('event_count', 1))
            nodes_b = max(1, b_feat.get('event_count', 1))

            # Text features
            anchor_text = item.get('anchor_text', '')
            text_a = item.get('text_a', '')
            text_b = item.get('text_b', '')

            if not anchor_text or not text_a or not text_b:
                return None

            # Compute text embeddings
            emb_anchor = self.text_model.encode([anchor_text], convert_to_numpy=True)
            emb_a = self.text_model.encode([text_a], convert_to_numpy=True)
            emb_b = self.text_model.encode([text_b], convert_to_numpy=True)

            text_sim_a = cosine_similarity(emb_anchor, emb_a)[0][0]
            text_sim_b = cosine_similarity(emb_anchor, emb_b)[0][0]

            return {
                'drs_sims_a': drs_sims_a,
                'drs_sims_b': drs_sims_b,
                'text_sim_a': text_sim_a,
                'text_sim_b': text_sim_b,
                'nodes_anchor': nodes_anchor,
                'nodes_a': nodes_a,
                'nodes_b': nodes_b,
                'ground_truth': item['text_a_is_closer']
            }

        except Exception as e:
            print(f"  Warning: Error at idx {idx}: {e}")
            return None

    def _compute_aggregate(self, drs_sims, weights):
        """Compute weighted aggregate of DRS similarities."""
        total = 0.0
        weight_sum = 0.0

        for metric, weight in weights.items():
            if metric in drs_sims and weight > 0:
                total += weight * drs_sims[metric]
                weight_sum += weight

        return total / weight_sum if weight_sum > 0 else 0.0

    def _calculate_asymmetric_ratio(self, anchor_n, story_n):
        """Calculate asymmetric ratio (from your original code)."""
        if story_n == 0 or anchor_n == 0:
            return 0.0
        if story_n >= anchor_n:
            return 1.0 / (1.0 + np.log10(story_n / anchor_n))
        else:
            return story_n / anchor_n

    def evaluate_config(self, drs_weights, text_weight, struct_weight, use_ratio=True):
        """
        Evaluate a specific configuration.

        Args:
            drs_weights: Dict of {metric_name: weight} for DRS aggregate
            text_weight: Weight for text similarity (0-1)
            struct_weight: Weight for DRS similarity (0-1)
            use_ratio: Whether to apply asymmetric ratio adjustment

        Returns:
            accuracy, correct_count, total_count
        """
        correct = 0
        total = 0

        for idx, features in self.all_features.items():
            # Compute DRS aggregate scores
            drs_agg_a = self._compute_aggregate(features['drs_sims_a'], drs_weights)
            drs_agg_b = self._compute_aggregate(features['drs_sims_b'], drs_weights)

            # Apply ratio adjustment if enabled
            if use_ratio:
                ratio_a = self._calculate_asymmetric_ratio(
                    features['nodes_anchor'], features['nodes_a']
                )
                ratio_b = self._calculate_asymmetric_ratio(
                    features['nodes_anchor'], features['nodes_b']
                )
                drs_agg_a *= ratio_a
                drs_agg_b *= ratio_b

            # Get text similarities
            text_a = features['text_sim_a']
            text_b = features['text_sim_b']

            # Combined score
            score_a = text_weight * text_a + struct_weight * drs_agg_a
            score_b = text_weight * text_b + struct_weight * drs_agg_b

            # Prediction
            predicted_a_closer = score_a > score_b
            ground_truth = features['ground_truth']

            if predicted_a_closer == ground_truth:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total

    def grid_search(self):
        """
        Run comprehensive grid search.

        Returns:
            List of (config, accuracy) sorted by accuracy descending
        """
        results = []

        # ==========================================
        # DEFINE SEARCH SPACE
        # ==========================================

        # DRS feature combinations to try
        drs_feature_sets = [
            # Your current best
            {
                'logic_overlap': 2.0,
                'event_sequence': 1.0,
                'temporal_relation_semantic': 1.0,
            },
            # Add event_type
            {
                'logic_overlap': 2.0,
                'event_sequence': 1.0,
                'temporal_relation_semantic': 1.0,
                'event_type': 1.0,
            },
            # Add tense
            {
                'logic_overlap': 2.0,
                'event_sequence': 1.0,
                'temporal_relation_semantic': 1.0,
                'tense': 1.0,
            },
            # Add both event_type and tense
            {
                'logic_overlap': 2.0,
                'event_sequence': 1.0,
                'temporal_relation_semantic': 1.0,
                'event_type': 1.0,
                'tense': 0.5,
            },
            # Add cosine (general structural)
            {
                'logic_overlap': 2.0,
                'event_sequence': 1.0,
                'temporal_relation_semantic': 1.0,
                'cosine': 0.5,
            },
            # Full kitchen sink
            {
                'logic_overlap': 2.0,
                'event_sequence': 1.5,
                'temporal_relation_semantic': 1.0,
                'event_type': 1.0,
                'tense': 0.5,
                'cosine': 0.5,
            },
            # Logic-heavy
            {
                'logic_overlap': 3.0,
                'logic_sim': 1.0,
                'event_sequence': 1.0,
                'temporal_relation_semantic': 0.5,
            },
            # Temporal-heavy
            {
                'logic_overlap': 1.5,
                'event_sequence': 1.5,
                'temporal_relation_semantic': 2.0,
                'temporal_density': 1.0,
            },
            # Minimal (just logic_overlap)
            {
                'logic_overlap': 1.0,
            },
            # Event-focused
            {
                'logic_overlap': 2.0,
                'event_sequence': 2.0,
                'event_type': 1.5,
                'event_count_ratio': 0.5,
            },
            # With verbnet
            {
                'logic_overlap': 2.0,
                'event_sequence': 1.0,
                'temporal_relation_semantic': 1.0,
                'verbnet_distribution': 1.0,
            },
        ]

        # Text/Struct weight combinations
        weight_ratios = [
            (0.80, 0.20),  # Heavy text
            (0.75, 0.25),
            (0.70, 0.30),  # Your current
            (0.65, 0.35),
            (0.60, 0.40),
            (0.55, 0.45),
            (0.50, 0.50),  # Equal
            (0.45, 0.55),
            (0.40, 0.60),  # Heavy DRS
        ]

        # Whether to use ratio adjustment
        use_ratio_options = [True, False]

        # ==========================================
        # RUN GRID SEARCH
        # ==========================================

        total_configs = len(drs_feature_sets) * len(weight_ratios) * len(use_ratio_options)
        print(f"\nTesting {total_configs} configurations...")
        print("=" * 60)

        config_num = 0
        for drs_weights in drs_feature_sets:
            for text_w, struct_w in weight_ratios:
                for use_ratio in use_ratio_options:
                    config_num += 1

                    accuracy, correct, total = self.evaluate_config(
                        drs_weights, text_w, struct_w, use_ratio
                    )

                    config = {
                        'drs_weights': drs_weights,
                        'text_weight': text_w,
                        'struct_weight': struct_w,
                        'use_ratio': use_ratio,
                    }

                    results.append((config, accuracy, correct, total))

                    if config_num % 20 == 0:
                        print(f"  Progress: {config_num}/{total_configs}")

        # Sort by accuracy
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def print_top_results(self, results, top_n=20):
        """Print top N results."""
        print("\n" + "=" * 80)
        print(f"TOP {top_n} CONFIGURATIONS")
        print("=" * 80)

        for i, (config, accuracy, correct, total) in enumerate(results[:top_n], 1):
            print(f"\n{'─' * 80}")
            print(f"RANK {i}: Accuracy = {accuracy:.4f} ({correct}/{total})")
            print(f"{'─' * 80}")
            print(f"  Text weight:   {config['text_weight']:.2f}")
            print(f"  Struct weight: {config['struct_weight']:.2f}")
            print(f"  Use ratio:     {config['use_ratio']}")
            print(f"  DRS weights:")
            for metric, weight in config['drs_weights'].items():
                print(f"    - {metric}: {weight}")

        return results[:top_n]

    def analyze_trends(self, results):
        """Analyze which factors matter most."""
        print("\n" + "=" * 80)
        print("TREND ANALYSIS")
        print("=" * 80)

        # Group by text_weight
        by_text_weight = defaultdict(list)
        for config, acc, _, _ in results:
            by_text_weight[config['text_weight']].append(acc)

        print("\n📊 Average accuracy by text_weight:")
        for tw in sorted(by_text_weight.keys()):
            avg = np.mean(by_text_weight[tw])
            print(f"  text_weight={tw:.2f}: {avg:.4f}")

        # Group by use_ratio
        by_ratio = defaultdict(list)
        for config, acc, _, _ in results:
            by_ratio[config['use_ratio']].append(acc)

        print("\n📊 Average accuracy by use_ratio:")
        for ratio in [True, False]:
            avg = np.mean(by_ratio[ratio])
            print(f"  use_ratio={ratio}: {avg:.4f}")

        # Best DRS feature set
        drs_set_scores = defaultdict(list)
        for config, acc, _, _ in results:
            key = tuple(sorted(config['drs_weights'].keys()))
            drs_set_scores[key].append(acc)

        print("\n📊 Best DRS feature combinations (by avg accuracy):")
        sorted_drs = sorted(drs_set_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
        for features, accs in sorted_drs[:5]:
            print(f"  {features}: {np.mean(accs):.4f}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python grid_search_weights.py <drs_dir> <jsonl_path>")
        print("\nExample:")
        print("  python grid_search_weights.py ./data/drs ./data/dev_track_a.jsonl")
        sys.exit(1)

    drs_dir = sys.argv[1]
    jsonl_path = sys.argv[2]

    print("\n" + "=" * 80)
    print("GRID SEARCH FOR OPTIMAL HYBRID WEIGHTS")
    print("=" * 80)

    # Initialize optimizer
    optimizer = GridSearchOptimizer(drs_dir, jsonl_path)

    # Run grid search
    results = optimizer.grid_search()

    # Print top results
    top_results = optimizer.print_top_results(results, top_n=20)

    # Analyze trends
    optimizer.analyze_trends(results)

    # Print final recommendation
    best = results[0]
    print("\n" + "=" * 80)
    print("🏆 RECOMMENDED CONFIGURATION FOR SUBMISSION")
    print("=" * 80)
    print(f"\nBest accuracy: {best[1]:.4f} ({best[2]}/{best[3]})")
    print(f"\nConfiguration:")
    print(f"  w_text = {best[0]['text_weight']}")
    print(f"  w_struct = {best[0]['struct_weight']}")
    print(f"  use_ratio = {best[0]['use_ratio']}")
    print(f"\n  DRS aggregate weights:")
    print(f"  weights = {{")
    for metric, weight in best[0]['drs_weights'].items():
        print(f"      '{metric}': {weight},")
    print(f"  }}")

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()