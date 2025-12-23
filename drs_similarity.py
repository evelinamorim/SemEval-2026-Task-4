"""
DRS Similarity - Compare DRS representations for narrative similarity

Computes similarity between DRS pairs and evaluates predictions
against ground truth labels from dev_track_a.jsonl
"""

import json
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy
from drs_parser import parse_drs_file

from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = None
def _get_embedding_model():
    """Get cached embedding model (load once, reuse)."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print("Loading embedding model (one-time)...")
        _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
    return _EMBEDDING_MODEL

class DRSSimilarity:
    """Compute similarity between two DRS representations."""

    def __init__(self, drs1_features, drs2_features):
        """
        Initialize with feature dictionaries from two DRS files.

        Args:
            drs1_features: dict from parser.get_feature_vector()
            drs2_features: dict from parser.get_feature_vector()
        """
        self.feat1 = drs1_features
        self.feat2 = drs2_features

    def cosine_similarity(self, keys=None):
        """
        Cosine similarity between feature vectors.

        Args:
            keys: List of feature keys to use. If None, use all numeric features.

        Returns:
            float: Cosine similarity [0, 1] (1 = identical)
        """
        if keys is None:
            # Use all numeric features
            keys = [k for k, v in self.feat1.items()
                    if isinstance(v, (int, float))]

        vec1 = np.array([self.feat1.get(k, 0) for k in keys])
        vec2 = np.array([self.feat2.get(k, 0) for k in keys])

        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity = 1 - cosine distance
        return 1 - cosine(vec1, vec2)

    def euclidean_distance(self, keys=None, normalize=True):
        """
        Euclidean distance between feature vectors.

        Args:
            keys: List of feature keys to use
            normalize: If True, normalize features first

        Returns:
            float: Euclidean distance (lower = more similar)
        """
        if keys is None:
            keys = [k for k, v in self.feat1.items()
                    if isinstance(v, (int, float))]

        vec1 = np.array([self.feat1.get(k, 0) for k in keys], dtype=float)
        vec2 = np.array([self.feat2.get(k, 0) for k in keys], dtype=float)

        if normalize:
            # Min-max normalization
            max_vals = np.maximum(np.abs(vec1), np.abs(vec2))
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            vec1 = vec1 / max_vals
            vec2 = vec2 / max_vals

        return euclidean(vec1, vec2)

    def event_type_similarity(self):
        """
        Compare event type distributions using KL divergence.

        Returns:
            float: Similarity score [0, 1] (1 = identical distribution)
        """
        # Get event type counts
        types = ['state_count', 'process_count', 'transition_count']

        counts1 = np.array([self.feat1.get(t, 0) for t in types], dtype=float)
        counts2 = np.array([self.feat2.get(t, 0) for t in types], dtype=float)

        # Avoid division by zero
        if counts1.sum() == 0 or counts2.sum() == 0:
            return 0.0

        # Convert to probability distributions
        dist1 = counts1 / counts1.sum()
        dist2 = counts2 / counts2.sum()

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        dist1 = dist1 + epsilon
        dist2 = dist2 + epsilon
        dist1 = dist1 / dist1.sum()
        dist2 = dist2 / dist2.sum()

        # Jensen-Shannon divergence (symmetric version of KL)
        # JS divergence is between 0 and 1
        m = 0.5 * (dist1 + dist2)
        js_div = 0.5 * entropy(dist1, m) + 0.5 * entropy(dist2, m)

        # Convert to similarity: 1 - sqrt(JS divergence)
        return 1 - np.sqrt(js_div)

    def _extract_temporal_relation_tuples(self, features):
        """
        Extract temporal relations with actual event words.

        Returns:
            list: [(relation_type, event1_word, event2_word), ...]
        """
        # Get events and temporal relations
        events = features.get('events', [])
        temp_relations = features.get('temporal_relations_raw', [])

        if not events or not temp_relations:
            return []

        # Build mapping from event variable to event word
        var_to_word = {}
        for event in events:
            var = event.get('variable', '')
            word = event.get('text', event.get('word', ''))
            if var and word:
                var_to_word[var] = word

        # Extract ALL relation tuples (not just first few)
        tuples = []
        for rel in temp_relations:
            rel_type = rel.get('type', '')
            source = rel.get('source', '')
            target = rel.get('target', '')

            source_word = var_to_word.get(source, '')
            target_word = var_to_word.get(target, '')

            # Only add if both words exist and relation is occursBefore/After
            if source_word and target_word and rel_type in ['occursBefore', 'occursAfter']:
                tuples.append((rel_type, source_word, target_word))

        return tuples

    def temporal_relation_semantic_similarity(self):
        """Compare temporal relations using semantic similarity."""

        # Get temporal relations
        temp_rels_1 = self._extract_temporal_relation_tuples(self.feat1)
        temp_rels_2 = self._extract_temporal_relation_tuples(self.feat2)

        # DEBUG
        #print(f"  Relations extracted:")
        #print(f"    Story 1: {len(temp_rels_1)} total")
        #print(f"    Story 2: {len(temp_rels_2)} total")

        if len(temp_rels_1) == 0 or len(temp_rels_2) == 0:
            #print(f"  → Similarity: 0.0 (empty)")
            return 0.0

        # Use CACHED model (loaded once)
        model = _get_embedding_model()

        # Precompute embeddings for all event pairs
        phrases_1 = [f"{e1} {e2}" for _, e1, e2 in temp_rels_1]
        phrases_2 = [f"{e3} {e4}" for _, e3, e4 in temp_rels_2]

        emb_1 = model.encode(phrases_1)
        emb_2 = model.encode(phrases_2)

        # Compare: for each relation in story 1, find best match in story 2
        similarities = []

        for i, (rel1_type, _, _) in enumerate(temp_rels_1):
            max_sim = 0.0

            for j, (rel2_type, _, _) in enumerate(temp_rels_2):
                # Only compare same relation type
                if rel1_type != rel2_type:
                    continue

                # Cosine similarity
                sim = np.dot(emb_1[i], emb_2[j]) / (np.linalg.norm(emb_1[i]) * np.linalg.norm(emb_2[j]))
                max_sim = max(max_sim, sim)

            similarities.append(max_sim)

        result = np.mean(similarities) if similarities else 0.0
        #print(f"  → Similarity: {result:.3f}")
        return result

    def temporal_relation_similarity(self):
        """
        Jaccard similarity of temporal relation types.

        Compares which temporal relations are present (not counts).

        Returns:
            float: Jaccard similarity [0, 1]
        """
        temp_rel_keys = [
            'occurs_before_count', 'occurs_after_count',
            'overlaps_count', 'during_count', 'result_state_count'
        ]

        # Get set of present relations (count > 0)
        present1 = set(k for k in temp_rel_keys if self.feat1.get(k, 0) > 0)
        present2 = set(k for k in temp_rel_keys if self.feat2.get(k, 0) > 0)

        # Jaccard similarity
        if len(present1) == 0 and len(present2) == 0:
            return 1.0  # Both have no temporal relations

        intersection = len(present1 & present2)
        union = len(present1 | present2)

        return intersection / union if union > 0 else 0.0

    def temporal_density_similarity(self):
        """
        Compare temporal density (ratio of temporal relations to events).

        Returns:
            float: Similarity [0, 1] based on absolute difference
        """
        d1 = self.feat1.get('temporal_density', 0)
        d2 = self.feat2.get('temporal_density', 0)

        # Convert difference to similarity
        # Max difference is 1.0, so similarity = 1 - |diff|
        diff = abs(d1 - d2)
        return max(0, 1 - diff)

    def tense_similarity(self):
        """
        Compare tense distributions.

        Returns:
            float: Similarity [0, 1]
        """
        tense_keys = ['past_count', 'present_count', 'future_count']

        counts1 = np.array([self.feat1.get(k, 0) for k in tense_keys], dtype=float)
        counts2 = np.array([self.feat2.get(k, 0) for k in tense_keys], dtype=float)

        if counts1.sum() == 0 or counts2.sum() == 0:
            return 0.0

        dist1 = counts1 / counts1.sum()
        dist2 = counts2 / counts2.sum()

        # Cosine similarity of distributions
        return 1 - cosine(dist1, dist2)

    def event_sequence_similarity(self):
        """Compute similarity based on event sequences (bigrams)."""
        # FIX: Change features1 → feat1, features2 → feat2
        bigrams1 = self.feat1.get('event_bigrams', [])
        bigrams2 = self.feat2.get('event_bigrams', [])

        set1 = set(bigrams1) if bigrams1 else set()
        set2 = set(bigrams2) if bigrams2 else set()

        # if len(set1) == 0 and len(set2) == 0:
        #    return 1.0
        if len(set1 | set2) == 0:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def event_trigram_similarity(self):
        """Compute similarity based on event trigrams."""
        # FIX: Change features1 → feat1, features2 → feat2
        trigrams1 = self.feat1.get('event_trigrams', [])
        trigrams2 = self.feat2.get('event_trigrams', [])

        set1 = set(trigrams1) if trigrams1 else set()
        set2 = set(trigrams2) if trigrams2 else set()

        #if len(set1) == 0 and len(set2) == 0:
        #    return 1.0
        if len(set1 | set2) == 0:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


    def event_count_ratio(self):
        """
        Ratio of event counts (smaller/larger).

        Returns:
            float: Ratio [0, 1] (1 = same count)
        """
        c1 = self.feat1.get('event_count', 1)
        c2 = self.feat2.get('event_count', 1)

        return min(c1, c2) / max(c1, c2) if max(c1, c2) > 0 else 0.0

    def compute_all_similarities(self):
        """
        Compute all similarity metrics.

        Returns:
            dict: All similarity scores
        """
        return {
            'cosine': self.cosine_similarity(),
            'euclidean': self.euclidean_distance(),
            'event_type': self.event_type_similarity(),
            'temporal_relation': self.temporal_relation_similarity(),
            'temporal_density': self.temporal_density_similarity(),
            'tense': self.tense_similarity(),
            'event_count_ratio': self.event_count_ratio(),
            'event_sequence': self.event_sequence_similarity(),
            'event_trigram': self.event_trigram_similarity(),
            'temporal_relation_semantic': self.temporal_relation_semantic_similarity()
        }

    def aggregate_similarity(self, weights=None):
        """
        Weighted combination of all similarities.

        Args:
            weights: dict of weights for each metric. If None, use equal weights.

        Returns:
            float: Aggregated similarity score
        """
        sims = self.compute_all_similarities()

        if weights is None:
            # Default equal weights
            weights = {k: 1.0 for k in sims.keys()}

        # Normalize euclidean distance to [0, 1] similarity
        if 'euclidean' in sims:
            # Convert distance to similarity (assuming max distance is ~5)
            sims['euclidean'] = 1 / (1 + sims['euclidean'])

        # Weighted average
        total_weight = sum(weights.get(k, 0) for k in sims.keys())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(sims[k] * weights.get(k, 1.0) for k in sims.keys())
        return weighted_sum / total_weight


class TripletEvaluator:
    """Evaluate DRS similarity predictions against ground truth."""

    def __init__(self, drs_dir, jsonl_path):
        """
        Initialize evaluator.

        Args:
            drs_dir: Directory containing DRS files
            jsonl_path: Path to dev_track_a.jsonl with ground truth
        """
        self.drs_dir = drs_dir
        self.jsonl_path = jsonl_path
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self):
        """Load ground truth from jsonl file."""
        ground_truth = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                ground_truth.append({
                    'text_a_is_closer': data['text_a_is_closer']
                })
        return ground_truth

    def evaluate_triplet(self, idx, similarity_method='aggregate', weights=None):
        """
        Evaluate a single triplet.

        Args:
            idx: Triplet index (0, 1, 2, ...)
            similarity_method: Which similarity metric to use
            weights: Weights for aggregate method

        Returns:
            dict: Prediction and ground truth
        """
        import os

        # Load DRS files
        anchor_path = os.path.join(self.drs_dir, f"{idx}_anchor_drs.txt")
        a_path = os.path.join(self.drs_dir, f"{idx}_a_drs.txt")
        b_path = os.path.join(self.drs_dir, f"{idx}_b_drs.txt")

        # Check files exist
        if not all(os.path.exists(p) for p in [anchor_path, a_path, b_path]):
            return None

        # Parse DRS files
        anchor_parser = parse_drs_file(anchor_path)
        a_parser = parse_drs_file(a_path)
        b_parser = parse_drs_file(b_path)

        anchor_feat = anchor_parser.get_feature_vector()
        a_feat = a_parser.get_feature_vector()
        b_feat = b_parser.get_feature_vector()

        # Compute similarities
        sim_anchor_a = DRSSimilarity(anchor_feat, a_feat)
        sim_anchor_b = DRSSimilarity(anchor_feat, b_feat)

        # Get similarity scores
        if similarity_method == 'aggregate':
            score_a = sim_anchor_a.aggregate_similarity(weights)
            score_b = sim_anchor_b.aggregate_similarity(weights)
        else:
            sims_a = sim_anchor_a.compute_all_similarities()
            sims_b = sim_anchor_b.compute_all_similarities()
            score_a = sims_a.get(similarity_method, 0)
            score_b = sims_b.get(similarity_method, 0)

        # Predict: A is closer if score_a > score_b
        predicted_a_is_closer = score_a > score_b

        # Ground truth
        ground_truth_a_is_closer = self.ground_truth[idx]['text_a_is_closer']

        return {
            'idx': idx,
            'score_a': score_a,
            'score_b': score_b,
            'predicted_a_is_closer': predicted_a_is_closer,
            'ground_truth_a_is_closer': ground_truth_a_is_closer,
            'correct': predicted_a_is_closer == ground_truth_a_is_closer
        }

    def evaluate_all(self, similarity_method='aggregate', weights=None, verbose=False):
        """
        Evaluate all triplets.

        Args:
            similarity_method: Similarity metric to use
            weights: Weights for aggregate method
            verbose: If True, print detailed results

        Returns:
            dict: Evaluation results with accuracy
        """
        results = []

        for idx in range(len(self.ground_truth)):
            result = self.evaluate_triplet(idx, similarity_method, weights)
            if result:
                results.append(result)

                if verbose:
                    status = "✓" if result['correct'] else "✗"
                    print(f"{status} [{idx}] A={result['score_a']:.3f}, B={result['score_b']:.3f}, "
                          f"Pred={result['predicted_a_is_closer']}, "
                          f"GT={result['ground_truth_a_is_closer']}")

        # Calculate accuracy
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        accuracy = correct / total if total > 0 else 0.0

        return {
            'method': similarity_method,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }

    def compare_methods(self, methods=None):
        """
        Compare different similarity methods.

        Args:
            methods: List of method names. If None, test all methods.

        Returns:
            dict: Results for each method
        """
        if methods is None:
            methods = [
                'aggregate',
                'cosine',
                'event_type',
                'temporal_relation',
                'temporal_density',
                'tense',
                'event_count_ratio'
            ]

        results = {}

        print("=" * 60)
        print("COMPARING SIMILARITY METHODS")
        print("=" * 60)

        for method in methods:
            print(f"\nEvaluating: {method}")
            eval_result = self.evaluate_all(similarity_method=method)
            results[method] = eval_result
            print(f"  Accuracy: {eval_result['accuracy']:.3f} "
                  f"({eval_result['correct']}/{eval_result['total']})")

        # Rank by accuracy
        ranked = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print("\n" + "=" * 60)
        print("RANKING:")
        print("=" * 60)
        for i, (method, result) in enumerate(ranked, 1):
            print(f"{i}. {method:25s}: {result['accuracy']:.3f}")

        return results


# ========== Convenience Functions ==========

def compute_triplet_similarity(anchor_path, a_path, b_path, method='aggregate'):
    """
    Compute similarity for a single triplet.

    Args:
        anchor_path: Path to anchor DRS file
        a_path: Path to story A DRS file
        b_path: Path to story B DRS file
        method: Similarity method to use

    Returns:
        dict: Similarity scores and prediction
    """
    # Parse files
    anchor_parser = parse_drs_file(anchor_path)
    a_parser = parse_drs_file(a_path)
    b_parser = parse_drs_file(b_path)

    anchor_feat = anchor_parser.get_feature_vector()
    a_feat = a_parser.get_feature_vector()
    b_feat = b_parser.get_feature_vector()

    # Compute similarities
    sim_a = DRSSimilarity(anchor_feat, a_feat)
    sim_b = DRSSimilarity(anchor_feat, b_feat)

    if method == 'aggregate':
        score_a = sim_a.aggregate_similarity()
        score_b = sim_b.aggregate_similarity()
    else:
        sims_a = sim_a.compute_all_similarities()
        sims_b = sim_b.compute_all_similarities()
        score_a = sims_a.get(method, 0)
        score_b = sims_b.get(method, 0)

    return {
        'score_a': score_a,
        'score_b': score_b,
        'predicted_a_is_closer': score_a > score_b,
        'all_similarities_a': sim_a.compute_all_similarities(),
        'all_similarities_b': sim_b.compute_all_similarities()
    }


# ========== Testing ==========

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python drs_similarity.py <drs_dir> <jsonl_path>")
        print("\nExample:")
        print("  python drs_similarity.py ./data/drs ./data/dev_track_a.jsonl")
        sys.exit(1)

    drs_dir = sys.argv[1]
    jsonl_path = sys.argv[2]

    print("Evaluating DRS-based narrative similarity...")
    print(f"DRS directory: {drs_dir}")
    print(f"Ground truth: {jsonl_path}")

    evaluator = TripletEvaluator(drs_dir, jsonl_path)

    # Compare all methods
    results = evaluator.compare_methods()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)