"""
Text Baseline and Hybrid Approach for Narrative Similarity

Compares three approaches:
1. Text-only (SBERT embeddings)
2. DRS-only (structural features)
3. Hybrid (DRS + Text combined)
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import os

from drs_parser import parse_drs_file
from drs_similarity import DRSSimilarity


class TextBaseline:
    """Compute text-based similarity using sentence embeddings."""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize with a sentence transformer model.

        Args:
            model_name: HuggingFace model name. Options:
                - 'all-MiniLM-L6-v2' (fast, 384 dim)
                - 'all-mpnet-base-v2' (better, 768 dim)
                - 'paraphrase-multilingual-MiniLM-L12-v2' (multilingual)
        """
        print(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✓ Model loaded")

    def compute_similarity(self, text1, text2):
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            float: Cosine similarity [-1, 1]
        """
        emb1 = self.model.encode([text1], convert_to_numpy=True)
        emb2 = self.model.encode([text2], convert_to_numpy=True)

        return cosine_similarity(emb1, emb2)[0][0]

    def evaluate_triplet(self, anchor_text, text_a, text_b):
        """
        Evaluate a single triplet.

        Returns:
            dict: Similarity scores and prediction
        """
        sim_a = self.compute_similarity(anchor_text, text_a)
        sim_b = self.compute_similarity(anchor_text, text_b)

        return {
            'score_a': sim_a,
            'score_b': sim_b,
            'predicted_a_is_closer': sim_a > sim_b
        }


class HybridEvaluator:
    """Combines DRS and text features for narrative similarity."""

    def __init__(self, drs_dir, jsonl_path, text_model='all-MiniLM-L6-v2'):
        """
        Initialize hybrid evaluator.

        Args:
            drs_dir: Directory with DRS files
            jsonl_path: Path to dev_track_a.jsonl
            text_model: Sentence transformer model name
        """
        self.drs_dir = drs_dir
        self.jsonl_path = jsonl_path
        self.text_baseline = TextBaseline(text_model)

        # Load data
        self.data = self._load_data()

    def _load_data(self):
        """Load all data from jsonl."""
        data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                item['idx'] = idx
                data.append(item)
        return data

    def extract_drs_features(self, idx):
        """
        Extract DRS features for a triplet.

        Returns:
            dict or None if DRS files don't exist
        """
        anchor_path = os.path.join(self.drs_dir, f"{idx}_anchor_drs.txt")
        a_path = os.path.join(self.drs_dir, f"{idx}_a_drs.txt")
        b_path = os.path.join(self.drs_dir, f"{idx}_b_drs.txt")

        # Check if all files exist
        if not all(os.path.exists(p) for p in [anchor_path, a_path, b_path]):
            return None

        try:
            # Parse DRS files
            anchor_parser = parse_drs_file(anchor_path)
            a_parser = parse_drs_file(a_path)
            b_parser = parse_drs_file(b_path)

            anchor_feat = anchor_parser.get_feature_vector()
            a_feat = a_parser.get_feature_vector()
            b_feat = b_parser.get_feature_vector()

            # Compute DRS similarities
            sim_anchor_a = DRSSimilarity(anchor_feat, a_feat)
            sim_anchor_b = DRSSimilarity(anchor_feat, b_feat)

            # Get all similarity metrics
            drs_sims_a = sim_anchor_a.compute_all_similarities()
            drs_sims_b = sim_anchor_b.compute_all_similarities()

            # Create feature vector: [sim(anchor,A) for each metric, sim(anchor,B) for each metric]
            feature_names = sorted(drs_sims_a.keys())

            features = []
            for fname in feature_names:
                features.append(drs_sims_a[fname])
            for fname in feature_names:
                features.append(drs_sims_b[fname])

            return {
                'features': np.array(features),
                'feature_names': feature_names,
                'sims_a': drs_sims_a,
                'sims_b': drs_sims_b
            }
        except Exception as e:
            print(f"Warning: Error parsing DRS for idx {idx}: {e}")
            return None

    def extract_text_features(self, idx):
        """
        Extract text embedding features for a triplet.

        Returns:
            dict with text similarities
        """
        item = self.data[idx]

        anchor_text = item['anchor_text']
        text_a = item['text_a']
        text_b = item['text_b']

        # Compute text similarities
        sim_a = self.text_baseline.compute_similarity(anchor_text, text_a)
        sim_b = self.text_baseline.compute_similarity(anchor_text, text_b)

        return {
            'features': np.array([sim_a, sim_b]),
            'sim_a': sim_a,
            'sim_b': sim_b
        }

    def extract_hybrid_features(self, idx):
        """
        Extract combined DRS + Text features.

        Returns:
            dict or None if DRS unavailable
        """
        drs_feat = self.extract_drs_features(idx)
        text_feat = self.extract_text_features(idx)

        if drs_feat is None:
            return None

        # Concatenate features
        combined = np.concatenate([drs_feat['features'], text_feat['features']])

        return {
            'features': combined,
            'drs_features': drs_feat,
            'text_features': text_feat
        }

    def evaluate_text_only(self, verbose=False):
        """
        Evaluate text-only baseline.

        Returns:
            dict: Evaluation results
        """
        print("\n" + "=" * 60)
        print("EVALUATING: TEXT-ONLY (SBERT)")
        print("=" * 60)

        results = []

        for idx, item in enumerate(self.data):
            text_feat = self.extract_text_features(idx)

            predicted_a_is_closer = text_feat['sim_a'] > text_feat['sim_b']
            ground_truth = item['text_a_is_closer']
            correct = predicted_a_is_closer == ground_truth

            results.append({
                'idx': idx,
                'predicted': predicted_a_is_closer,
                'ground_truth': ground_truth,
                'correct': correct
            })

            if verbose:
                status = "✓" if correct else "✗"
                print(f"{status} [{idx}] A={text_feat['sim_a']:.3f}, "
                      f"B={text_feat['sim_b']:.3f}")

        accuracy = sum(r['correct'] for r in results) / len(results)

        print(f"\nAccuracy: {accuracy:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")

        return {
            'method': 'text_only',
            'accuracy': accuracy,
            'results': results
        }

    def evaluate_drs_only(self, similarity_method='cosine', verbose=False):
        """
        Evaluate DRS-only baseline.

        Returns:
            dict: Evaluation results
        """
        print("\n" + "=" * 60)
        print(f"EVALUATING: DRS-ONLY ({similarity_method})")
        print("=" * 60)

        results = []

        for idx, item in enumerate(self.data):
            drs_feat = self.extract_drs_features(idx)

            if drs_feat is None:
                continue

            score_a = drs_feat['sims_a'][similarity_method]
            score_b = drs_feat['sims_b'][similarity_method]

            predicted_a_is_closer = score_a > score_b
            ground_truth = item['text_a_is_closer']
            correct = predicted_a_is_closer == ground_truth

            results.append({
                'idx': idx,
                'predicted': predicted_a_is_closer,
                'ground_truth': ground_truth,
                'correct': correct
            })

            if verbose:
                status = "✓" if correct else "✗"
                print(f"{status} [{idx}] A={score_a:.3f}, B={score_b:.3f}")

        if len(results) == 0:
            print("No DRS files found!")
            return None

        accuracy = sum(r['correct'] for r in results) / len(results)

        print(f"\nAccuracy: {accuracy:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
        print(f"Coverage: {len(results)}/{len(self.data)} instances")

        return {
            'method': f'drs_only_{similarity_method}',
            'accuracy': accuracy,
            'results': results
        }

    def evaluate_hybrid_simple(self, verbose=False):
        """
        Simple hybrid: average of DRS cosine + text similarity.

        Returns:
            dict: Evaluation results
        """
        print("\n" + "=" * 60)
        print("EVALUATING: HYBRID (Simple Average)")
        print("=" * 60)

        results = []

        for idx, item in enumerate(self.data):
            hybrid_feat = self.extract_hybrid_features(idx)

            if hybrid_feat is None:
                continue

            # Simple average: (DRS_cosine + Text) / 2
            drs_a = hybrid_feat['drs_features']['sims_a']['cosine']
            drs_b = hybrid_feat['drs_features']['sims_b']['cosine']
            text_a = hybrid_feat['text_features']['sim_a']
            text_b = hybrid_feat['text_features']['sim_b']

            combined_a = (drs_a + text_a) / 2
            combined_b = (drs_b + text_b) / 2

            predicted_a_is_closer = combined_a > combined_b
            ground_truth = item['text_a_is_closer']
            correct = predicted_a_is_closer == ground_truth

            results.append({
                'idx': idx,
                'predicted': predicted_a_is_closer,
                'ground_truth': ground_truth,
                'correct': correct
            })

            if verbose:
                status = "✓" if correct else "✗"
                print(f"{status} [{idx}] A={combined_a:.3f}, B={combined_b:.3f}")

        if len(results) == 0:
            print("No DRS files found!")
            return None

        accuracy = sum(r['correct'] for r in results) / len(results)

        print(f"\nAccuracy: {accuracy:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
        print(f"Coverage: {len(results)}/{len(self.data)} instances")

        return {
            'method': 'hybrid_simple',
            'accuracy': accuracy,
            'results': results
        }

    def evaluate_hybrid_ml(self, classifier='logistic', cv_folds=5, verbose=False):
        """
        ML-based hybrid: train classifier on DRS + Text features.

        Args:
            classifier: 'logistic' or 'random_forest'
            cv_folds: Number of cross-validation folds

        Returns:
            dict: Cross-validation results
        """
        print("\n" + "=" * 60)
        print(f"EVALUATING: HYBRID (ML - {classifier.upper()})")
        print("=" * 60)

        # Collect features and labels
        X = []
        y = []
        indices = []

        for idx, item in enumerate(self.data):
            hybrid_feat = self.extract_hybrid_features(idx)

            if hybrid_feat is None:
                continue

            X.append(hybrid_feat['features'])
            y.append(1 if item['text_a_is_closer'] else 0)  # 1 = A closer, 0 = B closer
            indices.append(idx)

        if len(X) < cv_folds:
            print(f"Not enough instances ({len(X)}) for {cv_folds}-fold CV!")
            return None

        X = np.array(X)
        y = np.array(y)

        print(f"Dataset: {len(X)} instances with {X.shape[1]} features")

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Choose classifier
        if classifier == 'logistic':
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

        print(f"\nCross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

        return {
            'method': f'hybrid_ml_{classifier}',
            'accuracy': scores.mean(),
            'std': scores.std(),
            'cv_scores': scores.tolist(),
            'n_instances': len(X),
            'n_features': X.shape[1]
        }

    def compare_all_approaches(self):
        """
        Compare all approaches: Text, DRS, Hybrid (simple and ML).

        Returns:
            dict: Results for all approaches
        """
        results = {}

        print("\n" + "=" * 70)
        print("COMPREHENSIVE EVALUATION: TEXT vs DRS vs HYBRID")
        print("=" * 70)

        # 1. Text-only
        results['text_only'] = self.evaluate_text_only()

        # 2. DRS-only (best method from previous evaluation)
        results['drs_cosine'] = self.evaluate_drs_only('cosine')
        results['drs_event_type'] = self.evaluate_drs_only('event_type')

        # 3. Hybrid - Simple
        results['hybrid_simple'] = self.evaluate_hybrid_simple()

        # 4. Hybrid - ML
        results['hybrid_logistic'] = self.evaluate_hybrid_ml('logistic', cv_folds=5)
        results['hybrid_rf'] = self.evaluate_hybrid_ml('random_forest', cv_folds=5)

        # Summary
        print("\n" + "=" * 70)
        print("FINAL RANKING")
        print("=" * 70)

        ranking = []
        for name, result in results.items():
            if result is not None:
                ranking.append((name, result['accuracy']))

        ranking.sort(key=lambda x: x[1], reverse=True)

        for i, (name, acc) in enumerate(ranking, 1):
            print(f"{i}. {name:25s}: {acc:.3f}")

        return results


# ========== Main Execution ==========

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python drs_text_baseline.py <drs_dir> <jsonl_path>")
        print("\nExample:")
        print("  python drs_text_baseline.py ./data/drs ./data/dev_track_a.jsonl")
        sys.exit(1)

    drs_dir = sys.argv[1]
    jsonl_path = sys.argv[2]

    # Create evaluator
    evaluator = HybridEvaluator(drs_dir, jsonl_path)

    # Compare all approaches
    results = evaluator.compare_all_approaches()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)