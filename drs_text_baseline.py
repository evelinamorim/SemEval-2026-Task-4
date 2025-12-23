"""
Text Baseline and Hybrid Approach for Narrative Similarity

Compares three approaches:
1. Text-only (SBERT embeddings)
2. DRS-only (structural features)
3. Hybrid (DRS + Text combined)

Updated version: Supports separate train/test datasets with different DRS directories
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
from xgboost import XGBClassifier

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

    def __init__(self,
                 train_drs_dir=None,
                 train_jsonl=None,
                 test_drs_dir=None,
                 test_jsonl=None,
                 text_model='all-MiniLM-L6-v2'):
        """
        Initialize hybrid evaluator with separate train/test datasets.

        Args:
            train_drs_dir: Directory with training DRS files (synthetic)
            train_jsonl: Path to training jsonl (synthetic)
            test_drs_dir: Directory with test DRS files (dev)
            test_jsonl: Path to test jsonl (dev)
            text_model: Sentence transformer model name

        Note: For backward compatibility (cross-validation mode),
              can also use: HybridEvaluator(test_drs_dir, test_jsonl)
        """
        self.train_drs_dir = train_drs_dir
        self.test_drs_dir = test_drs_dir
        self.text_baseline = TextBaseline(text_model)

        # Load datasets
        self.train_data = self._load_data(train_jsonl) if train_jsonl else None
        self.test_data = self._load_data(test_jsonl) if test_jsonl else None

        # For backward compatibility (single dataset mode for CV)
        if test_drs_dir and test_jsonl and not train_drs_dir:
            self.data = self.test_data
            self.drs_dir = test_drs_dir

        # Trained models
        self.trained_model = None
        self.scaler = None

        print(f"\n{'='*60}")
        print("HYBRID EVALUATOR INITIALIZED")
        print(f"{'='*60}")
        if self.train_data:
            print(f"Training data: {len(self.train_data)} instances")
            print(f"Training DRS dir: {train_drs_dir}")
        if self.test_data:
            print(f"Test data: {len(self.test_data)} instances")
            print(f"Test DRS dir: {test_drs_dir}")
        print(f"{'='*60}\n")

    def _load_data(self, jsonl_path):
        """Load all data from jsonl."""
        if jsonl_path is None:
            return None

        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                item['idx'] = idx
                data.append(item)
        return data

    def extract_drs_features(self, idx, dataset='test'):
        """
        Extract DRS features for a triplet.

        Args:
            idx: Instance index
            dataset: 'train' or 'test'

        Returns:
            dict or None if DRS files don't exist
        """
        # Select correct DRS directory
        if dataset == 'train':
            drs_dir = self.train_drs_dir
        elif dataset == 'test':
            drs_dir = self.test_drs_dir
        else:
            # For backward compatibility
            drs_dir = getattr(self, 'drs_dir', self.test_drs_dir)

        if drs_dir is None:
            return None

        # DRS file paths (both synthetic and dev use same naming: 0_anchor_drs.txt)
        anchor_path = os.path.join(drs_dir, f"{idx}_anchor_drs.txt")
        a_path = os.path.join(drs_dir, f"{idx}_a_drs.txt")
        b_path = os.path.join(drs_dir, f"{idx}_b_drs.txt")

        # Check if all files exist (skip if missing)
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

            # DEBUG: Check if new features exist
            # if idx < 3:  # Only print first 3 to avoid spam
            #    print(f"\nDEBUG idx {idx}:")
            #    print(f"  Keys in similarities: {sorted(drs_sims_a.keys())}")
            #    if 'event_sequence' in drs_sims_a:
            #        print(f"  event_sequence A: {drs_sims_a['event_sequence']:.3f}, B: {drs_sims_b['event_sequence']:.3f}")
            #    else:
            #        print(f"  event_sequence: NOT FOUND")
            #    if 'event_trigram' in drs_sims_a:
            #        print(f"  event_trigram A: {drs_sims_a['event_trigram']:.3f}, B: {drs_sims_b['event_trigram']:.3f}")
            #    else:
            #        print(f"  event_trigram: NOT FOUND")

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
            print(f"Warning: Error parsing DRS for idx {idx} in {dataset} set: {e}")
            return None

    def extract_text_features(self, idx, dataset='test'):
        """
        Extract text embedding features for a triplet.

        Args:
            idx: Instance index
            dataset: 'train' or 'test'

        Returns:
            dict with text similarities, or None if text fields are missing
        """
        # Select correct dataset
        if dataset == 'train':
            data_source = self.train_data
        elif dataset == 'test':
            data_source = self.test_data
        else:
            # For backward compatibility
            data_source = getattr(self, 'data', self.test_data)

        if data_source is None:
            return None

        item = data_source[idx]

        # Get text fields
        anchor_text = item.get('anchor_text')
        text_a = item.get('text_a')
        text_b = item.get('text_b')

        # Validate: skip if any text field is None or empty
        if not anchor_text or not text_a or not text_b:
            return None

        # Additional validation: check if strings are not just whitespace
        if not anchor_text.strip() or not text_a.strip() or not text_b.strip():
            return None

        try:
            # Compute text similarities
            sim_a = self.text_baseline.compute_similarity(anchor_text, text_a)
            sim_b = self.text_baseline.compute_similarity(anchor_text, text_b)

            return {
                'features': np.array([sim_a, sim_b]),
                'sim_a': sim_a,
                'sim_b': sim_b
            }
        except Exception as e:
            print(f"Warning: Error computing text similarity for idx {idx}: {e}")
            return None

    def extract_hybrid_features(self, idx, dataset='test'):
        """
        Extract combined DRS + Text features.

        Args:
            idx: Instance index
            dataset: 'train' or 'test'

        Returns:
            dict or None if DRS unavailable
        """
        drs_feat = self.extract_drs_features(idx, dataset)
        text_feat = self.extract_text_features(idx, dataset)

        if drs_feat is None or text_feat is None:
            return None

        # Concatenate features
        combined = np.concatenate([drs_feat['features'], text_feat['features']])

        return {
            'features': combined,
            'drs_features': drs_feat,
            'text_features': text_feat
        }

    def train_hybrid_model(self, classifier='random_forest', verbose=True):
        """
        Train classifier on synthetic training data.

        Args:
            classifier: 'logistic' or 'random_forest'
            verbose: Print progress

        Returns:
            dict: Training statistics
        """
        if self.train_data is None:
            raise ValueError("No training data loaded! Provide train_jsonl in __init__")

        if self.train_drs_dir is None:
            raise ValueError("No training DRS directory! Provide train_drs_dir in __init__")

        print("\n" + "=" * 60)
        print(f"TRAINING: HYBRID MODEL ({classifier.upper()})")
        print("=" * 60)

        # Collect features and labels
        X = []
        y = []
        skipped_drs = 0
        skipped_text = 0
        skipped_both = 0

        for idx, item in enumerate(self.train_data):
            # Check what's available
            drs_feat = self.extract_drs_features(idx, dataset='train')
            text_feat = self.extract_text_features(idx, dataset='train')

            # Track what's missing
            if drs_feat is None and text_feat is None:
                skipped_both += 1
                continue
            elif drs_feat is None:
                skipped_drs += 1
                continue
            elif text_feat is None:
                skipped_text += 1
                if verbose and idx < 10:  # Show first few for debugging
                    print(f"  Warning: Skipping idx {idx} - missing text fields")
                continue

            # Both available - combine features
            combined = np.concatenate([drs_feat['features'], text_feat['features']])

            X.append(combined)
            y.append(1 if item['text_a_is_closer'] else 0)  # 1 = A closer, 0 = B closer

            if verbose and (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(self.train_data)} instances...")

        if len(X) == 0:
            raise ValueError("No valid training instances! All instances skipped.")

        X = np.array(X)
        y = np.array(y)

        total_skipped = skipped_drs + skipped_text + skipped_both

        print(f"\n✓ Training instances: {len(X)}")
        print(f"✗ Skipped instances: {total_skipped}")
        if skipped_drs > 0:
            print(f"  - Missing DRS: {skipped_drs}")
        if skipped_text > 0:
            print(f"  - Missing text: {skipped_text}")
        if skipped_both > 0:
            print(f"  - Missing both: {skipped_both}")
        print(f"✓ Feature dimension: {X.shape[1]}")
        print(f"✓ Class distribution: {np.sum(y)} A-closer ({np.sum(y)/len(y)*100:.1f}%), "
              f"{len(y) - np.sum(y)} B-closer ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")

        # Normalize features
        print("\nNormalizing features...")
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        print(f"Training {classifier} classifier...")
        if classifier == 'logistic':
            self.trained_model = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier == 'random_forest':
            self.trained_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # Limit tree depth (default: None)
                min_samples_split=10,  # Need more samples to split (default: 2)
                min_samples_leaf=5,  # Need more samples in leaf (default: 1)
                max_features='sqrt',  # Use fewer features per split
                random_state=42
            )
        elif classifier == 'xgboost':
            self.trained_model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42
           )
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        self.trained_model.fit(X_scaled, y)

        # Training accuracy
        train_acc = self.trained_model.score(X_scaled, y)
        print(f"\n✓ Training accuracy: {train_acc:.3f}")

        return {
            'classifier': classifier,
            'n_instances': len(X),
            'n_features': X.shape[1],
            'train_accuracy': train_acc,
            'skipped_drs': skipped_drs,
            'skipped_text': skipped_text,
            'skipped_both': skipped_both,
            'skipped_total': total_skipped,
            'class_distribution': {
                'a_closer': int(np.sum(y)),
                'b_closer': int(len(y) - np.sum(y))
            }
        }

    def test_hybrid_model(self, verbose=True):
        """
        Test trained model on dev/test set.

        Args:
            verbose: Print detailed results

        Returns:
            dict: Test results
        """
        if self.trained_model is None:
            raise ValueError("No trained model! Call train_hybrid_model() first.")

        if self.test_data is None:
            raise ValueError("No test data loaded! Provide test_jsonl in __init__")

        if self.test_drs_dir is None:
            raise ValueError("No test DRS directory! Provide test_drs_dir in __init__")

        print("\n" + "=" * 60)
        print("TESTING: HYBRID MODEL ON DEV SET")
        print("=" * 60)

        results = []
        skipped_drs = 0
        skipped_text = 0
        skipped_both = 0

        for idx, item in enumerate(self.test_data):
            # Check what's available
            drs_feat = self.extract_drs_features(idx, dataset='test')
            text_feat = self.extract_text_features(idx, dataset='test')

            # Track what's missing
            if drs_feat is None and text_feat is None:
                skipped_both += 1
                if verbose:
                    print(f"⊘ [{idx}] Skipped (no DRS and no text)")
                continue
            elif drs_feat is None:
                skipped_drs += 1
                if verbose:
                    print(f"⊘ [{idx}] Skipped (no DRS)")
                continue
            elif text_feat is None:
                skipped_text += 1
                if verbose:
                    print(f"⊘ [{idx}] Skipped (no text)")
                continue

            # Both available - combine and predict
            combined = np.concatenate([drs_feat['features'], text_feat['features']])

            # Scale and predict
            X_test = self.scaler.transform([combined])
            prediction = self.trained_model.predict(X_test)[0]
            proba = self.trained_model.predict_proba(X_test)[0]

            predicted_a_is_closer = bool(prediction)
            ground_truth = item['text_a_is_closer']
            correct = predicted_a_is_closer == ground_truth

            results.append({
                'idx': idx,
                'predicted': predicted_a_is_closer,
                'ground_truth': ground_truth,
                'correct': correct,
                'confidence': float(max(proba))
            })

            if verbose:
                status = "✓" if correct else "✗"
                pred_str = "A" if predicted_a_is_closer else "B"
                truth_str = "A" if ground_truth else "B"
                conf = max(proba)
                print(f"{status} [{idx}] Pred={pred_str}, Truth={truth_str}, Conf={conf:.3f}")

        if len(results) == 0:
            print("ERROR: No test instances with complete features!")
            return None

        accuracy = sum(r['correct'] for r in results) / len(results)
        total_skipped = skipped_drs + skipped_text + skipped_both

        print(f"\n{'='*60}")
        print(f"✓ Test accuracy: {accuracy:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
        print(f"✓ Coverage: {len(results)}/{len(self.test_data)} instances")
        print(f"✗ Skipped: {total_skipped} instances")
        if skipped_drs > 0:
            print(f"  - Missing DRS: {skipped_drs}")
        if skipped_text > 0:
            print(f"  - Missing text: {skipped_text}")
        if skipped_both > 0:
            print(f"  - Missing both: {skipped_both}")
        print(f"{'='*60}")

        return {
            'method': 'hybrid_trained',
            'accuracy': accuracy,
            'results': results,
            'skipped_drs': skipped_drs,
            'skipped_text': skipped_text,
            'skipped_both': skipped_both,
            'skipped_total': total_skipped,
            'n_test': len(results),
            'n_total': len(self.test_data)
        }

    def analyze_errors(self, test_results, verbose=True):
        """
        Analyze prediction errors to find patterns.

        Args:
            test_results: Output from test_hybrid_model()
            verbose: Print detailed analysis

        Returns:
            dict: Error statistics
        """
        results = test_results['results']
        errors = [r for r in results if not r['correct']]
        correct = [r for r in results if r['correct']]

        print("\n" + "=" * 60)
        print("ERROR ANALYSIS")
        print("=" * 60)

        print(f"\nTotal instances: {len(results)}")
        print(f"Correct: {len(correct)} ({len(correct) / len(results) * 100:.1f}%)")
        print(f"Errors: {len(errors)} ({len(errors) / len(results) * 100:.1f}%)")

        # Confidence analysis
        low_conf_errors = [e for e in errors if e['confidence'] < 0.6]
        high_conf_errors = [e for e in errors if e['confidence'] >= 0.6]

        print(f"\nError breakdown by confidence:")
        print(f"  Low confidence (<0.6): {len(low_conf_errors)} - model was uncertain")
        print(f"  High confidence (≥0.6): {len(high_conf_errors)} - model was wrong confidently")

        if verbose and len(high_conf_errors) > 0:
            print(f"\nTop 10 high-confidence errors (most wrong):")
            high_conf_errors.sort(key=lambda x: x['confidence'], reverse=True)

            for i, e in enumerate(high_conf_errors[:10], 1):
                idx = e['idx']
                item = self.test_data[idx]

                print(f"\n{i}. Instance {idx}:")
                print(f"   Predicted: {'A closer' if e['predicted'] else 'B closer'} (conf: {e['confidence']:.3f})")
                print(f"   Truth: {'A closer' if e['ground_truth'] else 'B closer'}")
                print(f"   Anchor: {item['anchor_text'][:100]}...")
                print(f"   Story A: {item['text_a'][:100]}...")
                print(f"   Story B: {item['text_b'][:100]}...")

        return {
            'total_errors': len(errors),
            'low_conf_errors': len(low_conf_errors),
            'high_conf_errors': len(high_conf_errors),
            'error_rate': len(errors) / len(results),
            'error_indices': [e['idx'] for e in errors]
        }

    def analyze_feature_importance(self, plot=True):
        """
        Analyze feature importance from trained Random Forest model.

        Args:
            plot: Whether to save importance plot

        Returns:
            dict: Feature importance statistics
        """
        if self.trained_model is None:
            raise ValueError("No trained model! Call train_hybrid_model() first.")

        if not hasattr(self.trained_model, 'feature_importances_'):
            print("Model doesn't support feature importance (only works with tree-based models)")
            return None

        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        # Get importances
        importances = self.trained_model.feature_importances_

        # Build feature names (DRS features + text features)
        # Get DRS feature names from one sample
        sample_drs = self.extract_drs_features(0, 'train')
        if sample_drs is None:
            # Try test set
            sample_drs = self.extract_drs_features(0, 'test')

        drs_feature_names = sorted(sample_drs['feature_names']) if sample_drs else []

        feature_names = []
        for name in drs_feature_names:
            feature_names.append(f'drs_{name}_A')
        for name in drs_feature_names:
            feature_names.append(f'drs_{name}_B')
        feature_names.extend(['text_sim_A', 'text_sim_B'])

        # Sort by importance
        importance_data = list(zip(feature_names, importances))
        importance_data.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 10 Most Important Features:")
        for i, (name, imp) in enumerate(importance_data[:10], 1):
            print(f"{i:2d}. {name:30s}: {imp:.4f}")

        # Group by type
        drs_importance = sum(imp for name, imp in importance_data if name.startswith('drs_'))
        text_importance = sum(imp for name, imp in importance_data if name.startswith('text_'))

        print(f"\n{'=' * 60}")
        print(f"Total DRS importance:  {drs_importance:.4f} ({drs_importance * 100:.1f}%)")
        print(f"Total Text importance: {text_importance:.4f} ({text_importance * 100:.1f}%)")
        print(f"{'=' * 60}")

        # Plot
        if plot:
            try:
                import matplotlib.pyplot as plt

                names, imps = zip(*importance_data[:15])

                plt.figure(figsize=(10, 8))
                plt.barh(range(len(names)), imps)
                plt.yticks(range(len(names)), names)
                plt.xlabel('Importance')
                plt.title('Top 15 Feature Importances')
                plt.tight_layout()
                plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
                print("\n✓ Plot saved to feature_importance.png")
            except Exception as e:
                print(f"\nCouldn't create plot: {e}")

        return {
            'importances': dict(importance_data),
            'drs_total': drs_importance,
            'text_total': text_importance,
            'top_10': importance_data[:10]
        }

    def evaluate_text_only(self, dataset='test', verbose=False):
        """
        Evaluate text-only baseline.

        Args:
            dataset: 'train' or 'test'
            verbose: Print detailed results

        Returns:
            dict: Evaluation results
        """
        if dataset == 'train':
            data_source = self.train_data
            data_name = "TRAINING"
        else:
            data_source = self.test_data
            data_name = "TEST"

        if data_source is None:
            print(f"ERROR: No {dataset} data loaded!")
            return None

        print("\n" + "=" * 60)
        print(f"EVALUATING: TEXT-ONLY (SBERT) on {data_name} SET")
        print("=" * 60)

        results = []
        skipped = 0

        for idx, item in enumerate(data_source):
            text_feat = self.extract_text_features(idx, dataset)

            if text_feat is None:
                skipped += 1
                continue

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

        if len(results) == 0:
            print("ERROR: No valid text instances!")
            return None

        accuracy = sum(r['correct'] for r in results) / len(results)

        print(f"\nAccuracy: {accuracy:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
        if skipped > 0:
            print(f"Skipped: {skipped} instances with missing text")

        return {
            'method': 'text_only',
            'accuracy': accuracy,
            'results': results,
            'dataset': dataset,
            'skipped': skipped
        }

    def _compute_aggregate_score(self, drs_sims):
        """
        Compute aggregate similarity score from individual metrics.

        Uses weighted combination of all DRS similarity metrics.

        Args:
            drs_sims: Dictionary of similarity scores

        Returns:
            float: Aggregate similarity score
        """
        # Weights for different metrics (adjust based on importance)
        weights = {
            'cosine': 0.3,
            'event_type': 0.2,
            'temporal_relation': 0.15,
            'temporal_density': 0.1,
            'tense': 0.1,
            'event_count_ratio': 0.15
        }

        aggregate = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in drs_sims:
                aggregate += weight * drs_sims[metric]
                total_weight += weight

        # Normalize by actual total weight (in case some metrics are missing)
        if total_weight > 0:
            aggregate /= total_weight

        return aggregate

    def evaluate_drs_only(self, similarity_method='cosine', dataset='test', verbose=False):
        """
        Evaluate DRS-only baseline.

        Args:
            similarity_method: DRS similarity metric to use
                Valid options: 'cosine', 'event_type', 'temporal_relation',
                              'temporal_density', 'tense', 'event_count_ratio', 'aggregate'
            dataset: 'train' or 'test'
            verbose: Print detailed results

        Returns:
            dict: Evaluation results
        """
        if dataset == 'train':
            data_source = self.train_data
            data_name = "TRAINING"
        else:
            data_source = self.test_data
            data_name = "TEST"

        if data_source is None:
            print(f"ERROR: No {dataset} data loaded!")
            return None

        print("\n" + "=" * 60)
        print(f"EVALUATING: DRS-ONLY ({similarity_method}) on {data_name} SET")
        print("=" * 60)

        results = []

        for idx, item in enumerate(data_source):
            drs_feat = self.extract_drs_features(idx, dataset)

            if drs_feat is None:
                continue

            # Get similarity scores (handle 'aggregate' specially)
            if similarity_method == 'aggregate':
                score_a = self._compute_aggregate_score(drs_feat['sims_a'])
                score_b = self._compute_aggregate_score(drs_feat['sims_b'])
            else:
                # Check if method exists
                if similarity_method not in drs_feat['sims_a']:
                    available_methods = list(drs_feat['sims_a'].keys())
                    print(f"ERROR: Method '{similarity_method}' not found!")
                    print(f"Available methods: {available_methods}")
                    return None

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
        print(f"Coverage: {len(results)}/{len(data_source)} instances")

        return {
            'method': f'drs_only_{similarity_method}',
            'accuracy': accuracy,
            'results': results,
            'dataset': dataset
        }

    def evaluate_hybrid_simple(self, dataset='test', verbose=False):
        """
        Simple hybrid: average of DRS cosine + text similarity.

        Args:
            dataset: 'train' or 'test'
            verbose: Print detailed results

        Returns:
            dict: Evaluation results
        """
        if dataset == 'train':
            data_source = self.train_data
            data_name = "TRAINING"
        else:
            data_source = self.test_data
            data_name = "TEST"

        if data_source is None:
            print(f"ERROR: No {dataset} data loaded!")
            return None

        print("\n" + "=" * 60)
        print(f"EVALUATING: HYBRID (Simple Average) on {data_name} SET")
        print("=" * 60)

        results = []

        for idx, item in enumerate(data_source):
            hybrid_feat = self.extract_hybrid_features(idx, dataset)

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
        print(f"Coverage: {len(results)}/{len(data_source)} instances")

        return {
            'method': 'hybrid_simple',
            'accuracy': accuracy,
            'results': results,
            'dataset': dataset
        }

    def evaluate_hybrid_ml(self, classifier='logistic', cv_folds=5, verbose=False):
        """
        ML-based hybrid: train classifier on DRS + Text features (CROSS-VALIDATION MODE).

        NOTE: This is for backward compatibility. For train/test split, use:
              train_hybrid_model() + test_hybrid_model()

        Args:
            classifier: 'logistic' or 'random_forest'
            cv_folds: Number of cross-validation folds

        Returns:
            dict: Cross-validation results
        """
        # Use test data for CV (backward compatibility)
        data_source = self.test_data if self.test_data else self.train_data

        if data_source is None:
            raise ValueError("No data loaded for cross-validation!")

        print("\n" + "=" * 60)
        print(f"EVALUATING: HYBRID (ML - {classifier.upper()}) - CROSS-VALIDATION")
        print("=" * 60)

        # Collect features and labels
        X = []
        y = []
        indices = []

        dataset_type = 'test' if self.test_data else 'train'

        for idx, item in enumerate(data_source):
            hybrid_feat = self.extract_hybrid_features(idx, dataset_type)

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
        elif:
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else classifier == 'xgboost':
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42
           )

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

    def compare_all_approaches(self, dataset='test'):
        """
        Compare all approaches: Text, DRS, Hybrid (simple and ML).

        NOTE: For train/test evaluation, use train_hybrid_model() + test_hybrid_model()
              This method is for cross-validation on a single dataset.

        Args:
            dataset: 'train' or 'test'

        Returns:
            dict: Results for all approaches
        """
        results = {}

        print("\n" + "=" * 70)
        print(f"COMPREHENSIVE EVALUATION: TEXT vs DRS vs HYBRID (on {dataset.upper()} set)")
        print("=" * 70)

        # 1. Text-only
        results['text_only'] = self.evaluate_text_only(dataset)

        # 2. DRS-only (best methods from previous evaluation)
        results['drs_cosine'] = self.evaluate_drs_only('cosine', dataset)
        results['drs_event_type'] = self.evaluate_drs_only('event_type', dataset)

        # 3. Hybrid - Simple
        results['hybrid_simple'] = self.evaluate_hybrid_simple(dataset)

        # 4. Hybrid - ML (only if using single dataset for CV)
        if dataset == 'test' and self.train_data is None:
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

    print("\n" + "=" * 70)
    print("NARRATIVE SIMILARITY EVALUATION - TRAIN/TEST MODE")
    print("=" * 70)

    # Parse command line arguments
    if len(sys.argv) == 3:
        # Cross-validation mode (backward compatibility)
        print("\nMode: CROSS-VALIDATION (single dataset)")
        drs_dir = sys.argv[1]
        jsonl_path = sys.argv[2]

        evaluator = HybridEvaluator(
            test_drs_dir=drs_dir,
            test_jsonl=jsonl_path
        )

        # Compare all approaches
        results = evaluator.compare_all_approaches()

    elif len(sys.argv) == 5:
        # Train/test mode
        print("\nMode: TRAIN/TEST (separate datasets)")
        train_drs_dir = sys.argv[1]
        train_jsonl = sys.argv[2]
        test_drs_dir = sys.argv[3]
        test_jsonl = sys.argv[4]

        evaluator = HybridEvaluator(
            text_model='all-mpnet-base-v2',
            train_drs_dir=train_drs_dir,
            train_jsonl=train_jsonl,
            test_drs_dir=test_drs_dir,
            test_jsonl=test_jsonl
        )

        print("\n" + "=" * 70)
        print("STEP 1: EVALUATE BASELINES ON TEST SET")
        print("=" * 70)

        # Evaluate baselines on test set
        text_result = evaluator.evaluate_text_only(dataset='test')
        drs_cosine = evaluator.evaluate_drs_only('cosine', dataset='test')
        drs_aggregate = evaluator.evaluate_drs_only('aggregate', dataset='test')
        hybrid_simple = evaluator.evaluate_hybrid_simple(dataset='test')

        print("\n" + "=" * 70)
        print("STEP 2: TRAIN HYBRID MODEL ON SYNTHETIC DATA")
        print("=" * 70)

        # Train on synthetic data
        train_stats = evaluator.train_hybrid_model(classifier='xgboost')

        print("\n" + "=" * 70)
        print("STEP 3: TEST HYBRID MODEL ON DEV SET")
        print("=" * 70)

        # Test on dev set
        test_result = evaluator.test_hybrid_model(verbose=True)
        # Error analysis
        error_stats = evaluator.analyze_errors(test_result, verbose=True)
        # Feature importance analysis
        importance_stats = evaluator.analyze_feature_importance(plot=True)

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)

        results_summary = []
        if text_result:
            results_summary.append(('Text-only (SBERT)', text_result['accuracy']))
        if drs_cosine:
            results_summary.append(('DRS-only (cosine)', drs_cosine['accuracy']))
        if drs_aggregate:
            results_summary.append(('DRS-only (aggregate)', drs_aggregate['accuracy']))
        if hybrid_simple:
            results_summary.append(('Hybrid (simple avg)', hybrid_simple['accuracy']))
        if test_result:
            results_summary.append(('Hybrid (trained RF)', test_result['accuracy']))

        results_summary.sort(key=lambda x: x[1], reverse=True)

        for i, (name, acc) in enumerate(results_summary, 1):
            print(f"{i}. {name:25s}: {acc:.3f}")

    else:
        print("\nUsage:")
        print("\n1. Cross-validation mode (single dataset):")
        print("   python drs_text_baseline.py <drs_dir> <jsonl_path>")
        print("\n   Example:")
        print("   python drs_text_baseline.py ./data/drs ./data/dev_track_a.jsonl")

        print("\n2. Train/test mode (separate datasets):")
        print("   python drs_text_baseline.py <train_drs_dir> <train_jsonl> <test_drs_dir> <test_jsonl>")
        print("\n   Example:")
        print("   python drs_text_baseline.py ./drs_synthetic ./synthetic_train.jsonl ./drs_dev ./dev_track_a.jsonl")

        sys.exit(1)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
