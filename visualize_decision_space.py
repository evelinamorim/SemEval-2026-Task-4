#!/usr/bin/env python3
"""
Decision Space Visualization for Narrative Similarity Task

This script visualizes whether your DRS features can discriminate between
correct and incorrect story matches.

Usage:
    python visualize_decision_space.py --jsonl data/dev_track_a.jsonl --drs-dir data/drs/
    python visualize_decision_space.py --jsonl data/dev_track_a.jsonl --drs-dir data/drs/ --output results/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules - adjust paths as needed
import sys
sys.path.append('.')

try:
    from drs_parser import DRSParser
    from drs_similarity import DRSSimilarity
    print("✓ Successfully imported DRS modules")
except ImportError as e:
    print(f"✗ Error importing DRS modules: {e}")
    print("  Make sure drs_parser.py and drs_similarity.py are in the current directory")
    sys.exit(1)


def load_jsonl_data(jsonl_path):
    """
    Load data from JSONL file.
    
    Expected format per line:
    {
        "anchor_text": "...",
        "text_a": "...",
        "text_b": "...",
        "text_a_is_closer": true/false
    }
    """
    data = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                
                # Convert label to 1/0
                label = 1 if item.get('text_a_is_closer', False) else 0
                
                data.append({
                    'idx': idx,
                    'anchor_text': item.get('anchor_text', ''),
                    'text_a': item.get('text_a', ''),
                    'text_b': item.get('text_b', ''),
                    'label': label,  # 1 = A is correct, 0 = B is correct
                })
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping line {idx} - JSON error: {e}")
                continue
    
    print(f"✓ Loaded {len(data)} instances from {jsonl_path}")
    return data


def load_drs_triplet(idx, drs_dir):
    """
    Load DRS files for a triplet.
    
    File naming convention:
    - {idx}_anchor_drs.txt
    - {idx}_a_drs.txt
    - {idx}_b_drs.txt
    """
    drs_dir = Path(drs_dir)
    
    anchor_path = drs_dir / f"{idx}_anchor_drs.txt"
    a_path = drs_dir / f"{idx}_a_drs.txt"
    b_path = drs_dir / f"{idx}_b_drs.txt"
    
    anchor_drs, a_drs, b_drs = None, None, None
    
    try:
        if anchor_path.exists():
            anchor_drs = DRSParser(str(anchor_path))
            anchor_drs.parse()
        else:
            print(f"  Warning: Anchor DRS not found: {anchor_path}")
    except Exception as e:
        print(f"  Warning: Error loading anchor DRS {anchor_path}: {e}")
    
    try:
        if a_path.exists():
            a_drs = DRSParser(str(a_path))
            a_drs.parse()
        else:
            print(f"  Warning: Story A DRS not found: {a_path}")
    except Exception as e:
        print(f"  Warning: Error loading Story A DRS {a_path}: {e}")
    
    try:
        if b_path.exists():
            b_drs = DRSParser(str(b_path))
            b_drs.parse()
        else:
            print(f"  Warning: Story B DRS not found: {b_path}")
    except Exception as e:
        print(f"  Warning: Error loading Story B DRS {b_path}: {e}")
    
    return anchor_drs, a_drs, b_drs


def compute_similarity_features(drs1, drs2, feature_names=None):
    """
    Compute similarity features between two DRS objects.
    Returns all features from compute_all_similarities().
    """
    try:
        sim = DRSSimilarity(drs1, drs2)
        scores = sim.compute_all_similarities()
        
        # Debug: print available features on first call
        if not hasattr(compute_similarity_features, '_printed_features'):
            print(f"  Available features: {list(scores.keys())}")
            print(f"  Sample values: {scores}")
            compute_similarity_features._printed_features = True
        
        # If specific features requested, filter; otherwise return all
        if feature_names is not None:
            result = {}
            for f in feature_names:
                if f in scores:
                    result[f] = scores[f]
                else:
                    print(f"  Warning: Feature '{f}' not found in scores")
                    result[f] = 0.0
            return result
        else:
            return scores
    except Exception as e:
        print(f"  Warning: Error computing similarity: {e}")
        import traceback
        traceback.print_exc()
        if feature_names:
            return {f: 0.0 for f in feature_names}
        return {}


def get_available_features(drs_dir, sample_idx=0):
    """
    Get available feature names by running one similarity computation.
    """
    anchor_drs, a_drs, b_drs = load_drs_triplet(sample_idx, drs_dir)
    
    if anchor_drs is None or a_drs is None:
        print(f"  Could not load DRS files for sample {sample_idx}")
        return None
    
    try:
        sim = DRSSimilarity(anchor_drs, a_drs)
        scores = sim.compute_all_similarities()
        print(f"\n  Available features from compute_all_similarities():")
        for name, value in sorted(scores.items()):
            print(f"    - {name}: {value:.4f}")
        return list(scores.keys())
    except Exception as e:
        print(f"  Error getting features: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_difference_features(data, drs_dir, feature_names=None):
    """
    Extract difference features for all instances.
    
    For each instance: features = (anchor-A similarity) - (anchor-B similarity)
    
    Returns:
        X: numpy array of shape (n_instances, n_features)
        y: numpy array of labels (1 if A is correct, 0 if B is correct)
        feature_names: list of feature names
        valid_indices: indices of successfully processed instances
    """
    # Auto-detect available features if not specified
    if feature_names is None:
        print("  Auto-detecting available features...")
        feature_names = get_available_features(drs_dir, sample_idx=0)
        
        if feature_names is None:
            print("  Error: Could not detect features. Using defaults from transcript.")
            # These are the actual features from compute_all_similarities() based on transcript
            feature_names = [
                'cosine', 'event_type', 'temporal_relation', 
                'temporal_density', 'tense', 'event_count_ratio', 'event_sequence', 
                'event_trigram', 'temporal_relation_semantic', 'verbnet_distribution', 
                'logic_sim', 'logic_overlap'
            ]
        else:
            # Filter out 'euclidean' - it's a distance, not a similarity
            if 'euclidean' in feature_names:
                feature_names = [f for f in feature_names if f != 'euclidean']
                print("  (Excluded 'euclidean' - it's a distance, not a similarity)")
            print(f"  Using {len(feature_names)} detected features")
    
    X = []
    y = []
    valid_indices = []
    
    print(f"\nExtracting features from {len(data)} instances...")
    
    for i, item in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i + 1}/{len(data)}...")
        
        idx = item['idx']
        
        # Load DRS files
        anchor_drs, a_drs, b_drs = load_drs_triplet(idx, drs_dir)
        
        if anchor_drs is None or a_drs is None or b_drs is None:
            continue
        
        # Compute similarities
        scores_a = compute_similarity_features(anchor_drs, a_drs, feature_names)
        scores_b = compute_similarity_features(anchor_drs, b_drs, feature_names)
        
        # Debug: print first instance scores
        if i == 0:
            print(f"\n  First instance scores_a: {scores_a}")
            print(f"  First instance scores_b: {scores_b}")
        
        # Difference features: positive means A scores higher
        # Use .get() with default 0.0 for safety
        diff = [scores_a.get(f, 0.0) - scores_b.get(f, 0.0) for f in feature_names]
        
        # Debug: print first instance diff
        if i == 0:
            print(f"  First instance diff: {diff}\n")
        
        X.append(diff)
        y.append(item['label'])
        valid_indices.append(idx)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Successfully extracted features for {len(X)} instances")
    
    # Check for NaN/Inf values
    nan_count = np.sum(np.isnan(X))
    inf_count = np.sum(np.isinf(X))
    if nan_count > 0 or inf_count > 0:
        print(f"\n  ⚠ Found {nan_count} NaN and {inf_count} Inf values in features!")
        print("  Replacing with 0.0...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Debug: check for all-zero features
    if len(X) > 0:
        feature_sums = np.abs(X).sum(axis=0)
        print(f"\n  Feature activity (sum of absolute values):")
        for fname, fsum in zip(feature_names, feature_sums):
            status = "✓" if fsum > 0 else "✗ ZERO"
            print(f"    {fname}: {fsum:.4f} {status}")
    
    return X, y, feature_names, valid_indices


def compute_separability_metrics(X, y):
    """
    Compute metrics that indicate how separable the classes are.
    """
    metrics = {}
    
    # 1. K-Nearest Neighbors cross-validation
    n_samples = len(y)
    cv_folds = min(5, n_samples // 2)  # Ensure we have enough samples per fold
    
    if cv_folds >= 2:
        for k in [3, 5, 7]:
            if k < n_samples:
                knn = KNeighborsClassifier(n_neighbors=min(k, n_samples - 1))
                scores = cross_val_score(knn, X, y, cv=cv_folds, scoring='accuracy')
                metrics[f'knn_{k}_accuracy'] = scores.mean()
                metrics[f'knn_{k}_std'] = scores.std()
    
    # 2. Class centroids distance
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]
    
    if len(X_class_0) > 0 and len(X_class_1) > 0:
        centroid_0 = X_class_0.mean(axis=0)
        centroid_1 = X_class_1.mean(axis=0)
        
        centroid_distance = np.linalg.norm(centroid_0 - centroid_1)
        metrics['centroid_distance'] = centroid_distance
        
        # 3. Within-class variance
        var_0 = np.mean(np.var(X_class_0, axis=0)) if len(X_class_0) > 1 else 0
        var_1 = np.mean(np.var(X_class_1, axis=0)) if len(X_class_1) > 1 else 0
        metrics['avg_within_class_variance'] = (var_0 + var_1) / 2
        
        # 4. Fisher's discriminant ratio
        between_class_var = centroid_distance ** 2
        within_class_var = var_0 + var_1
        metrics['fisher_ratio'] = between_class_var / (within_class_var + 1e-8)
    
    return metrics


def compute_feature_correlations(X, y, feature_names):
    """
    Compute correlation of each feature with the correct label.
    """
    correlations = {}
    
    for i, name in enumerate(feature_names):
        if np.std(X[:, i]) > 0 and np.std(y) > 0:
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations[name] = corr if not np.isnan(corr) else 0.0
        else:
            correlations[name] = 0.0
    
    return correlations


def plot_decision_space(X, y, output_path, title_suffix='', use_tsne=True):
    """
    Create visualization of the decision space.
    Uses t-SNE by default (if use_tsne=True), falls back to PCA if issues occur.
    """
    if len(X) < 5:
        print("  Warning: Not enough samples for visualization")
        return None
    
    # Handle NaN/Inf values - this was likely causing the segfault!
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print(f"  Warning: Found {np.sum(np.isnan(X))} NaN and {np.sum(np.isinf(X))} Inf values")
        print("  Replacing NaN/Inf with 0...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check again after scaling
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        print("  Warning: NaN/Inf after scaling, replacing with 0...")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_2d = None
    method_used = "t-SNE" if use_tsne else "PCA"
    
    if use_tsne:
        # Try t-SNE first
        try:
            perplexity = min(30, len(X) - 1)
            print(f"  Running t-SNE with perplexity={perplexity}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            X_2d = tsne.fit_transform(X_scaled)
            method_used = "t-SNE"
        except Exception as e:
            print(f"  Warning: t-SNE failed: {e}")
            X_2d = None
    
    # Try PCA if t-SNE failed or wasn't requested
    if X_2d is None:
        try:
            print("  Running PCA for visualization...")
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_scaled)
            explained_var = sum(pca.explained_variance_ratio_) * 100
            print(f"  PCA explained variance: {explained_var:.1f}%")
            method_used = "PCA"
        except Exception as e:
            print(f"  Warning: PCA failed: {e}")
    
    # If both fail, use first two features
    if X_2d is None:
        print("  Using first two features for visualization...")
        X_2d = X_scaled[:, :2]
        method_used = "First 2 Features"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot points
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                        cmap='RdYlGn', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Label (1=A correct, 0=B correct)', fontsize=11)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Labels and title
    ax.set_xlabel(f'{method_used} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method_used} Dimension 2', fontsize=12)
    ax.set_title(f'Decision Space ({method_used}): Can Features Separate Correct from Wrong?{title_suffix}', 
                fontsize=14, fontweight='bold')
    
    # Add class distribution info
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    ax.text(0.02, 0.98, f'Class distribution:\n  A correct: {n_class_1}\n  B correct: {n_class_0}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved decision space plot to {output_path}")
    
    return X_2d


def plot_feature_correlations(correlations, output_path):
    """
    Create bar plot of feature correlations with the label.
    """
    # Sort by absolute correlation
    sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars by sign
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + 0.01 if val >= 0 else val - 0.01
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:+.3f}',
               va='center', ha=ha, fontsize=9)
    
    # Reference line
    ax.axvline(x=0, color='black', linewidth=1)
    
    # Labels and title
    ax.set_xlabel('Correlation with Correct Label', fontsize=12)
    ax.set_title('Feature Importance: Which Features Predict the Correct Answer?', 
                fontsize=14, fontweight='bold')
    
    # Add interpretation guide
    ax.text(0.98, 0.02, 'Green: Higher value → A wins\nRed: Higher value → B wins',
           transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature correlations plot to {output_path}")


def plot_feature_distributions(X, y, feature_names, output_path):
    """
    Create distribution plots for each feature, split by class.
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, name in enumerate(feature_names):
        ax = axes[i]
        
        # Get feature values for each class
        feat_class_0 = X[y == 0, i]
        feat_class_1 = X[y == 1, i]
        
        # Plot histograms
        ax.hist(feat_class_0, bins=20, alpha=0.5, label='B correct', color='red', density=True)
        ax.hist(feat_class_1, bins=20, alpha=0.5, label='A correct', color='green', density=True)
        
        ax.set_xlabel(f'{name} difference (A - B)')
        ax.set_ylabel('Density')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved feature distributions plot to {output_path}")


def generate_report(X, y, feature_names, correlations, metrics, output_path):
    """
    Generate a text report summarizing the analysis.
    """
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DECISION SPACE ANALYSIS REPORT\n")
        f.write("Narrative Similarity Task - DRS Features\n")
        f.write("=" * 70 + "\n\n")
        
        # Dataset summary
        f.write("DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total instances analyzed: {len(y)}\n")
        f.write(f"Class distribution:\n")
        f.write(f"  - A is correct: {np.sum(y == 1)} ({100*np.mean(y == 1):.1f}%)\n")
        f.write(f"  - B is correct: {np.sum(y == 0)} ({100*np.mean(y == 0):.1f}%)\n")
        f.write(f"Number of features: {len(feature_names)}\n\n")
        
        # Separability metrics
        f.write("SEPARABILITY METRICS\n")
        f.write("-" * 40 + "\n")
        
        knn_acc = metrics.get('knn_5_accuracy', 0)
        knn_std = metrics.get('knn_5_std', 0)
        
        f.write(f"5-NN Cross-Validation Accuracy: {knn_acc:.3f} ± {knn_std:.3f}\n")
        f.write(f"3-NN Cross-Validation Accuracy: {metrics.get('knn_3_accuracy', 0):.3f}\n")
        f.write(f"7-NN Cross-Validation Accuracy: {metrics.get('knn_7_accuracy', 0):.3f}\n\n")
        
        f.write(f"Centroid Distance: {metrics.get('centroid_distance', 0):.4f}\n")
        f.write(f"Fisher's Discriminant Ratio: {metrics.get('fisher_ratio', 0):.4f}\n\n")
        
        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        
        if knn_acc > 0.75:
            f.write("✓ STRONG SIGNAL: Features show excellent separability.\n")
            f.write("  → XGBoost/learning will very likely improve results significantly.\n")
            f.write("  → Expected improvement: +0.10 to +0.15 over hand-tuned weights.\n")
        elif knn_acc > 0.65:
            f.write("✓ GOOD SIGNAL: Features show meaningful separability.\n")
            f.write("  → XGBoost/learning should improve results.\n")
            f.write("  → Expected improvement: +0.05 to +0.10 over hand-tuned weights.\n")
        elif knn_acc > 0.55:
            f.write("⚠ WEAK SIGNAL: Features show some separability.\n")
            f.write("  → Learning might help, but gains may be modest.\n")
            f.write("  → Consider adding more discriminative features.\n")
        else:
            f.write("✗ NO SIGNAL: Features do not separate classes well.\n")
            f.write("  → Current features are not discriminative enough.\n")
            f.write("  → Need to develop new features before learning will help.\n")
        
        f.write("\n")
        
        # Feature correlations
        f.write("FEATURE CORRELATIONS WITH CORRECT LABEL\n")
        f.write("-" * 40 + "\n")
        f.write("(Positive = A wins when feature is higher)\n")
        f.write("(Negative = B wins when feature is higher)\n\n")
        
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for name, corr in sorted_corr:
            direction = "→ A wins" if corr > 0 else "→ B wins"
            strength = "STRONG" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.15 else "weak"
            f.write(f"  {name:35s}: {corr:+.4f} ({strength}, {direction})\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Find best and worst features
        good_features = [name for name, corr in correlations.items() if abs(corr) > 0.15]
        bad_features = [name for name, corr in correlations.items() if abs(corr) < 0.05]
        
        if good_features:
            f.write(f"Most useful features: {', '.join(good_features)}\n")
        
        if bad_features:
            f.write(f"Low-signal features (consider removing): {', '.join(bad_features)}\n")
        
        f.write("\n")
        
        if knn_acc > 0.60:
            f.write("NEXT STEPS:\n")
            f.write("  1. Train XGBoost on these difference features\n")
            f.write("  2. Use cross-validation to find optimal hyperparameters\n")
            f.write("  3. Analyze feature importance from trained model\n")
        else:
            f.write("NEXT STEPS:\n")
            f.write("  1. Investigate why current features don't discriminate\n")
            f.write("  2. Add new features (graph structure, embeddings)\n")
            f.write("  3. Re-run this analysis with extended feature set\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Saved analysis report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize decision space for narrative similarity task'
    )
    parser.add_argument('--jsonl', type=str, required=True,
                       help='Path to JSONL data file (e.g., dev_track_a.jsonl)')
    parser.add_argument('--drs-dir', type=str, required=True,
                       help='Directory containing DRS files ({idx}_anchor_drs.txt, {idx}_a_drs.txt, {idx}_b_drs.txt)')
    parser.add_argument('--output', type=str, default='./visualization_results',
                       help='Output directory for plots and report')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                       help='Specific features to analyze (default: all)')
    parser.add_argument('--use-pca', action='store_true',
                       help='Use PCA instead of t-SNE for visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("DECISION SPACE VISUALIZATION")
    print("=" * 60 + "\n")
    
    # 1. Load JSONL data
    print("Step 1: Loading JSONL data...")
    data = load_jsonl_data(args.jsonl)
    
    if args.max_samples and len(data) > args.max_samples:
        print(f"  Limiting to {args.max_samples} samples for testing")
        data = data[:args.max_samples]
    
    # 2. Extract features
    print("\nStep 2: Extracting difference features...")
    # If user didn't specify features, pass None for auto-detection
    feature_names = args.features  # Will be None if not specified
    
    X, y, feature_names, valid_indices = extract_difference_features(
        data, drs_dir=args.drs_dir, feature_names=feature_names
    )
    
    if len(X) < 10:
        print("\n✗ Error: Not enough valid instances to analyze")
        print("  Check that DRS files exist in the specified directory")
        print(f"  Expected files like: {args.drs_dir}/0_anchor_drs.txt, {args.drs_dir}/0_a_drs.txt, {args.drs_dir}/0_b_drs.txt")
        return
    
    # 3. Compute metrics
    print("\nStep 3: Computing separability metrics...")
    metrics = compute_separability_metrics(X, y)
    
    knn_acc = metrics.get('knn_5_accuracy', 0)
    print(f"\n  5-NN CV Accuracy: {knn_acc:.3f} ± {metrics.get('knn_5_std', 0):.3f}")
    
    if knn_acc > 0.65:
        print("  → Good signal! Learning should help.")
    elif knn_acc > 0.55:
        print("  → Weak signal. Learning might help marginally.")
    else:
        print("  → No clear signal. Features may not be discriminative enough.")
    
    # 4. Compute feature correlations
    print("\nStep 4: Computing feature correlations...")
    correlations = compute_feature_correlations(X, y, feature_names)
    
    # 5. Generate visualizations
    print("\nStep 5: Generating visualizations...")
    
    # Decision space plot (with error handling)
    try:
        plot_decision_space(
            X, y, 
            output_dir / 'decision_space.png',
            title_suffix=f'\n(5-NN Accuracy: {knn_acc:.3f})',
            use_tsne=not args.use_pca  # Use t-SNE by default, PCA if --use-pca
        )
    except Exception as e:
        print(f"  Warning: Could not generate decision space plot: {e}")
    
    # Feature correlations plot
    try:
        plot_feature_correlations(
            correlations,
            output_dir / 'feature_correlations.png'
        )
    except Exception as e:
        print(f"  Warning: Could not generate feature correlations plot: {e}")
    
    # Feature distributions plot
    try:
        plot_feature_distributions(
            X, y, feature_names,
            output_dir / 'feature_distributions.png'
        )
    except Exception as e:
        print(f"  Warning: Could not generate feature distributions plot: {e}")
    
    # 6. Generate report
    print("\nStep 6: Generating analysis report...")
    generate_report(
        X, y, feature_names, correlations, metrics,
        output_dir / 'analysis_report.txt'
    )
    
    # 7. Save raw data for further analysis
    np.savez(
        output_dir / 'extracted_features.npz',
        X=X, y=y, feature_names=np.array(feature_names), valid_indices=np.array(valid_indices)
    )
    print(f"✓ Saved extracted features to {output_dir / 'extracted_features.npz'}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nAnalyzed {len(X)} instances with {len(feature_names)} features")
    print(f"\n5-NN Cross-Validation Accuracy: {knn_acc:.3f}")
    print("\nTop 3 most predictive features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, corr in sorted_corr[:3]:
        print(f"  - {name}: {corr:+.3f}")
    
    print(f"\nResults saved to: {output_dir}/")
    print("  - decision_space.png")
    print("  - feature_correlations.png")
    print("  - feature_distributions.png")
    print("  - analysis_report.txt")
    print("  - extracted_features.npz")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    main()
