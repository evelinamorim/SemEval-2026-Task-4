"""
Evaluation Script for Five-Component DRS Encoder

Evaluates a trained five-component model on test/dev set.

Usage:
    python evaluate_five.py ../data/preprocessed_dev.jsonl \
        --checkpoint ./checkpoints_five/best_model.pt \
        --train_data ../data/synthetic/preprocessed_synth.jsonl
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
import json

# Import our modules
from data_loader import DRSDataset, Triplet
from five_component_encoder import (
    FiveComponentConfig,
    FiveComponentModel,
    StoryDataProcessor,
)


class FiveComponentEvaluator:
    """Evaluator for five-component model."""

    def __init__(
            self,
            model: FiveComponentModel,
            processor: StoryDataProcessor,
            device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.processor = processor
        self.device = device

    @torch.no_grad()
    def evaluate_triplet(self, triplet: Triplet) -> Dict:
        """Evaluate a single triplet."""
        anchor_data, story_a_data, story_b_data = self.processor.process_triplet(
            triplet, self.device
        )

        output = self.model(anchor_data, story_a_data, story_b_data)

        logits = output['logits'].item()
        pred_a_closer = logits > 0
        label_a_closer = triplet.label
        correct = pred_a_closer == label_a_closer

        return {
            'idx': triplet.idx,
            'logits': logits,
            'sim_a': output['sim_a'].item(),
            'sim_b': output['sim_b'].item(),
            'pred_a_closer': pred_a_closer,
            'label_a_closer': label_a_closer,
            'correct': correct,
        }

    @torch.no_grad()
    def evaluate_dataset(
            self,
            dataset: DRSDataset,
            verbose: bool = True,
    ) -> Dict:
        """Evaluate on entire dataset."""
        results = []
        correct_count = 0
        total_count = 0

        for i, triplet in enumerate(dataset.triplets):
            result = self.evaluate_triplet(triplet)
            results.append(result)

            if result['correct']:
                correct_count += 1
            total_count += 1

            if verbose and (i + 1) % 50 == 0:
                print(f"  Evaluated {i + 1}/{len(dataset)} triplets...")

        accuracy = correct_count / total_count if total_count > 0 else 0

        # Confidence analysis
        logits = [r['logits'] for r in results]
        confident_correct = sum(1 for r in results if r['correct'] and abs(r['logits']) > 0.1)
        confident_wrong = sum(1 for r in results if not r['correct'] and abs(r['logits']) > 0.1)
        uncertain = sum(1 for r in results if abs(r['logits']) <= 0.1)

        return {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': total_count,
            'results': results,
            'confident_correct': confident_correct,
            'confident_wrong': confident_wrong,
            'uncertain': uncertain,
            'mean_logits': np.mean(logits),
            'std_logits': np.std(logits),
        }

    @torch.no_grad()
    def get_component_embeddings(self, triplet: Triplet) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get individual component embeddings for analysis.
        """
        anchor_data, story_a_data, story_b_data = self.processor.process_triplet(
            triplet, self.device
        )

        embeddings = {'anchor': {}, 'story_a': {}, 'story_b': {}}

        for name, data in [('anchor', anchor_data), ('story_a', story_a_data), ('story_b', story_b_data)]:
            # Component 1: Temporal
            event_node_features = None
            if self.model.temporal_encoder:
                # FIX: Unpack the tuple (pooled_emb, node_features)
                temporal_emb, event_node_features = self.model.temporal_encoder(
                    data['event_features'],
                    data['temporal_adjacency'],
                    data['temporal_edge_types'],
                )
                embeddings[name]['temporal'] = temporal_emb

            # Component 2: Logical (Returns a tensor)
            if self.model.logical_encoder:
                embeddings[name]['logical'] = self.model.logical_encoder(
                    data['event_type_dist'],
                    data['verbnet_indices'],
                    data['predicate_multihot'],
                )

            # Component 3: Participant
            actor_node_features = None
            if self.model.participant_encoder:
                # FIX: Unpack the tuple (pooled_emb, node_features)
                participant_emb, actor_node_features = self.model.participant_encoder(
                    data['actor_features'],
                    data['coref_adjacency'],
                )
                embeddings[name]['participant'] = participant_emb

            # Component 4: Semantic
            if (self.model.semantic_encoder and 
                event_node_features is not None and 
                actor_node_features is not None):
                
                # FIX: Use the actual GNN node outputs instead of raw local node_encoder
                embeddings[name]['semantic'] = self.model.semantic_encoder(
                    event_node_features,
                    actor_node_features,
                    data['semantic_edges'],
                    data['event_id_to_idx'],
                    data['actor_id_to_idx'],
                )

            # Component 5: Text
            if self.model.text_encoder:
                embeddings[name]['text'] = self.model.text_encoder(data['text'])

        return embeddings

    @torch.no_grad()
    def component_similarity_analysis(self, triplet: Triplet) -> Dict[str, Dict[str, float]]:
        """
        Analyze which components contribute most to similarity.

        Returns:
            Dict with per-component similarities for A and B
        """
        embeddings = self.get_component_embeddings(triplet)

        analysis = {}

        for comp_name in ['temporal', 'logical', 'participant', 'semantic', 'text']:
            if comp_name in embeddings['anchor']:
                anchor_emb = embeddings['anchor'][comp_name]
                a_emb = embeddings['story_a'].get(comp_name)
                b_emb = embeddings['story_b'].get(comp_name)

                if a_emb is not None and b_emb is not None:
                    sim_a = F.cosine_similarity(anchor_emb.unsqueeze(0), a_emb.unsqueeze(0)).item()
                    sim_b = F.cosine_similarity(anchor_emb.unsqueeze(0), b_emb.unsqueeze(0)).item()

                    analysis[comp_name] = {
                        'sim_a': sim_a,
                        'sim_b': sim_b,
                        'diff': sim_a - sim_b,
                        'prediction': 'A' if sim_a > sim_b else 'B',
                    }

        return analysis


def load_model_and_processor(
        checkpoint_path: str,
        train_data_path: str,
        device: str = 'cpu',
) -> Tuple[FiveComponentModel, StoryDataProcessor, Dict]:
    """Load trained model and processor."""

    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'

    # Load config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        print(f"✓ Loaded config from {config_path}")
    else:
        print(f"⚠ No config.json found, using defaults")
        config_dict = {}

    # Load training data for vocabulary
    print(f"Loading vocabulary from: {train_data_path}")
    train_dataset = DRSDataset(train_data_path)
    #text_model_name = config_dict.get('text_model_name', 'all-MiniLM-L6-v2')
    text_model_name = "all-MiniLM-L6-v2"
    processor = StoryDataProcessor(train_dataset, text_model_name=text_model_name)

    # Create config
    config = FiveComponentConfig(
        temporal_dim=config_dict.get('temporal_dim', 64),
        logical_dim=config_dict.get('logical_dim', 64),
        participant_dim=config_dict.get('participant_dim', 64),
        semantic_dim=config_dict.get('semantic_dim', 64),
        hidden_dim=config_dict.get('hidden_dim', 128),
        output_dim=config_dict.get('output_dim', 128),
        dropout=config_dict.get('dropout', 0.1),
        text_model_name=config_dict.get('text_model_name', 'all-MiniLM-L6-v2'),
        num_verbnet_classes=len(processor.verbnet_vocab),
        num_predicates=len(processor.predicate_vocab),
        use_temporal=config_dict.get('use_temporal', True),
        use_logical=config_dict.get('use_logical', True),
        use_participant=config_dict.get('use_participant', True),
        use_semantic=config_dict.get('use_semantic', True),
        use_text=config_dict.get('use_text', True),
    )

    # Create model
    model = FiveComponentModel(config, device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}")

    return model, processor, config_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate five-component model')
    parser.add_argument('test_data', type=str, help='Path to test/dev preprocessed JSONL')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--output', type=str, default=None, help='Save results to JSON')
    parser.add_argument('--show_errors', action='store_true', help='Show misclassified examples')
    parser.add_argument('--num_errors', type=int, default=10, help='Number of errors to show')
    parser.add_argument('--component_analysis', action='store_true', help='Show per-component analysis')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("\n" + "=" * 70)
    print("FIVE-COMPONENT MODEL EVALUATION")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Test data: {args.test_data}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model and processor
    model, processor, config_dict = load_model_and_processor(
        checkpoint_path=args.checkpoint,
        train_data_path=args.train_data,
        device=device,
    )

    # Load test data
    print(f"\nLoading test data: {args.test_data}")
    test_dataset = DRSDataset(args.test_data)

    # Create evaluator
    evaluator = FiveComponentEvaluator(model, processor, device)

    # Run evaluation
    print("\n" + "-" * 40)
    print("RUNNING EVALUATION")
    print("-" * 40)

    metrics = evaluator.evaluate_dataset(test_dataset, verbose=True)

    # Print results
    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)

    print(f"\nComponents enabled:")
    print(f"  Temporal:    {'✓' if config_dict.get('use_temporal', True) else '✗'}")
    print(f"  Logical:     {'✓' if config_dict.get('use_logical', True) else '✗'}")
    print(f"  Participant: {'✓' if config_dict.get('use_participant', True) else '✗'}")
    print(f"  Semantic:    {'✓' if config_dict.get('use_semantic', True) else '✗'}")
    print(f"  Text:        {'✓' if config_dict.get('use_text', True) else '✗'}")

    print(f"\nAccuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Correct: {metrics['correct']} / {metrics['total']}")

    print(f"\nConfidence analysis:")
    print(f"  Confident & correct: {metrics['confident_correct']}")
    print(f"  Confident & wrong:   {metrics['confident_wrong']}")
    print(f"  Uncertain (|logit| <= 0.1): {metrics['uncertain']}")

    print(f"\nLogits statistics:")
    print(f"  Mean: {metrics['mean_logits']:.4f}")
    print(f"  Std:  {metrics['std_logits']:.4f}")

    # Component analysis
    if args.component_analysis:
        print("\n" + "-" * 40)
        print("PER-COMPONENT SIMILARITY ANALYSIS (first 5 triplets)")
        print("-" * 40)

        for i in range(min(5, len(test_dataset))):
            triplet = test_dataset[i]
            analysis = evaluator.component_similarity_analysis(triplet)

            print(f"\nTriplet {triplet.idx} (Label: {'A' if triplet.label else 'B'}):")
            for comp_name, comp_data in analysis.items():
                pred_marker = '✓' if (comp_data['prediction'] == 'A') == triplet.label else '✗'
                print(f"  {comp_name:12}: sim_A={comp_data['sim_a']:.3f}, sim_B={comp_data['sim_b']:.3f}, "
                      f"pred={comp_data['prediction']} {pred_marker}")

    # Show errors
    if args.show_errors:
        print("\n" + "-" * 40)
        print(f"MISCLASSIFIED EXAMPLES (first {args.num_errors})")
        print("-" * 40)

        errors = [r for r in metrics['results'] if not r['correct']]

        for i, error in enumerate(errors[:args.num_errors]):
            # Find triplet
            triplet = None
            for t in test_dataset.triplets:
                if t.idx == error['idx']:
                    triplet = t
                    break

            print(f"\n[{i + 1}] Triplet {error['idx']}:")
            print(f"  Predicted: {'A closer' if error['pred_a_closer'] else 'B closer'}")
            print(f"  Actual:    {'A closer' if error['label_a_closer'] else 'B closer'}")
            print(f"  Logits: {error['logits']:.4f}")
            print(f"  Sim A: {error['sim_a']:.4f}, Sim B: {error['sim_b']:.4f}")

            if triplet:
                print(f"  Anchor events: {triplet.anchor.num_events}")
                print(f"  Story A events: {triplet.story_a.num_events}")
                print(f"  Story B events: {triplet.story_b.num_events}")

                # Component analysis for errors
                if args.component_analysis:
                    analysis = evaluator.component_similarity_analysis(triplet)
                    print(f"  Component breakdown:")
                    for comp_name, comp_data in analysis.items():
                        print(f"    {comp_name}: diff={comp_data['diff']:.3f}")

    # Save results
    if args.output:
        output_data = {
            'accuracy': metrics['accuracy'],
            'correct': metrics['correct'],
            'total': metrics['total'],
            'confident_correct': metrics['confident_correct'],
            'confident_wrong': metrics['confident_wrong'],
            'uncertain': metrics['uncertain'],
            'mean_logits': metrics['mean_logits'],
            'std_logits': metrics['std_logits'],
            'checkpoint': args.checkpoint,
            'test_data': args.test_data,
            'config': config_dict,
            'per_example': metrics['results'],
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {args.output}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    main()