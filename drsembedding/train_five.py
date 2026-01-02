"""
Training Script for Five-Component DRS Encoder

Trains the five-component model on triplet similarity task.

Usage:
    python train_five.py ../data/synthetic/preprocessed_synth.jsonl --epochs 50

    # Ablation: disable specific components
    python train_five.py data.jsonl --no_temporal --no_participant
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import time
from pathlib import Path
import json

# Import our modules
from data_loader import DRSDataset, Triplet
from five_component_encoder import (
    FiveComponentConfig,
    FiveComponentModel,
    StoryDataProcessor,
)


class FiveTripletDataset(Dataset):
    """
    PyTorch Dataset for five-component model.
    Pre-processes all triplets for efficient training.
    """

    def __init__(
            self,
            drs_dataset: DRSDataset,
            processor: StoryDataProcessor,
            indices: Optional[List[int]] = None,
            device: str = 'cpu',
            verbose: bool = True,
    ):
        self.drs_dataset = drs_dataset
        self.processor = processor
        self.indices = indices if indices is not None else list(range(len(drs_dataset)))
        self.device = device

        # Pre-process all triplets
        if verbose:
            print(f"Pre-processing {len(self.indices)} triplets...")

        self.processed_data = []
        start_time = time.time()

        for i, idx in enumerate(self.indices):
            triplet = drs_dataset[idx]
            anchor_data, story_a_data, story_b_data = processor.process_triplet(triplet, device)

            self.processed_data.append({
                'anchor': anchor_data,
                'story_a': story_a_data,
                'story_b': story_b_data,
                'label': 1 if triplet.label else 0,
                'idx': triplet.idx,
            })

            if verbose and (i + 1) % 200 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(self.indices) - i - 1) / rate
                print(f"  Processed {i + 1}/{len(self.indices)} ({rate:.1f}/s, ETA: {eta:.0f}s)")

        elapsed = time.time() - start_time
        if verbose:
            print(f"✓ Processed {len(self.processed_data)} triplets in {elapsed:.1f}s")

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict:
        return self.processed_data[idx]


def collate_fn(batch: List[Dict]) -> List[Dict]:
    """
    Simple collate - returns list of dicts since each story has different sizes.
    We process one triplet at a time in training.
    """
    return batch


class FiveComponentTrainer:
    """
    Trainer for five-component model.
    """

    def __init__(
            self,
            model: FiveComponentModel,
            train_dataset: FiveTripletDataset,
            val_dataset: FiveTripletDataset,
            optimizer: optim.Optimizer,
            scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
            device: str = 'cpu',
            checkpoint_dir: str = './checkpoints_five',
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Shuffle indices
        indices = list(range(len(self.train_dataset)))
        np.random.shuffle(indices)

        for idx in indices:
            item = self.train_dataset[idx]

            # Forward pass
            self.optimizer.zero_grad()

            output = self.model(
                item['anchor'],
                item['story_a'],
                item['story_b'],
            )

            # Compute loss
            label = torch.tensor([float(item['label'])], device=self.device)
            loss = F.binary_cross_entropy_with_logits(output['logits'], label)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            pred = (output['logits'] > 0).float()
            total_correct += (pred == label).sum().item()
            total_samples += 1

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, dataset: FiveTripletDataset) -> Tuple[float, float]:
        """Evaluate on dataset."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for idx in range(len(dataset)):
            item = dataset[idx]

            output = self.model(
                item['anchor'],
                item['story_a'],
                item['story_b'],
            )

            label = torch.tensor([float(item['label'])], device=self.device)
            loss = F.binary_cross_entropy_with_logits(output['logits'], label)

            total_loss += loss.item()
            pred = (output['logits'] > 0).float()
            total_correct += (pred == label).sum().item()
            total_samples += 1

        return total_loss / total_samples, total_correct / total_samples

    def train(
            self,
            num_epochs: int,
            patience: int = 10,
            verbose: bool = True,
    ) -> Dict:
        """Full training loop."""
        no_improve_count = 0

        print("\n" + "=" * 70)
        print("TRAINING STARTED")
        print("=" * 70)
        print(
            f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10} {'LR':>12} {'Time':>8}")
        print("-" * 70)

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Evaluate
            val_loss, val_acc = self.evaluate(self.val_dataset)

            elapsed = time.time() - start_time

            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                no_improve_count = 0
                self.save_checkpoint('best_model.pt', epoch)
                marker = '*'
            else:
                no_improve_count += 1
                marker = ''

            # Print progress
            if verbose:
                print(f"{epoch + 1:>6} {train_loss:>12.4f} {train_acc * 100:>9.2f}% "
                      f"{val_loss:>12.4f} {val_acc * 100:>9.2f}%{marker} {current_lr:>12.6f} {elapsed:>7.1f}s")

            # Early stopping
            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

        print("-" * 70)
        print(f"Best validation accuracy: {self.best_val_acc * 100:.2f}% at epoch {self.best_epoch + 1}")

        # Save final model
        self.save_checkpoint('final_model.pt', epoch)

        return self.history

    def save_checkpoint(self, filename: str, epoch: int):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        return checkpoint['epoch']


def create_datasets(
        jsonl_path: str,
        processor: StoryDataProcessor,
        val_split: float = 0.1,
        seed: int = 42,
        device: str = 'cpu',
) -> Tuple[FiveTripletDataset, FiveTripletDataset]:
    """Create train and validation datasets."""

    drs_dataset = DRSDataset(jsonl_path)

    # Split indices
    num_samples = len(drs_dataset)
    indices = list(range(num_samples))

    np.random.seed(seed)
    np.random.shuffle(indices)

    val_size = int(num_samples * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f"\nData split: {len(train_indices)} train, {len(val_indices)} val")

    # Create datasets
    train_dataset = FiveTripletDataset(drs_dataset, processor, train_indices, device)
    val_dataset = FiveTripletDataset(drs_dataset, processor, val_indices, device)

    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description='Train five-component DRS encoder')
    parser.add_argument('data_path', type=str, help='Path to preprocessed JSONL')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_five', help='Checkpoint dir')
    parser.add_argument('--device', type=str, default='auto', help='Device')

    # Component dimensions
    parser.add_argument('--temporal_dim', type=int, default=64, help='Temporal component dim')
    parser.add_argument('--logical_dim', type=int, default=64, help='Logical component dim')
    parser.add_argument('--participant_dim', type=int, default=64, help='Participant component dim')
    parser.add_argument('--semantic_dim', type=int, default=64, help='Semantic component dim')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dim')
    parser.add_argument('--output_dim', type=int, default=128, help='Output embedding dim')

    # Text encoder
    parser.add_argument('--text_model', type=str, default='all-MiniLM-L6-v2', help='Sentence-transformer model')

    # Ablation flags
    parser.add_argument('--no_temporal', action='store_true', help='Disable temporal component')
    parser.add_argument('--no_logical', action='store_true', help='Disable logical component')
    parser.add_argument('--no_participant', action='store_true', help='Disable participant component')
    parser.add_argument('--no_semantic', action='store_true', help='Disable semantic component')
    parser.add_argument('--no_text', action='store_true', help='Disable text component')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("\n" + "=" * 70)
    print("FIVE-COMPONENT DRS ENCODER TRAINING")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Data: {args.data_path}")

    # Load dataset for vocabulary
    print("\n" + "-" * 40)
    print("LOADING DATA")
    print("-" * 40)

    drs_dataset = DRSDataset(args.data_path)
    processor = StoryDataProcessor(drs_dataset, text_model_name=args.text_model)

    # Create config
    config = FiveComponentConfig(
        temporal_dim=args.temporal_dim,
        logical_dim=args.logical_dim,
        participant_dim=args.participant_dim,
        semantic_dim=args.semantic_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        text_model_name=args.text_model,
        num_verbnet_classes=len(processor.verbnet_vocab),
        num_predicates=len(processor.predicate_vocab),
        use_temporal=not args.no_temporal,
        use_logical=not args.no_logical,
        use_participant=not args.no_participant,
        use_semantic=not args.no_semantic,
        use_text=not args.no_text,
    )

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        args.data_path,
        processor,
        val_split=args.val_split,
        seed=args.seed,
        device=device,
    )

    # Create model
    print("\n" + "-" * 40)
    print("MODEL CONFIGURATION")
    print("-" * 40)

    model = FiveComponentModel(config, device)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Create trainer
    trainer = FiveComponentTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience}")

    history = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience,
    )

    # Save config and history
    config_dict = {
        'temporal_dim': config.temporal_dim,
        'logical_dim': config.logical_dim,
        'participant_dim': config.participant_dim,
        'semantic_dim': config.semantic_dim,
        'text_dim': config.text_dim,
        'hidden_dim': config.hidden_dim,
        'output_dim': config.output_dim,
        'dropout': config.dropout,
        'text_model_name': config.text_model_name,
        'num_verbnet_classes': config.num_verbnet_classes,
        'num_predicates': config.num_predicates,
        'use_temporal': config.use_temporal,
        'use_logical': config.use_logical,
        'use_participant': config.use_participant,
        'use_semantic': config.use_semantic,
        'use_text': config.use_text,
    }

    with open(trainer.checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    with open(trainer.checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {trainer.best_val_acc * 100:.2f}%")
    print(f"  Checkpoints saved to: {trainer.checkpoint_dir}")

    return trainer, history


if __name__ == "__main__":
    main()