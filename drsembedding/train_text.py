"""
Training Script with Text-Enhanced Features

Supports both base features and text-enhanced features.

Usage:
    # With base features only (original)
    python train_text.py ../data/synthetic/preprocessed_synth.jsonl --epochs 50

    # With text-enhanced features (new)
    python train_text.py ../data/synthetic/preprocessed_synth.jsonl --epochs 50 --use_text

    # With specific model
    python train_text.py ../data/synthetic/preprocessed_synth.jsonl --epochs 50 --use_text --text_model all-mpnet-base-v2
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import argparse
import time
from pathlib import Path
import json

# Import our modules
from data_loader import DRSDataset, Triplet
from feature_encoder import EventFeatureEncoder
from text_feature_encoder import TextEnhancedFeatureEncoder
from simple_encoder import TripletSimilarityModel, compute_triplet_loss, compute_accuracy


class TripletDataset(Dataset):
    """
    PyTorch Dataset wrapper for DRS triplets.
    Supports both base and text-enhanced encoders.
    """

    def __init__(
            self,
            drs_dataset: DRSDataset,
            feature_encoder: Union[EventFeatureEncoder, TextEnhancedFeatureEncoder],
            indices: Optional[List[int]] = None,
            verbose: bool = True,
    ):
        self.drs_dataset = drs_dataset
        self.feature_encoder = feature_encoder
        self.indices = indices if indices is not None else list(range(len(drs_dataset)))

        # Pre-encode all triplets
        if verbose:
            print(f"Pre-encoding {len(self.indices)} triplets...")

        self.encoded_data = []
        start_time = time.time()

        for i, idx in enumerate(self.indices):
            triplet = drs_dataset[idx]
            encoded = feature_encoder.encode_triplet(triplet)

            self.encoded_data.append({
                'anchor': encoded['anchor'],
                'story_a': encoded['story_a'],
                'story_b': encoded['story_b'],
                'label': 1 if triplet.label else 0,
                'idx': triplet.idx,
            })

            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(self.indices) - i - 1) / rate
                print(f"  Encoded {i + 1}/{len(self.indices)} triplets... "
                      f"({rate:.1f}/s, ETA: {eta:.0f}s)")

        elapsed = time.time() - start_time
        if verbose:
            print(f"✓ Encoded {len(self.encoded_data)} triplets in {elapsed:.1f}s")

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_data[idx]


def collate_triplets(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for batching triplets with padding."""
    max_anchor = max(item['anchor'].size(0) for item in batch)
    max_a = max(item['story_a'].size(0) for item in batch)
    max_b = max(item['story_b'].size(0) for item in batch)

    feature_dim = batch[0]['anchor'].size(1)
    batch_size = len(batch)

    anchor_padded = torch.zeros(batch_size, max_anchor, feature_dim)
    a_padded = torch.zeros(batch_size, max_a, feature_dim)
    b_padded = torch.zeros(batch_size, max_b, feature_dim)

    anchor_mask = torch.zeros(batch_size, max_anchor)
    a_mask = torch.zeros(batch_size, max_a)
    b_mask = torch.zeros(batch_size, max_b)

    labels = []

    for i, item in enumerate(batch):
        anchor_len = item['anchor'].size(0)
        anchor_padded[i, :anchor_len] = item['anchor']
        anchor_mask[i, :anchor_len] = 1

        a_len = item['story_a'].size(0)
        a_padded[i, :a_len] = item['story_a']
        a_mask[i, :a_len] = 1

        b_len = item['story_b'].size(0)
        b_padded[i, :b_len] = item['story_b']
        b_mask[i, :b_len] = 1

        labels.append(item['label'])

    return {
        'anchor': anchor_padded,
        'story_a': a_padded,
        'story_b': b_padded,
        'anchor_mask': anchor_mask,
        'a_mask': a_mask,
        'b_mask': b_mask,
        'labels': torch.tensor(labels, dtype=torch.float32),
    }


class Trainer:
    """Trainer class for triplet similarity model."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: optim.Optimizer,
            scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
            device: str = 'cpu',
            checkpoint_dir: str = './checkpoints',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.train_loader:
            anchor = batch['anchor'].to(self.device)
            story_a = batch['story_a'].to(self.device)
            story_b = batch['story_b'].to(self.device)
            anchor_mask = batch['anchor_mask'].to(self.device)
            a_mask = batch['a_mask'].to(self.device)
            b_mask = batch['b_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(
                anchor, story_a, story_b,
                anchor_mask, a_mask, b_mask,
            )

            loss = compute_triplet_loss(output['logits'], labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = (output['logits'] > 0).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in loader:
            anchor = batch['anchor'].to(self.device)
            story_a = batch['story_a'].to(self.device)
            story_b = batch['story_b'].to(self.device)
            anchor_mask = batch['anchor_mask'].to(self.device)
            a_mask = batch['a_mask'].to(self.device)
            b_mask = batch['b_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            output = self.model(
                anchor, story_a, story_b,
                anchor_mask, a_mask, b_mask,
            )

            loss = compute_triplet_loss(output['logits'], labels)

            total_loss += loss.item() * labels.size(0)
            predictions = (output['logits'] > 0).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def train(
            self,
            num_epochs: int,
            patience: int = 10,
            verbose: bool = True,
    ) -> Dict:
        no_improve_count = 0

        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10} {'LR':>10}")
        print("-" * 60)

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                no_improve_count = 0
                self.save_checkpoint('best_model.pt', epoch)
            else:
                no_improve_count += 1

            if verbose:
                marker = '*' if val_acc == self.best_val_acc else ''
                print(f"{epoch + 1:>6} {train_loss:>12.4f} {train_acc * 100:>9.2f}% "
                      f"{val_loss:>12.4f} {val_acc * 100:>9.2f}%{marker} {current_lr:>10.6f}")

            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print("-" * 60)
        print(f"Best validation accuracy: {self.best_val_acc * 100:.2f}% at epoch {self.best_epoch + 1}")

        self.save_checkpoint('final_model.pt', epoch)

        return self.history

    def save_checkpoint(self, filename: str, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        return checkpoint['epoch']


def create_data_loaders(
        jsonl_path: str,
        use_text: bool = False,
        text_model: str = 'all-MiniLM-L6-v2',
        batch_size: int = 32,
        val_split: float = 0.1,
        seed: int = 42,
        device: str = 'cpu',
) -> Tuple[DataLoader, DataLoader, Union[EventFeatureEncoder, TextEnhancedFeatureEncoder]]:
    """Create train and validation data loaders."""

    drs_dataset = DRSDataset(jsonl_path)

    # Create appropriate encoder
    if use_text:
        print(f"\nUsing TEXT-ENHANCED features with model: {text_model}")
        feature_encoder = TextEnhancedFeatureEncoder(
            dataset=drs_dataset,
            model_name=text_model,
            device=device,
        )
    else:
        print(f"\nUsing BASE features only")
        feature_encoder = EventFeatureEncoder(dataset=drs_dataset)

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
    train_dataset = TripletDataset(drs_dataset, feature_encoder, train_indices)
    val_dataset = TripletDataset(drs_dataset, feature_encoder, val_indices)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_triplets,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_triplets,
    )

    return train_loader, val_loader, feature_encoder


def main():
    parser = argparse.ArgumentParser(description='Train triplet similarity model')
    parser.add_argument('data_path', type=str, help='Path to preprocessed JSONL')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=64, help='Output embedding dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint dir')
    parser.add_argument('--device', type=str, default='auto', help='Device')

    # Text-enhanced features
    parser.add_argument('--use_text', action='store_true', help='Use text-enhanced features')
    parser.add_argument('--text_model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model name')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("\n" + "=" * 60)
    print("TRAINING WITH " + ("TEXT-ENHANCED" if args.use_text else "BASE") + " FEATURES")
    print("=" * 60)
    print(f"\nDevice: {device}")

    # Create data loaders
    train_loader, val_loader, feature_encoder = create_data_loaders(
        args.data_path,
        use_text=args.use_text,
        text_model=args.text_model,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        device=device,
    )

    # Create model
    print("\n" + "-" * 40)
    print("MODEL CONFIGURATION")
    print("-" * 40)

    model = TripletSimilarityModel(
        input_dim=feature_encoder.feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Input dim: {feature_encoder.feature_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Output dim: {args.output_dim}")
    print(f"Total parameters: {total_params:,}")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Update checkpoint directory for text model
    if args.use_text:
        checkpoint_dir = args.checkpoint_dir + '_text'
    else:
        checkpoint_dir = args.checkpoint_dir

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Train
    history = trainer.train(num_epochs=args.epochs, patience=args.patience)

    # Save history
    history_path = trainer.checkpoint_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        'use_text': args.use_text,
        'text_model': args.text_model if args.use_text else None,
        'input_dim': feature_encoder.feature_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'dropout': args.dropout,
    }

    config_path = trainer.checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {trainer.best_val_acc * 100:.2f}%")
    print(f"  Checkpoints saved to: {trainer.checkpoint_dir}")

    return trainer, history


if __name__ == "__main__":
    main()