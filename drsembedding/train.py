"""
Step 2.6: Training Loop

Complete training pipeline for triplet similarity model:
1. Data loading with train/val split
2. Batching with padding
3. Training loop with early stopping
4. Evaluation and metrics
5. Checkpoint saving

Usage:
    python train.py ../data/synthetic/preprocessed_synth.jsonl --epochs 50 --batch_size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import time
from pathlib import Path
import json

# Import our modules
from data_loader import DRSDataset, Triplet
from feature_encoder import EventFeatureEncoder
from simple_encoder import TripletSimilarityModel, compute_triplet_loss, compute_accuracy


class TripletDataset(Dataset):
    """
    PyTorch Dataset wrapper for DRS triplets.
    
    Pre-encodes all features for faster training.
    """
    
    def __init__(
        self,
        drs_dataset: DRSDataset,
        feature_encoder: EventFeatureEncoder,
        indices: Optional[List[int]] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            drs_dataset: DRSDataset with loaded triplets
            feature_encoder: EventFeatureEncoder for feature extraction
            indices: Optional subset of indices to use
        """
        self.drs_dataset = drs_dataset
        self.feature_encoder = feature_encoder
        self.indices = indices if indices is not None else list(range(len(drs_dataset)))
        
        # Pre-encode all triplets for speed
        print(f"Pre-encoding {len(self.indices)} triplets...")
        self.encoded_data = []
        
        for idx in self.indices:
            triplet = drs_dataset[idx]
            encoded = feature_encoder.encode_triplet(triplet)
            
            self.encoded_data.append({
                'anchor': encoded['anchor'],
                'story_a': encoded['story_a'],
                'story_b': encoded['story_b'],
                'label': 1 if triplet.label else 0,  # 1 = A closer, 0 = B closer
                'idx': triplet.idx,
            })
        
        print(f"✓ Encoded {len(self.encoded_data)} triplets")
    
    def __len__(self) -> int:
        return len(self.encoded_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded_data[idx]


def collate_triplets(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching triplets with padding.
    
    Pads sequences to max length in batch.
    """
    # Find max lengths
    max_anchor = max(item['anchor'].size(0) for item in batch)
    max_a = max(item['story_a'].size(0) for item in batch)
    max_b = max(item['story_b'].size(0) for item in batch)
    
    feature_dim = batch[0]['anchor'].size(1)
    batch_size = len(batch)
    
    # Initialize padded tensors
    anchor_padded = torch.zeros(batch_size, max_anchor, feature_dim)
    a_padded = torch.zeros(batch_size, max_a, feature_dim)
    b_padded = torch.zeros(batch_size, max_b, feature_dim)
    
    # Masks (1 = valid, 0 = padding)
    anchor_mask = torch.zeros(batch_size, max_anchor)
    a_mask = torch.zeros(batch_size, max_a)
    b_mask = torch.zeros(batch_size, max_b)
    
    labels = []
    
    for i, item in enumerate(batch):
        # Anchor
        anchor_len = item['anchor'].size(0)
        anchor_padded[i, :anchor_len] = item['anchor']
        anchor_mask[i, :anchor_len] = 1
        
        # Story A
        a_len = item['story_a'].size(0)
        a_padded[i, :a_len] = item['story_a']
        a_mask[i, :a_len] = 1
        
        # Story B
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
    """
    Trainer class for triplet similarity model.
    """
    
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
        """
        Initialize trainer.
        
        Args:
            model: TripletSimilarityModel
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
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
        """
        Train for one epoch.
        
        Returns:
            (loss, accuracy) for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in self.train_loader:
            # Move to device
            anchor = batch['anchor'].to(self.device)
            story_a = batch['story_a'].to(self.device)
            story_b = batch['story_b'].to(self.device)
            anchor_mask = batch['anchor_mask'].to(self.device)
            a_mask = batch['a_mask'].to(self.device)
            b_mask = batch['b_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(
                anchor, story_a, story_b,
                anchor_mask, a_mask, b_mask,
            )
            
            # Compute loss
            loss = compute_triplet_loss(output['logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            predictions = (output['logits'] > 0).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate on a data loader.
        
        Returns:
            (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in loader:
            # Move to device
            anchor = batch['anchor'].to(self.device)
            story_a = batch['story_a'].to(self.device)
            story_b = batch['story_b'].to(self.device)
            anchor_mask = batch['anchor_mask'].to(self.device)
            a_mask = batch['a_mask'].to(self.device)
            b_mask = batch['b_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            output = self.model(
                anchor, story_a, story_b,
                anchor_mask, a_mask, b_mask,
            )
            
            # Compute loss
            loss = compute_triplet_loss(output['logits'], labels)
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            predictions = (output['logits'] > 0).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            patience: Early stopping patience
            verbose: Print progress
        
        Returns:
            Training history
        """
        no_improve_count = 0
        
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10} {'LR':>10}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            val_loss, val_acc = self.evaluate(self.val_loader)
            
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
                
                # Save best model
                self.save_checkpoint('best_model.pt', epoch)
            else:
                no_improve_count += 1
            
            # Print progress
            if verbose:
                elapsed = time.time() - start_time
                marker = '*' if val_acc == self.best_val_acc else ''
                print(f"{epoch+1:>6} {train_loss:>12.4f} {train_acc*100:>9.2f}% "
                      f"{val_loss:>12.4f} {val_acc*100:>9.2f}%{marker} {current_lr:>10.6f}")
            
            # Early stopping
            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        print("-" * 60)
        print(f"Best validation accuracy: {self.best_val_acc*100:.2f}% at epoch {self.best_epoch+1}")
        
        # Save final model
        self.save_checkpoint('final_model.pt', epoch)
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        return checkpoint['epoch']


def create_data_loaders(
    jsonl_path: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, EventFeatureEncoder]:
    """
    Create train and validation data loaders.
    
    Returns:
        (train_loader, val_loader, feature_encoder)
    """
    # Load dataset
    drs_dataset = DRSDataset(jsonl_path)
    
    # Create feature encoder
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
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_triplets,
        num_workers=num_workers,
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
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("\n" + "=" * 60)
    print("STEP 2.6: TRAINING LOOP")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Data: {args.data_path}")
    
    # Create data loaders
    train_loader, val_loader, feature_encoder = create_data_loaders(
        args.data_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience}")
    
    history = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience,
    )
    
    # Save history
    history_path = trainer.checkpoint_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation accuracy: {trainer.best_val_acc*100:.2f}%")
    print(f"  Checkpoints saved to: {trainer.checkpoint_dir}")
    
    # Final evaluation
    print("\n" + "-" * 40)
    print("FINAL EVALUATION")
    print("-" * 40)
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    val_loss, val_acc = trainer.evaluate(val_loader)
    print(f"Best model validation accuracy: {val_acc*100:.2f}%")
    
    return trainer, history


# ============================================================
# VERIFICATION SCRIPT (for testing without full training)
# ============================================================

def verify_training_setup(jsonl_path: str):
    """Quick verification of training setup without full training."""
    
    print("\n" + "=" * 60)
    print("STEP 2.6: TRAINING SETUP VERIFICATION")
    print("=" * 60)
    
    # Create data loaders with small batch
    train_loader, val_loader, feature_encoder = create_data_loaders(
        jsonl_path,
        batch_size=8,
        val_split=0.1,
    )
    
    # Create model
    model = TripletSimilarityModel(
        input_dim=feature_encoder.feature_dim,
        hidden_dim=128,
        output_dim=64,
    )
    
    print("\n" + "-" * 40)
    print("DATA LOADER TEST")
    print("-" * 40)
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Anchor shape: {batch['anchor'].shape}")
    print(f"Story A shape: {batch['story_a'].shape}")
    print(f"Story B shape: {batch['story_b'].shape}")
    print(f"Labels: {batch['labels']}")
    
    print("\n" + "-" * 40)
    print("FORWARD PASS TEST")
    print("-" * 40)
    
    model.train()
    output = model(
        batch['anchor'],
        batch['story_a'],
        batch['story_b'],
        batch['anchor_mask'],
        batch['a_mask'],
        batch['b_mask'],
    )
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Logits: {output['logits']}")
    
    # Compute loss
    loss = compute_triplet_loss(output['logits'], batch['labels'])
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\n" + "-" * 40)
    print("MINI TRAINING TEST (3 epochs)")
    print("-" * 40)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device='cpu',
        checkpoint_dir='./checkpoints_test',
    )
    
    history = trainer.train(num_epochs=3, patience=10)
    
    print("\n" + "=" * 60)
    print("✓ TRAINING SETUP VERIFICATION COMPLETE")
    print("=" * 60)
    
    return history


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Full training: python train.py <data.jsonl> --epochs 50")
        print("  Quick verify:  python train.py <data.jsonl> --verify")
        sys.exit(1)
    
    if '--verify' in sys.argv:
        # Quick verification mode
        verify_training_setup(sys.argv[1])
    else:
        # Full training mode
        main()