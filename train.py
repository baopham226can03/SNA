"""
Training script for HSGNN model

Usage:
    python train.py --config config.yaml
    python train.py --epochs 100 --batch_size 32 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import time

from dataset import create_dataloaders
from model import create_model


class RankIC:
    """Rank Information Coefficient - main metric for stock prediction"""
    
    @staticmethod
    def compute(predictions, targets, mask=None):
        """
        Compute Rank IC (Spearman correlation)
        
        Args:
            predictions: (B, N) predicted returns
            targets: (B, N) actual returns
            mask: (B, N) valid stock mask
        
        Returns:
            rank_ic: scalar rank IC value
        """
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
        
        # Flatten
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        if mask is not None:
            mask_flat = mask.flatten()
            valid = mask_flat > 0.5
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]
        
        # Remove NaN
        valid = ~torch.isnan(pred_flat) & ~torch.isnan(target_flat)
        pred_flat = pred_flat[valid]
        target_flat = target_flat[valid]
        
        if len(pred_flat) == 0:
            return 0.0
        
        # Compute Spearman correlation (rank correlation)
        pred_rank = pred_flat.argsort().argsort().float()
        target_rank = target_flat.argsort().argsort().float()
        
        # Pearson correlation of ranks
        pred_rank = pred_rank - pred_rank.mean()
        target_rank = target_rank - target_rank.mean()
        
        numerator = (pred_rank * target_rank).sum()
        denominator = torch.sqrt((pred_rank ** 2).sum() * (target_rank ** 2).sum()) + 1e-8
        
        rank_ic = numerator / denominator
        
        return rank_ic.item()


def masked_mse_loss(predictions, targets, mask):
    """
    MSE loss with masking for invalid stocks
    
    Args:
        predictions: (B, N) predicted returns
        targets: (B, N) actual returns
        mask: (B, N) valid stock mask
    
    Returns:
        loss: scalar MSE loss
    """
    # Apply mask
    predictions_masked = predictions * mask
    targets_masked = targets * mask
    
    # Compute MSE only on valid entries
    squared_error = (predictions_masked - targets_masked) ** 2
    num_valid = mask.sum() + 1e-8
    
    loss = squared_error.sum() / num_valid
    
    return loss


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_rank_ic = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch in pbar:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch)
        
        # Compute loss
        loss = masked_mse_loss(predictions, batch['y_target'], batch['mask'])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            rank_ic = RankIC.compute(predictions, batch['y_target'], batch['mask'])
        
        total_loss += loss.item()
        total_rank_ic += rank_ic
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'rank_ic': f'{rank_ic:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_rank_ic = total_rank_ic / num_batches
    
    return avg_loss, avg_rank_ic


@torch.no_grad()
def validate(model, val_loader, device, epoch):
    """Validate on validation set"""
    model.eval()
    
    total_loss = 0
    total_rank_ic = 0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    for batch in pbar:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        predictions = model(batch)
        
        # Compute loss
        loss = masked_mse_loss(predictions, batch['y_target'], batch['mask'])
        
        # Metrics
        rank_ic = RankIC.compute(predictions, batch['y_target'], batch['mask'])
        
        total_loss += loss.item()
        total_rank_ic += rank_ic
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'rank_ic': f'{rank_ic:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_rank_ic = total_rank_ic / num_batches
    
    return avg_loss, avg_rank_ic


def train(args):
    """Main training loop"""
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataloaders
    print("\n" + "="*80)
    print("Creating dataloaders...")
    print("="*80)
    
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        data_dir=args.data_dir,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)
    
    model = create_model(
        dataset_info=dataset_info,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        top_k=args.top_k,
        dropout=args.dropout
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    best_val_rank_ic = -float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_rank_ic = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, val_rank_ic = validate(model, val_loader, device, epoch)
        
        # Scheduler step
        scheduler.step(val_rank_ic)
        
        # Logging
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}, Train RankIC: {train_rank_ic:.4f}")
        print(f"  Val Loss: {val_loss:.6f}, Val RankIC: {val_rank_ic:.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('RankIC/train', train_rank_ic, epoch)
        writer.add_scalar('RankIC/val', val_rank_ic, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save best model
        if val_rank_ic > best_val_rank_ic:
            best_val_rank_ic = val_rank_ic
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rank_ic': val_rank_ic,
                'val_loss': val_loss,
                'args': args
            }, output_dir / 'best_model.pt')
            
            print(f"  ✓ Saved best model (RankIC: {val_rank_ic:.4f})")
        else:
            patience_counter += 1
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rank_ic': val_rank_ic,
                'val_loss': val_loss,
                'args': args
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n Early stopping triggered after {epoch} epochs")
            break
    
    print("\n" + "="*80)
    print(f"Training completed!")
    print(f"Best model at epoch {best_epoch} with Val RankIC: {best_val_rank_ic:.4f}")
    print("="*80)
    
    # Test on best model
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_rank_ic = validate(model, test_loader, device, epoch='Test')
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test RankIC: {test_rank_ic:.4f}")
    
    # Save test results
    results = {
        'best_epoch': best_epoch,
        'best_val_rank_ic': best_val_rank_ic,
        'test_loss': test_loss,
        'test_rank_ic': test_rank_ic
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    writer.close()
    
    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HSGNN model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--window_size', type=int, default=300, help='Input window size')
    parser.add_argument('--prediction_horizon', type=int, default=1, help='Prediction horizon')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_gat_layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K edges for implicit graph')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    train(args)
