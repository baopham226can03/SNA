"""
Training script for HSGNN with LLM Dynamic Graph

This is an enhanced version of train.py that uses LLM to build
dynamic stock relationship graphs.

Usage:
    # Without LLM (rule-based dynamic graphs)
    python train_llm_enhanced.py --epochs 20 --use_llm False
    
    # With LLM (requires API key)
    python train_llm_enhanced.py --epochs 20 --use_llm True --llm_provider openai
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
from model_llm_dynamic_graph import create_model_llm_dynamic


class RankIC:
    """Rank Information Coefficient - main metric"""
    
    @staticmethod
    def compute(predictions, targets, mask=None):
        """Compute Rank IC (Spearman correlation)"""
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
        
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        if mask is not None:
            mask_flat = mask.flatten()
            valid = mask_flat > 0.5
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]
        
        # Remove NaN/Inf
        valid_mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
        pred_flat = pred_flat[valid_mask]
        target_flat = target_flat[valid_mask]
        
        if len(pred_flat) < 2:
            return torch.tensor(0.0)
        
        # Spearman correlation
        pred_rank = pred_flat.argsort().argsort().float()
        target_rank = target_flat.argsort().argsort().float()
        
        pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + 1e-8)
        target_rank = (target_rank - target_rank.mean()) / (target_rank.std() + 1e-8)
        
        rank_ic = (pred_rank * target_rank).mean()
        
        return rank_ic


def masked_mse_loss(predictions, targets, mask):
    """MSE loss with masking"""
    if mask is None:
        return F.mse_loss(predictions, targets)
    
    loss = (predictions - targets) ** 2
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_rank_ic = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        targets = batch['y']
        mask = batch['mask']
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch)
        
        # Compute loss
        loss = masked_mse_loss(predictions, targets, mask)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (NEW - improve stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            rank_ic = RankIC.compute(predictions, targets, mask)
        
        total_loss += loss.item()
        total_rank_ic += rank_ic.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'rank_ic': f'{rank_ic.item():.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_rank_ic = total_rank_ic / num_batches
    
    return avg_loss, avg_rank_ic


def validate(model, val_loader, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0.0
    total_rank_ic = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Epoch [Val]")
        
        for batch in pbar:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            targets = batch['y']
            mask = batch['mask']
            
            # Forward pass
            predictions = model(batch)
            
            # Compute metrics
            loss = masked_mse_loss(predictions, targets, mask)
            rank_ic = RankIC.compute(predictions, targets, mask)
            
            total_loss += loss.item()
            total_rank_ic += rank_ic.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'rank_ic': f'{rank_ic.item():.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_rank_ic = total_rank_ic / num_batches
    
    return avg_loss, avg_rank_ic


def train(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "="*80)
    print("Creating dataloaders...")
    print("="*80)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        window_size=args.window_size,
        num_workers=args.num_workers
    )
    
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)
    print(f"LLM Enhanced: {args.use_llm}")
    if args.use_llm:
        print(f"LLM Provider: {args.llm_provider}")
    print()
    
    # Create model
    model = create_model_llm_dynamic(
        num_alpha_features=119,
        num_risk_features=3,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        top_k=args.top_k,
        dropout=args.dropout,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        device=device
    )
    
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
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_rank_ic = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, val_rank_ic = validate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('RankIC/train', train_rank_ic, epoch)
        writer.add_scalar('RankIC/val', val_rank_ic, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Scheduler step
        scheduler.step(val_rank_ic)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.6f}, Train RankIC: {train_rank_ic:.4f}")
        print(f"  Val Loss: {val_loss:.6f}, Val RankIC: {val_rank_ic:.4f}")
        
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
            }, output_dir / 'best_model.pt')
            
            print(f"  ✓ Saved best model (RankIC: {val_rank_ic:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {epoch} epochs (best epoch: {best_epoch})")
            break
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch{epoch}.pt')
    
    writer.close()
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation RankIC: {best_val_rank_ic:.4f} (epoch {best_epoch})")
    print(f"Models saved to: {output_dir}")
    print("="*80)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_rank_ic = validate(model, test_loader, device)
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test RankIC: {test_rank_ic:.4f}")
    
    # Save test results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'test_rank_ic': test_rank_ic,
            'best_epoch': best_epoch,
            'best_val_rank_ic': best_val_rank_ic
        }, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HSGNN with LLM Dynamic Graph')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--window_size', type=int, default=150, help='Sliding window size')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_gat_layers', type=int, default=2, help='Number of GAT layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K edges for implicit graph')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # LLM parameters (NEW)
    parser.add_argument('--use_llm', type=bool, default=False, help='Use LLM for dynamic graphs')
    parser.add_argument('--llm_provider', type=str, default='local', 
                       choices=['local', 'openai', 'anthropic'],
                       help='LLM provider')
    
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
    parser.add_argument('--output_dir', type=str, default='outputs/hsgnn_llm_enhanced', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    train(args)
