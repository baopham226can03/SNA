"""
Inference and backtesting utilities for HSGNN model
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm

from dataset import HSGNNDataset
from model import create_model


class HSGNNPredictor:
    """
    Predictor class for HSGNN model
    Handles loading checkpoint and making predictions
    """
    
    def __init__(self, checkpoint_path, device='cpu'):
        """
        Args:
            checkpoint_path: path to model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.args = checkpoint['args']
        
        # Create model
        dataset_info = {
            'num_alpha_features': 158,  # Should match your data
            'num_risk_features': 3,
            'num_stocks': 498  # Adjust based on your data
        }
        
        self.model = create_model(
            dataset_info=dataset_info,
            hidden_dim=self.args.hidden_dim if hasattr(self.args, 'hidden_dim') else 64,
            num_gat_layers=self.args.num_gat_layers if hasattr(self.args, 'num_gat_layers') else 2,
            num_heads=self.args.num_heads if hasattr(self.args, 'num_heads') else 4,
            top_k=self.args.top_k if hasattr(self.args, 'top_k') else 10,
            dropout=0.0  # No dropout during inference
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val RankIC: {checkpoint['val_rank_ic']:.4f}")
    
    @torch.no_grad()
    def predict(self, batch):
        """
        Make predictions for a batch
        
        Args:
            batch: dictionary with model inputs
        
        Returns:
            predictions: numpy array of shape (B, N)
        """
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Forward pass
        predictions = self.model(batch)
        
        return predictions.cpu().numpy()
    
    @torch.no_grad()
    def predict_dataset(self, dataset):
        """
        Make predictions for entire dataset
        
        Args:
            dataset: HSGNNDataset
        
        Returns:
            predictions: numpy array of shape (T, N)
            targets: numpy array of shape (T, N)
            masks: numpy array of shape (T, N)
        """
        all_predictions = []
        all_targets = []
        all_masks = []
        
        print(f"Making predictions for {len(dataset)} samples...")
        
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            
            # Add batch dimension
            batch = {key: val.unsqueeze(0) for key, val in sample.items()}
            
            # Predict
            pred = self.predict(batch)
            
            all_predictions.append(pred[0])
            all_targets.append(sample['y_target'].numpy())
            all_masks.append(sample['mask'].numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        masks = np.array(all_masks)
        
        return predictions, targets, masks


class Backtester:
    """
    Backtesting utilities for stock predictions
    """
    
    @staticmethod
    def compute_rank_ic(predictions, targets, masks):
        """
        Compute Rank IC (Information Coefficient) over time
        
        Args:
            predictions: (T, N) predicted returns
            targets: (T, N) actual returns
            masks: (T, N) valid stock masks
        
        Returns:
            rank_ic_series: (T,) rank IC for each timestep
            mean_rank_ic: scalar mean rank IC
        """
        T, N = predictions.shape
        rank_ic_series = []
        
        for t in range(T):
            pred_t = predictions[t]
            target_t = targets[t]
            mask_t = masks[t]
            
            # Apply mask
            valid = mask_t > 0.5
            if valid.sum() < 2:  # Need at least 2 stocks
                rank_ic_series.append(np.nan)
                continue
            
            pred_valid = pred_t[valid]
            target_valid = target_t[valid]
            
            # Compute rank correlation
            from scipy.stats import spearmanr
            ic, _ = spearmanr(pred_valid, target_valid)
            rank_ic_series.append(ic)
        
        rank_ic_series = np.array(rank_ic_series)
        mean_rank_ic = np.nanmean(rank_ic_series)
        
        return rank_ic_series, mean_rank_ic
    
    @staticmethod
    def compute_sharpe_ratio(returns, annualization_factor=252):
        """
        Compute Sharpe ratio
        
        Args:
            returns: (T,) portfolio returns
            annualization_factor: 252 for daily, 12 for monthly
        
        Returns:
            sharpe_ratio: scalar
        """
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return * np.sqrt(annualization_factor)
        return sharpe
    
    @staticmethod
    def top_k_portfolio(predictions, targets, masks, k=20):
        """
        Create portfolio by selecting top-k predicted stocks
        
        Args:
            predictions: (T, N) predicted returns
            targets: (T, N) actual returns
            masks: (T, N) valid masks
            k: number of stocks to select
        
        Returns:
            portfolio_returns: (T,) portfolio returns
            selected_stocks: (T, k) indices of selected stocks
        """
        T, N = predictions.shape
        portfolio_returns = []
        selected_stocks = []
        
        for t in range(T):
            pred_t = predictions[t]
            target_t = targets[t]
            mask_t = masks[t]
            
            # Apply mask
            valid = mask_t > 0.5
            if valid.sum() < k:
                portfolio_returns.append(0.0)
                selected_stocks.append(np.zeros(k, dtype=int))
                continue
            
            # Get valid predictions
            pred_valid = pred_t.copy()
            pred_valid[~valid] = -np.inf
            
            # Select top-k
            top_k_indices = np.argsort(pred_valid)[-k:]
            
            # Equal weight portfolio
            portfolio_return = np.mean(target_t[top_k_indices])
            
            portfolio_returns.append(portfolio_return)
            selected_stocks.append(top_k_indices)
        
        portfolio_returns = np.array(portfolio_returns)
        
        return portfolio_returns, selected_stocks
    
    @staticmethod
    def generate_report(predictions, targets, masks, sorted_tickers=None):
        """
        Generate comprehensive backtest report
        
        Args:
            predictions: (T, N) predicted returns
            targets: (T, N) actual returns
            masks: (T, N) valid masks
            sorted_tickers: list of ticker names
        
        Returns:
            report: dictionary with metrics
        """
        print("="*80)
        print("BACKTEST REPORT")
        print("="*80)
        
        # Rank IC
        rank_ic_series, mean_rank_ic = Backtester.compute_rank_ic(predictions, targets, masks)
        ic_std = np.nanstd(rank_ic_series)
        
        print(f"\n📊 Information Coefficient (IC):")
        print(f"  Mean Rank IC: {mean_rank_ic:.4f}")
        print(f"  Std Rank IC: {ic_std:.4f}")
        print(f"  IC IR (IC/std): {mean_rank_ic/ic_std:.4f}" if ic_std > 0 else "  IC IR: N/A")
        
        # Top-K portfolio strategies
        print(f"\n💰 Portfolio Performance (Top-K Long-only):")
        
        for k in [10, 20, 50]:
            portfolio_returns, _ = Backtester.top_k_portfolio(predictions, targets, masks, k=k)
            
            # Compute metrics
            mean_return = np.mean(portfolio_returns) * 252  # Annualized
            std_return = np.std(portfolio_returns) * np.sqrt(252)
            sharpe = Backtester.compute_sharpe_ratio(portfolio_returns)
            cumulative_return = np.cumprod(1 + portfolio_returns)[-1] - 1
            
            print(f"\n  Top-{k} Portfolio:")
            print(f"    Annualized Return: {mean_return*100:.2f}%")
            print(f"    Annualized Volatility: {std_return*100:.2f}%")
            print(f"    Sharpe Ratio: {sharpe:.4f}")
            print(f"    Cumulative Return: {cumulative_return*100:.2f}%")
        
        # Long-short portfolio (Top 50 long, Bottom 50 short)
        print(f"\n📈 Long-Short Portfolio (Top 50 Long / Bottom 50 Short):")
        
        T, N = predictions.shape
        ls_returns = []
        
        for t in range(T):
            pred_t = predictions[t]
            target_t = targets[t]
            mask_t = masks[t]
            
            valid = mask_t > 0.5
            if valid.sum() < 100:
                ls_returns.append(0.0)
                continue
            
            pred_valid = pred_t.copy()
            pred_valid[~valid] = 0
            
            # Long top 50
            top_50 = np.argsort(pred_valid)[-50:]
            long_return = np.mean(target_t[top_50])
            
            # Short bottom 50
            bottom_50 = np.argsort(pred_valid)[:50]
            short_return = -np.mean(target_t[bottom_50])
            
            ls_return = (long_return + short_return) / 2
            ls_returns.append(ls_return)
        
        ls_returns = np.array(ls_returns)
        
        mean_ls = np.mean(ls_returns) * 252
        std_ls = np.std(ls_returns) * np.sqrt(252)
        sharpe_ls = Backtester.compute_sharpe_ratio(ls_returns)
        cumulative_ls = np.cumprod(1 + ls_returns)[-1] - 1
        
        print(f"  Annualized Return: {mean_ls*100:.2f}%")
        print(f"  Annualized Volatility: {std_ls*100:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ls:.4f}")
        print(f"  Cumulative Return: {cumulative_ls*100:.2f}%")
        
        print("\n" + "="*80)
        
        report = {
            'mean_rank_ic': mean_rank_ic,
            'ic_std': ic_std,
            'rank_ic_series': rank_ic_series,
            'top10_sharpe': Backtester.compute_sharpe_ratio(
                Backtester.top_k_portfolio(predictions, targets, masks, k=10)[0]
            ),
            'top20_sharpe': Backtester.compute_sharpe_ratio(
                Backtester.top_k_portfolio(predictions, targets, masks, k=20)[0]
            ),
            'top50_sharpe': Backtester.compute_sharpe_ratio(
                Backtester.top_k_portfolio(predictions, targets, masks, k=50)[0]
            ),
            'long_short_sharpe': sharpe_ls
        }
        
        return report


if __name__ == '__main__':
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--window_size', type=int, default=300,
                        help='Window size')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    print("="*80)
    print("HSGNN BACKTESTING")
    print("="*80)
    
    # Load predictor
    predictor = HSGNNPredictor(args.checkpoint, device=args.device)
    
    # Create test dataset
    print(f"\nCreating test dataset...")
    test_dataset = HSGNNDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        prediction_horizon=1,
        train_start=1200,  # Use last portion as test
        train_end=None,
        normalize=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Make predictions
    predictions, targets, masks = predictor.predict_dataset(test_dataset)
    
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Load tickers
    with open('data/graph_data/sorted_tickers.pkl', 'rb') as f:
        sorted_tickers = pickle.load(f)
    
    # Generate report
    report = Backtester.generate_report(predictions, targets, masks, sorted_tickers)
    
    # Save results
    output_path = Path(args.checkpoint).parent / 'backtest_results.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'predictions': predictions,
            'targets': targets,
            'masks': masks,
            'report': report
        }, f)
    
    print(f"\n✓ Results saved to {output_path}")
