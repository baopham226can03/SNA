"""
HSGNNDataset - Dataset class for HSGNN model
Loads and prepares data for training the Hybrid Stock Graph Neural Network

Based on paper: "Modeling hybrid firm relationships with graph neural networks 
for stock investment decisions"
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import pickle


class HSGNNDataset(Dataset):
    """
    PyTorch Dataset for HSGNN model
    
    Loads:
    - alpha158_features: Node features (X_alpha) - shape (T, N, 158)
    - alpha158_labels: Target returns - shape (T, N)
    - risk_features: Risk features (X_barra) for implicit graph learning - shape (T, N, 3)
    - sector_adj_matrix: Static sector adjacency matrix (G_sc proxy) - shape (N, N)
    - money_flow_matrix: Money flow correlation matrix (G_mf proxy) - shape (N, N)
    
    Returns samples with sliding window of size window_size
    """
    
    def __init__(self, 
                 data_dir='data',
                 window_size=300,
                 prediction_horizon=1,
                 train_start=0,
                 train_end=None,
                 normalize=True):
        """
        Args:
            data_dir: str, directory containing data files
            window_size: int, number of days for input window (default 300)
            prediction_horizon: int, number of days ahead to predict (default 1)
            train_start: int, starting index for this split
            train_end: int, ending index for this split (None = use all)
            normalize: bool, whether to normalize features
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        
        print("="*80)
        print("Loading HSGNN Dataset...")
        print("="*80)
        
        # Load sorted tickers (for alignment reference)
        with open(self.data_dir / 'graph_data' / 'sorted_tickers.pkl', 'rb') as f:
            self.sorted_tickers = pickle.load(f)
        print(f"✓ Loaded {len(self.sorted_tickers)} tickers")
        
        # Load alpha158 features (X_alpha) - Node features for explicit graph
        print("\n[1/5] Loading alpha158_features...")
        print("   This may take a few minutes for large files...")
        
        # Use chunking for large files
        chunksize = 100000
        chunks = []
        try:
            for chunk in pd.read_csv(self.data_dir / 'alpha158' / 'alpha158_features.csv', 
                                    chunksize=chunksize):
                chunks.append(chunk)
                if len(chunks) % 10 == 0:
                    print(f"   Loaded {len(chunks) * chunksize} rows...")
            alpha_df = pd.concat(chunks, ignore_index=True)
            print(f"   ✓ Loaded total {len(alpha_df)} rows")
        except Exception as e:
            print(f"   Error: {e}")
            raise
        
        # Initialize dimensions
        self.num_stocks = len(self.sorted_tickers)
        
        # Detect column names (handle both 'ticker'/'date' and 'instrument'/'datetime')
        ticker_col = 'ticker' if 'ticker' in alpha_df.columns else 'instrument'
        date_col = 'date' if 'date' in alpha_df.columns else 'datetime'
        
        if ticker_col in alpha_df.columns and date_col in alpha_df.columns:
            # Long format - pivot to (T, N, F)
            alpha_df[date_col] = pd.to_datetime(alpha_df[date_col])
            
            # Get feature columns (exclude ticker and date)
            feature_cols = [c for c in alpha_df.columns if c not in [ticker_col, date_col]]
            self.num_alpha_features = len(feature_cols)
            
            print(f"   Converting from long format using vectorized operations...")
            
            # Create ticker to index mapping
            ticker_to_idx = {ticker: idx for idx, ticker in enumerate(self.sorted_tickers)}
            
            # Add ticker index and date index to dataframe
            alpha_df['ticker_idx'] = alpha_df[ticker_col].map(ticker_to_idx)
            
            # Get unique dates and create date mapping
            dates = sorted(alpha_df[date_col].unique())
            self.num_dates = len(dates)
            date_to_idx = {date: idx for idx, date in enumerate(dates)}
            alpha_df['date_idx'] = alpha_df[date_col].map(date_to_idx)
            
            # Initialize array
            self.alpha_features = np.zeros((self.num_dates, self.num_stocks, self.num_alpha_features))
            
            # Filter out rows with unknown tickers
            alpha_df = alpha_df[alpha_df['ticker_idx'].notna()].copy()
            
            # Vectorized assignment using numpy advanced indexing
            print(f"   Filling array for {self.num_dates} dates, {self.num_stocks} stocks, {self.num_alpha_features} features...")
            date_indices = alpha_df['date_idx'].astype(int).values
            ticker_indices = alpha_df['ticker_idx'].astype(int).values
            feature_values = alpha_df[feature_cols].values
            
            self.alpha_features[date_indices, ticker_indices, :] = feature_values
        else:
            raise ValueError(f"Cannot find ticker and date columns in alpha158_features. Found columns: {alpha_df.columns.tolist()}")
        
        print(f"   Alpha features shape: {self.alpha_features.shape}")
        
        # Load alpha158 labels (Target returns)
        print("\n[2/5] Loading alpha158_labels...")
        
        # Use chunking for large files
        chunks = []
        try:
            for chunk in pd.read_csv(self.data_dir / 'alpha158' / 'alpha158_labels.csv',
                                    chunksize=chunksize):
                chunks.append(chunk)
            labels_df = pd.concat(chunks, ignore_index=True)
            print(f"   ✓ Loaded total {len(labels_df)} rows")
        except Exception as e:
            print(f"   Error: {e}")
            raise
        
        # Detect column names
        ticker_col = 'ticker' if 'ticker' in labels_df.columns else 'instrument'
        date_col = 'date' if 'date' in labels_df.columns else 'datetime'
        label_col = 'label' if 'label' in labels_df.columns else [c for c in labels_df.columns if c.startswith('LABEL')][0]
        
        if ticker_col in labels_df.columns and date_col in labels_df.columns:
            # Long format - pivot to (T, N)
            labels_df[date_col] = pd.to_datetime(labels_df[date_col])
            
            print(f"   Converting labels using vectorized operations...")
            
            # Add ticker and date indices
            labels_df['ticker_idx'] = labels_df[ticker_col].map(ticker_to_idx)
            labels_df['date_idx'] = labels_df[date_col].map(date_to_idx)
            
            # Initialize labels array
            self.labels = np.zeros((self.num_dates, self.num_stocks))
            
            # Filter out rows with unknown tickers or dates
            labels_df = labels_df[labels_df['ticker_idx'].notna() & labels_df['date_idx'].notna()].copy()
            
            # Vectorized assignment using numpy advanced indexing
            date_indices = labels_df['date_idx'].astype(int).values
            ticker_indices = labels_df['ticker_idx'].astype(int).values
            label_values = labels_df[label_col].values
            
            self.labels[date_indices, ticker_indices] = label_values
        else:
            raise ValueError(f"Cannot find ticker and date columns in alpha158_labels. Found columns: {labels_df.columns.tolist()}")
        
        print(f"   Labels shape: {self.labels.shape}")
        
        # Load risk features (X_barra) - For implicit graph learning
        print("\n[3/5] Loading risk_features...")
        self.risk_features = np.load(self.data_dir / 'graph_data' / 'risk_features.npy')
        print(f"   Risk features shape: {self.risk_features.shape}")
        
        # Load sector adjacency matrix (G_sc proxy)
        print("\n[4/5] Loading sector_adj_matrix...")
        self.sector_adj_matrix = np.load(self.data_dir / 'graph_data' / 'sector_adj_matrix.npy')
        print(f"   Sector adjacency shape: {self.sector_adj_matrix.shape}")
        
        # Load money flow matrix (G_mf proxy for edge filtering)
        print("\n[5/5] Loading money_flow_matrix...")
        self.money_flow_matrix = np.load(self.data_dir / 'graph_data' / 'money_flow_matrix.npy')
        print(f"   Money flow matrix shape: {self.money_flow_matrix.shape}")
        
        # Update dimensions
        self.num_dates = self.risk_features.shape[0]
        self.num_stocks = self.risk_features.shape[1]
        self.num_risk_features = self.risk_features.shape[2]
        
        print(f"\n📊 Dataset dimensions:")
        print(f"   Time steps: {self.num_dates}")
        print(f"   Stocks: {self.num_stocks}")
        print(f"   Alpha features: {self.num_alpha_features}")
        print(f"   Risk features: {self.num_risk_features}")
        
        # Handle train/test split
        self.train_start = train_start
        self.train_end = train_end if train_end is not None else self.num_dates
        
        # Calculate valid indices (need window_size history + prediction_horizon future)
        self.valid_indices = []
        for i in range(self.train_start + window_size, self.train_end - prediction_horizon + 1):
            self.valid_indices.append(i)
        
        print(f"\n✓ Valid samples: {len(self.valid_indices)}")
        print(f"   Window size: {window_size}")
        print(f"   Prediction horizon: {prediction_horizon}")
        print(f"   Range: [{self.train_start}, {self.train_end})")
        
        # Normalization statistics (computed on training data)
        if normalize:
            print("\n[Normalization] Computing statistics...")
            self._compute_normalization_stats()
            print("   ✓ Normalization ready")
        
        print("\n" + "="*80)
        print("✓ Dataset loaded successfully!")
        print("="*80)
    
    def _compute_normalization_stats(self):
        """
        Compute mean and std for feature normalization
        Only computed on training portion to avoid data leakage
        """
        # Get training data only (before first valid index)
        train_end_idx = self.valid_indices[0] if len(self.valid_indices) > 0 else self.train_end
        
        # Alpha features normalization
        train_alpha = self.alpha_features[:train_end_idx]
        self.alpha_mean = np.nanmean(train_alpha, axis=(0, 1), keepdims=True)
        self.alpha_std = np.nanstd(train_alpha, axis=(0, 1), keepdims=True) + 1e-8
        
        # Risk features normalization
        train_risk = self.risk_features[:train_end_idx]
        self.risk_mean = np.nanmean(train_risk, axis=(0, 1), keepdims=True)
        self.risk_std = np.nanstd(train_risk, axis=(0, 1), keepdims=True) + 1e-8
    
    def _normalize_features(self, alpha_feat, risk_feat):
        """Apply z-score normalization"""
        if self.normalize:
            alpha_feat = (alpha_feat - self.alpha_mean) / self.alpha_std
            risk_feat = (risk_feat - self.risk_mean) / self.risk_std
        
        # Replace NaN/Inf with 0
        alpha_feat = np.nan_to_num(alpha_feat, nan=0.0, posinf=0.0, neginf=0.0)
        risk_feat = np.nan_to_num(risk_feat, nan=0.0, posinf=0.0, neginf=0.0)
        
        return alpha_feat, risk_feat
    
    def __len__(self):
        """Return number of valid samples"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Get a sample with sliding window
        
        Returns:
            dict with keys:
                - x_alpha: (window_size, N, F_alpha) - Alpha158 features for explicit graph
                - x_risk: (window_size, N, F_risk) - Risk features for implicit graph
                - sector_graph: (N, N) - Sector adjacency matrix (static)
                - money_flow_graph: (N, N) - Money flow correlation matrix (for edge filtering)
                - y_target: (N,) - Target returns at t+prediction_horizon
                - mask: (N,) - Valid stock mask (1 if valid, 0 if missing data)
        """
        # Get the prediction time index
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.window_size
        target_idx = end_idx + self.prediction_horizon - 1
        
        # Extract window of features
        alpha_window = self.alpha_features[start_idx:end_idx].copy()  # (window, N, F_alpha)
        risk_window = self.risk_features[start_idx:end_idx].copy()    # (window, N, F_risk)
        
        # Get target labels
        y_target = self.labels[target_idx].copy()  # (N,)
        
        # Normalize features
        alpha_window, risk_window = self._normalize_features(alpha_window, risk_window)
        
        # Create mask for valid stocks (not NaN in target)
        mask = ~np.isnan(y_target)
        y_target = np.nan_to_num(y_target, nan=0.0)
        
        # Convert to tensors
        sample = {
            'x_alpha': torch.FloatTensor(alpha_window),           # (window, N, F_alpha)
            'x_risk': torch.FloatTensor(risk_window),             # (window, N, F_risk)
            'sector_graph': torch.FloatTensor(self.sector_adj_matrix),  # (N, N)
            'money_flow_graph': torch.FloatTensor(self.money_flow_matrix),  # (N, N)
            'y_target': torch.FloatTensor(y_target),              # (N,)
            'mask': torch.FloatTensor(mask.astype(np.float32)),   # (N,)
        }
        
        return sample
    
    def get_num_features(self):
        """Return feature dimensions"""
        return {
            'num_alpha_features': self.num_alpha_features,
            'num_risk_features': self.num_risk_features,
            'num_stocks': self.num_stocks
        }


def create_dataloaders(data_dir='data',
                       window_size=300,
                       prediction_horizon=1,
                       train_ratio=0.6,
                       val_ratio=0.2,
                       batch_size=32,
                       num_workers=0):
    """
    Create train/val/test dataloaders with temporal split
    
    Args:
        data_dir: str, directory containing data
        window_size: int, input window size
        prediction_horizon: int, prediction horizon
        train_ratio: float, proportion for training
        val_ratio: float, proportion for validation
        batch_size: int, batch size
        num_workers: int, number of workers for dataloader
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    # First, create a temporary dataset to get total length
    temp_dataset = HSGNNDataset(
        data_dir=data_dir,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        train_start=0,
        train_end=None,
        normalize=False
    )
    
    total_dates = temp_dataset.num_dates
    
    # Calculate split points (temporal split)
    train_end = int(total_dates * train_ratio)
    val_end = int(total_dates * (train_ratio + val_ratio))
    
    print("\n" + "="*80)
    print("Creating dataloaders with temporal split...")
    print("="*80)
    print(f"Total dates: {total_dates}")
    print(f"Train: [0, {train_end})")
    print(f"Val: [{train_end}, {val_end})")
    print(f"Test: [{val_end}, {total_dates})")
    
    # Create datasets for each split
    train_dataset = HSGNNDataset(
        data_dir=data_dir,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        train_start=0,
        train_end=train_end,
        normalize=True
    )
    
    val_dataset = HSGNNDataset(
        data_dir=data_dir,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        train_start=train_end,
        train_end=val_end,
        normalize=True
    )
    
    test_dataset = HSGNNDataset(
        data_dir=data_dir,
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        train_start=val_end,
        train_end=total_dates,
        normalize=True
    )
    
    # Copy normalization stats from train to val/test
    val_dataset.alpha_mean = train_dataset.alpha_mean
    val_dataset.alpha_std = train_dataset.alpha_std
    val_dataset.risk_mean = train_dataset.risk_mean
    val_dataset.risk_std = train_dataset.risk_std
    
    test_dataset.alpha_mean = train_dataset.alpha_mean
    test_dataset.alpha_std = train_dataset.alpha_std
    test_dataset.risk_mean = train_dataset.risk_mean
    test_dataset.risk_std = train_dataset.risk_std
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Dataloaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    dataset_info = train_dataset.get_num_features()
    
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == '__main__':
    """Test the dataset"""
    print("Testing HSGNNDataset...")
    
    # Create dataset
    dataset = HSGNNDataset(
        data_dir='data',
        window_size=300,
        prediction_horizon=1,
        normalize=True
    )
    
    # Get a sample
    sample = dataset[0]
    
    print("\n" + "="*80)
    print("Sample data shapes:")
    print("="*80)
    for key, value in sample.items():
        print(f"  {key:20s}: {value.shape}")
    
    print("\n" + "="*80)
    print("Sample statistics:")
    print("="*80)
    print(f"  x_alpha mean: {sample['x_alpha'].mean():.6f}, std: {sample['x_alpha'].std():.6f}")
    print(f"  x_risk mean: {sample['x_risk'].mean():.6f}, std: {sample['x_risk'].std():.6f}")
    print(f"  y_target mean: {sample['y_target'].mean():.6f}, std: {sample['y_target'].std():.6f}")
    print(f"  Valid stocks: {sample['mask'].sum().item():.0f} / {sample['mask'].shape[0]}")
    
    # Test dataloader creation
    print("\n" + "="*80)
    print("Testing dataloader creation...")
    print("="*80)
    
    train_loader, val_loader, test_loader, info = create_dataloaders(
        data_dir='data',
        window_size=300,
        prediction_horizon=1,
        batch_size=8
    )
    
    # Test iteration
    batch = next(iter(train_loader))
    print("\n✓ Batch shapes:")
    for key, value in batch.items():
        print(f"  {key:20s}: {value.shape}")
    
    print("\n✓ Dataset test completed successfully!")
