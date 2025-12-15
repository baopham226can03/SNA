"""
Script để lấy thông tin GICS Sector cho các cổ phiếu S&P 500 và cập nhật sector_adj_matrix
"""

import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from pathlib import Path
from tqdm import tqdm
import time

# Đường dẫn
SORTED_TICKERS_PATH = "data/graph_data/sorted_tickers.pkl"
OUTPUT_DIR = "data/graph_data"
SECTOR_INFO_PATH = f"{OUTPUT_DIR}/sector_info.csv"

print("="*80)
print("FETCHING SECTOR INFORMATION FOR S&P 500 STOCKS")
print("="*80)

# Load sorted tickers
print("\n[1/3] Loading sorted tickers...")
with open(SORTED_TICKERS_PATH, 'rb') as f:
    sorted_tickers = pickle.load(f)

print(f"   ✓ Loaded {len(sorted_tickers)} tickers")

# Fetch sector information
print("\n[2/3] Fetching sector information from Yahoo Finance...")
print("   (This may take a few minutes...)")

sector_data = []

for i, ticker in enumerate(sorted_tickers):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        sector_data.append({
            'ticker': ticker,
            'sector': sector,
            'industry': industry
        })
        
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(sorted_tickers)} tickers...")
        
        # Rate limiting
        time.sleep(0.1)
        
    except Exception as e:
        print(f"   ! Warning: Error fetching {ticker}: {e}")
        sector_data.append({
            'ticker': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown'
        })

# Create DataFrame
sector_df = pd.DataFrame(sector_data)
print(f"\n   ✓ Fetched sector info for {len(sector_df)} tickers")

# Save sector info
sector_df.to_csv(SECTOR_INFO_PATH, index=False)
print(f"   ✓ Saved to {SECTOR_INFO_PATH}")

# Display statistics
print(f"\n   📊 Sector distribution:")
sector_counts = sector_df['sector'].value_counts()
for sector, count in sector_counts.head(15).items():
    print(f"      {sector}: {count}")

# Rebuild sector adjacency matrix
print("\n[3/3] Rebuilding sector_adj_matrix with actual sector data...")

num_stocks = len(sorted_tickers)
sector_adj_matrix = np.zeros((num_stocks, num_stocks))

# Create ticker to sector mapping
ticker_to_sector = dict(zip(sector_df['ticker'], sector_df['sector']))

# Build adjacency matrix
for i, ticker_i in enumerate(sorted_tickers):
    for j, ticker_j in enumerate(sorted_tickers):
        if i == j:
            # Self-loop
            sector_adj_matrix[i, j] = 1
        else:
            # Check if same sector
            sector_i = ticker_to_sector.get(ticker_i, 'Unknown')
            sector_j = ticker_to_sector.get(ticker_j, 'Unknown')
            
            if sector_i != 'Unknown' and sector_j != 'Unknown' and sector_i == sector_j:
                sector_adj_matrix[i, j] = 1

print(f"   ✓ Rebuilt sector_adj_matrix")
print(f"   Shape: {sector_adj_matrix.shape}")

num_edges = int(sector_adj_matrix.sum() - num_stocks)
print(f"   Số edges (không kể self-loops): {num_edges:,}")
print(f"   Trung bình số kết nối/node: {num_edges / num_stocks:.1f}")
print(f"   Density: {sector_adj_matrix.sum() / (num_stocks * num_stocks):.4f}")

# Save updated matrix
np.save(f"{OUTPUT_DIR}/sector_adj_matrix.npy", sector_adj_matrix)
with open(f"{OUTPUT_DIR}/sector_adj_matrix.pkl", 'wb') as f:
    pickle.dump(sector_adj_matrix, f)

print(f"\n   ✓ Saved updated sector_adj_matrix.npy and .pkl")

print("\n" + "="*80)
print("✅ COMPLETED! Sector adjacency matrix has been updated.")
print("="*80)

# Display some example connections
print("\n💡 Example connections (first 5 tickers):")
for i in range(min(5, num_stocks)):
    ticker = sorted_tickers[i]
    sector = ticker_to_sector.get(ticker, 'Unknown')
    connections = [sorted_tickers[j] for j in range(num_stocks) if sector_adj_matrix[i, j] == 1 and i != j]
    print(f"   {ticker} ({sector}): {len(connections)} connections")
    if len(connections) > 0:
        print(f"      e.g., {', '.join(connections[:5])}")
