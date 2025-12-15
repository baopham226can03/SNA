"""
Script sinh ra 3 thành phần dữ liệu cho mô hình HSGNN:
1. risk_features: Thay cho Barra Risk Factors (Volatility, Momentum, Turnover)
2. sector_adj_matrix: Thay cho Supply Chain Graph (dựa trên GICS Sector)
3. money_flow_matrix: Thay cho Level-2 Money Flow Graph (dựa trên Money Flow correlation)

Author: Generated for HSGNN reproduction project
Date: 2025-12-13
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Đường dẫn dữ liệu
SP500_DATA_PATH = "data/qlib_format/sp500_data.csv"
SP500_HISTORY_PATH = "data/sp500_history.csv"
OUTPUT_DIR = "data/graph_data"

# Tham số tính toán
ROLLING_WINDOW = 20  # Số ngày cho tính toán volatility, momentum, turnover
CORRELATION_WINDOW = 30  # Số ngày cho tính toán money flow correlation

# Tạo thư mục output
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*80)
print("HSGNN GRAPH BUILDER - S&P 500 REPRODUCTION")
print("="*80)

# ============================================================================
# BƯỚC 1: TẠO SORTED_TICKERS (Alignment Reference)
# ============================================================================

print("\n[1/4] Đang tải dữ liệu và tạo sorted_tickers...")

try:
    # Đọc sp500_data để lấy danh sách tickers và sector information
    sp500_data = pd.read_csv(SP500_DATA_PATH)
    print(f"   ✓ Đã đọc sp500_data: {sp500_data.shape}")
    print(f"   Cột: {sp500_data.columns.tolist()}")
    
    # Lấy unique tickers và sắp xếp theo bảng chữ cái
    if 'instrument' in sp500_data.columns:
        sorted_tickers = sorted(sp500_data['instrument'].unique())
    elif 'ticker' in sp500_data.columns:
        sorted_tickers = sorted(sp500_data['ticker'].unique())
    else:
        # Fallback: lấy từ columns nếu là multi-index
        print("   ! Cảnh báo: Không tìm thấy cột 'instrument' hoặc 'ticker'")
        print("   Đang thử đọc từ sp500_history...")
        
except Exception as e:
    print(f"   ✗ Lỗi khi đọc sp500_data: {e}")
    print("   Đang thử đọc tickers từ sp500_history...")
    sp500_data = None

# Đọc sp500_history (có thể có MultiIndex columns)
try:
    # Thử đọc với MultiIndex header
    history_df = pd.read_csv(SP500_HISTORY_PATH, header=[0, 1], index_col=0, low_memory=False)
    history_df.index = pd.to_datetime(history_df.index)
    
    # Nếu chưa có sorted_tickers, lấy từ columns level 0
    if 'sorted_tickers' not in locals() or sorted_tickers is None or len(sorted_tickers) == 0:
        sorted_tickers = sorted(history_df.columns.get_level_values(0).unique())
    
    print(f"   ✓ Đã đọc sp500_history: {history_df.shape}")
    print(f"   Số lượng tickers: {len(sorted_tickers)}")
    print(f"   Thời gian: {history_df.index.min()} đến {history_df.index.max()}")
    print(f"   10 tickers đầu tiên: {sorted_tickers[:10]}")
    
except Exception as e:
    print(f"   ✗ Lỗi khi đọc sp500_history với MultiIndex: {e}")
    print("   Đang thử đọc với single header...")
    
    try:
        # Thử đọc với single header
        history_df = pd.read_csv(SP500_HISTORY_PATH, index_col=0, low_memory=False)
        history_df.index = pd.to_datetime(history_df.index)
        
        # Giả sử format là long-form với cột 'instrument' hoặc 'ticker'
        if 'instrument' in history_df.columns:
            sorted_tickers = sorted(history_df['instrument'].unique())
        elif 'ticker' in history_df.columns:
            sorted_tickers = sorted(history_df['ticker'].unique())
        
        print(f"   ✓ Đã đọc sp500_history (long format): {history_df.shape}")
        print(f"   Số lượng tickers: {len(sorted_tickers)}")
        
    except Exception as e2:
        print(f"   ✗ Lỗi nghiêm trọng: Không thể đọc dữ liệu: {e2}")
        raise

num_stocks = len(sorted_tickers)
num_dates = len(history_df.index.unique())

print(f"\n   📊 Tổng quan:")
print(f"   - Số cổ phiếu: {num_stocks}")
print(f"   - Số ngày giao dịch: {num_dates}")
print(f"   - Tổng số điểm dữ liệu: {num_stocks * num_dates:,}")

# ============================================================================
# BƯỚC 2: TASK 1 - TẠO RISK_FEATURES
# ============================================================================

print("\n[2/4] Đang tạo risk_features (Volatility, Momentum, Turnover)...")

# Khởi tạo mảng kết quả: (Time, Num_Stocks, 3)
risk_features = np.zeros((num_dates, num_stocks, 3))

for i, ticker in enumerate(sorted_tickers):
    try:
        # Lấy dữ liệu Close và Volume cho ticker này
        if isinstance(history_df.columns, pd.MultiIndex):
            # MultiIndex format: (ticker, 'Close'), (ticker, 'Volume')
            close = history_df[(ticker, 'Close')].values
            volume = history_df[(ticker, 'Volume')].values
            high = history_df[(ticker, 'High')].values
            low = history_df[(ticker, 'Low')].values
        else:
            # Long format: filter by ticker
            ticker_data = history_df[history_df['instrument'] == ticker].sort_index()
            close = ticker_data['close'].values
            volume = ticker_data['volume'].values
            high = ticker_data['high'].values
            low = ticker_data['low'].values
        
        # Tính returns
        returns = pd.Series(close).pct_change()
        
        # Feature 1: Volatility (std of returns over 20 days)
        volatility = returns.rolling(window=ROLLING_WINDOW).std().fillna(0).values
        
        # Feature 2: Momentum (cumulative return over 20 days)
        momentum = pd.Series(close).pct_change(periods=ROLLING_WINDOW).fillna(0).values
        
        # Feature 3: Turnover (Volume / MA20_Volume)
        ma_volume = pd.Series(volume).rolling(window=ROLLING_WINDOW).mean()
        turnover = (volume / ma_volume).fillna(0).replace([np.inf, -np.inf], 0).values
        
        # Gán vào mảng kết quả
        risk_features[:, i, 0] = volatility
        risk_features[:, i, 1] = momentum
        risk_features[:, i, 2] = turnover
        
        if (i + 1) % 50 == 0:
            print(f"   Đã xử lý {i + 1}/{num_stocks} tickers...")
            
    except Exception as e:
        print(f"   ! Cảnh báo: Lỗi khi xử lý {ticker}: {e}")
        # Giữ nguyên giá trị 0 cho ticker này
        continue

print(f"   ✓ Hoàn thành risk_features")
print(f"   Shape: {risk_features.shape}")
print(f"   Stats: mean={risk_features.mean():.6f}, std={risk_features.std():.6f}")
print(f"   NaN count: {np.isnan(risk_features).sum()}")

# ============================================================================
# BƯỚC 3: TASK 2 - TẠO SECTOR_ADJ_MATRIX
# ============================================================================

print("\n[3/4] Đang tạo sector_adj_matrix (Same-Sector Adjacency)...")

# Khởi tạo ma trận kề: (Num_Stocks, Num_Stocks)
sector_adj_matrix = np.zeros((num_stocks, num_stocks))

# Tạo mapping từ ticker sang sector
ticker_to_sector = {}

if sp500_data is not None:
    # Tìm cột sector
    sector_col = None
    for col in ['sector', 'Sector', 'GICS Sector', 'gics_sector', 'industry', 'Industry']:
        if col in sp500_data.columns:
            sector_col = col
            break
    
    if sector_col:
        print(f"   Sử dụng cột: {sector_col}")
        
        # Tạo mapping
        if 'instrument' in sp500_data.columns:
            for _, row in sp500_data.iterrows():
                ticker_to_sector[row['instrument']] = row[sector_col]
        elif 'ticker' in sp500_data.columns:
            for _, row in sp500_data.iterrows():
                ticker_to_sector[row['ticker']] = row[sector_col]
        
        print(f"   Đã mapping {len(ticker_to_sector)} tickers đến sectors")
        print(f"   Số sectors unique: {len(set(ticker_to_sector.values()))}")
    else:
        print("   ! Cảnh báo: Không tìm thấy cột sector")
        print("   Sẽ tạo ma trận với chỉ self-loops")
else:
    print("   ! Không có sp500_data, tạo ma trận với chỉ self-loops")

# Tạo ma trận kề
for i, ticker_i in enumerate(sorted_tickers):
    for j, ticker_j in enumerate(sorted_tickers):
        if i == j:
            # Self-loop
            sector_adj_matrix[i, j] = 1
        else:
            # Kiểm tra cùng sector
            sector_i = ticker_to_sector.get(ticker_i, None)
            sector_j = ticker_to_sector.get(ticker_j, None)
            
            if sector_i and sector_j and sector_i == sector_j:
                sector_adj_matrix[i, j] = 1
            else:
                sector_adj_matrix[i, j] = 0

print(f"   ✓ Hoàn thành sector_adj_matrix")
print(f"   Shape: {sector_adj_matrix.shape}")
print(f"   Số edges (không kể self-loops): {(sector_adj_matrix.sum() - num_stocks):.0f}")
print(f"   Density: {sector_adj_matrix.sum() / (num_stocks * num_stocks):.4f}")

# ============================================================================
# BƯỚC 4: TASK 3 - TẠO MONEY_FLOW_MATRIX
# ============================================================================

print("\n[4/4] Đang tạo money_flow_matrix (Money Flow Correlation)...")

# Tạo DataFrame để tính Money Flow cho tất cả tickers
money_flow_data = pd.DataFrame(index=history_df.index)

for i, ticker in enumerate(sorted_tickers):
    try:
        # Lấy dữ liệu OHLCV cho ticker
        if isinstance(history_df.columns, pd.MultiIndex):
            close = history_df[(ticker, 'Close')]
            high = history_df[(ticker, 'High')]
            low = history_df[(ticker, 'Low')]
            volume = history_df[(ticker, 'Volume')]
        else:
            ticker_data = history_df[history_df['instrument'] == ticker].sort_index()
            close = ticker_data['close']
            high = ticker_data['high']
            low = ticker_data['low']
            volume = ticker_data['volume']
        
        # Tính Money Flow Multiplier (MFM)
        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        denominator = high - low
        denominator = denominator.replace(0, np.nan)  # Tránh chia cho 0
        
        mfm = ((close - low) - (high - close)) / denominator
        mfm = mfm.fillna(0)  # Nếu High = Low, set MFM = 0
        
        # Tính Money Flow Volume
        flow = mfm * volume
        
        # Thêm vào DataFrame
        money_flow_data[ticker] = flow.values
        
        if (i + 1) % 50 == 0:
            print(f"   Đã tính Money Flow cho {i + 1}/{num_stocks} tickers...")
            
    except Exception as e:
        print(f"   ! Cảnh báo: Lỗi khi xử lý Money Flow cho {ticker}: {e}")
        money_flow_data[ticker] = 0
        continue

# Tính ma trận tương quan trên CORRELATION_WINDOW ngày gần nhất
print(f"   Đang tính correlation matrix trên {CORRELATION_WINDOW} ngày gần nhất...")

# Lấy dữ liệu N ngày gần nhất
recent_flow = money_flow_data.tail(CORRELATION_WINDOW)

# Tính correlation
money_flow_matrix = recent_flow.corr().values

# Xử lý NaN (nếu có ticker không có dữ liệu)
money_flow_matrix = np.nan_to_num(money_flow_matrix, nan=0.0)

# Đảm bảo diagonal = 1
np.fill_diagonal(money_flow_matrix, 1.0)

print(f"   ✓ Hoàn thành money_flow_matrix")
print(f"   Shape: {money_flow_matrix.shape}")
print(f"   Correlation range: [{money_flow_matrix.min():.4f}, {money_flow_matrix.max():.4f}]")
print(f"   Mean correlation: {money_flow_matrix.mean():.4f}")

# ============================================================================
# BƯỚC 5: LƯU KẾT QUẢ
# ============================================================================

print("\n[5/5] Đang lưu kết quả...")

# Lưu sorted_tickers
with open(f"{OUTPUT_DIR}/sorted_tickers.pkl", 'wb') as f:
    pickle.dump(sorted_tickers, f)
print(f"   ✓ Đã lưu sorted_tickers.pkl ({len(sorted_tickers)} tickers)")

# Lưu risk_features
np.save(f"{OUTPUT_DIR}/risk_features.npy", risk_features)
with open(f"{OUTPUT_DIR}/risk_features.pkl", 'wb') as f:
    pickle.dump(risk_features, f)
print(f"   ✓ Đã lưu risk_features.npy và .pkl")

# Lưu sector_adj_matrix
np.save(f"{OUTPUT_DIR}/sector_adj_matrix.npy", sector_adj_matrix)
with open(f"{OUTPUT_DIR}/sector_adj_matrix.pkl", 'wb') as f:
    pickle.dump(sector_adj_matrix, f)
print(f"   ✓ Đã lưu sector_adj_matrix.npy và .pkl")

# Lưu money_flow_matrix
np.save(f"{OUTPUT_DIR}/money_flow_matrix.npy", money_flow_matrix)
with open(f"{OUTPUT_DIR}/money_flow_matrix.pkl", 'wb') as f:
    pickle.dump(money_flow_matrix, f)
print(f"   ✓ Đã lưu money_flow_matrix.npy và .pkl")

# ============================================================================
# BƯỚC 6: KIỂM TRA KẾT QUẢ
# ============================================================================

print("\n" + "="*80)
print("KIỂM TRA KẾT QUẢ CUỐI CÙNG")
print("="*80)

print(f"\n📁 Thư mục output: {OUTPUT_DIR}/")
print(f"\n📊 Shape của các biến:")
print(f"   • sorted_tickers:      ({len(sorted_tickers)},)")
print(f"   • risk_features:       {risk_features.shape}")
print(f"   • sector_adj_matrix:   {sector_adj_matrix.shape}")
print(f"   • money_flow_matrix:   {money_flow_matrix.shape}")

print(f"\n✅ Kiểm tra alignment:")
expected_shape = (num_stocks, num_stocks)
assert sector_adj_matrix.shape == expected_shape, "sector_adj_matrix shape mismatch!"
assert money_flow_matrix.shape == expected_shape, "money_flow_matrix shape mismatch!"
assert risk_features.shape[1] == num_stocks, "risk_features stocks dimension mismatch!"
print(f"   ✓ Tất cả các ma trận đều align với sorted_tickers")

print(f"\n📈 Thống kê:")
print(f"\n   risk_features:")
print(f"      - Feature 0 (Volatility): mean={risk_features[:,:,0].mean():.6f}, std={risk_features[:,:,0].std():.6f}")
print(f"      - Feature 1 (Momentum):   mean={risk_features[:,:,1].mean():.6f}, std={risk_features[:,:,1].std():.6f}")
print(f"      - Feature 2 (Turnover):   mean={risk_features[:,:,2].mean():.6f}, std={risk_features[:,:,2].std():.6f}")

print(f"\n   sector_adj_matrix:")
num_edges = int(sector_adj_matrix.sum() - num_stocks)
print(f"      - Số edges (không kể self-loops): {num_edges:,}")
print(f"      - Trung bình số kết nối/node: {num_edges / num_stocks:.1f}")
print(f"      - Density: {sector_adj_matrix.sum() / (num_stocks * num_stocks):.4f}")

print(f"\n   money_flow_matrix:")
print(f"      - Correlation range: [{money_flow_matrix.min():.4f}, {money_flow_matrix.max():.4f}]")
print(f"      - Mean correlation (off-diagonal): {(money_flow_matrix.sum() - num_stocks) / (num_stocks * num_stocks - num_stocks):.4f}")

print("\n" + "="*80)
print("✅ HOÀN THÀNH! Tất cả các file đã được lưu trong", OUTPUT_DIR)
print("="*80)

print(f"\n💡 Sử dụng trong code:")
print("""
import numpy as np
import pickle

# Load dữ liệu
with open('data/graph_data/sorted_tickers.pkl', 'rb') as f:
    sorted_tickers = pickle.load(f)
    
risk_features = np.load('data/graph_data/risk_features.npy')
sector_adj_matrix = np.load('data/graph_data/sector_adj_matrix.npy')
money_flow_matrix = np.load('data/graph_data/money_flow_matrix.npy')
""")
