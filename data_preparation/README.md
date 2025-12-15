# Data Preparation Pipeline

This directory contains scripts to prepare data for HSGNN stock prediction model.

## Pipeline Overview

```
Step 1: Download S&P 500 Data
    ↓ (data/sp500_history.csv)
Step 2: Prepare Qlib Format
    ↓ (data/qlib_format/sp500_data.csv)
Step 3: Generate Alpha158 Features
    ↓ (data/alpha158/*.csv)
Step 4: Build Graph Structures
    ↓ (data/graph_data/*.npy)
```

## Quick Start

### Run All Steps

```bash
cd data_preparation
python run_all_data_preparation.py
```

### Run Individual Steps

```bash
# Step 1: Download data from Yahoo Finance
python step1_download_sp500_data.py

# Step 2: Convert to Qlib format
python step2_prepare_qlib_format.py

# Step 3: Generate Alpha158 features (158 technical indicators)
python step3_generate_alpha158.py

# Step 4: Build graph structures (sector + money flow)
python step4_build_graphs.py
```

### Skip Steps

```bash
# Skip download if data already exists
python run_all_data_preparation.py --skip-download

# Start from specific step
python run_all_data_preparation.py --start-from 3

# Run only one step
python run_all_data_preparation.py --only 2
```

## Step Details

### Step 1: Download S&P 500 Data
- **Input**: S&P 500 ticker list from Wikipedia
- **Output**: `data/sp500_history.csv`
- **Description**: Downloads historical price data (open, high, low, close, volume) for all S&P 500 stocks from Yahoo Finance
- **Requirements**: `yfinance`, `pandas`
- **Duration**: ~10-20 minutes (depends on internet speed)

### Step 2: Prepare Qlib Format
- **Input**: `data/sp500_history.csv`
- **Output**: `data/qlib_format/sp500_data.csv`
- **Description**: Converts raw data to Qlib's expected format with proper date handling, removes invalid data
- **Requirements**: `pandas`
- **Duration**: ~1 minute

### Step 3: Generate Alpha158 Features
- **Input**: `data/qlib_format/sp500_data.csv`
- **Output**: 
  - `data/alpha158/alpha158_features.csv` (158 technical features)
  - `data/alpha158/alpha158_labels.csv` (target returns)
- **Description**: Computes 158 technical indicators (momentum, volatility, correlation features)
- **Requirements**: `pandas`, `numpy`
- **Duration**: ~5-10 minutes

### Step 4: Build Graph Structures
- **Input**: 
  - `data/alpha158/alpha158_features.csv`
  - S&P 500 sector information
- **Output**:
  - `data/graph_data/sector_adj_matrix.npy` (sector-based graph)
  - `data/graph_data/money_flow_adj_matrix.npy` (correlation-based graph)
  - `data/graph_data/stock_list.npy` (list of stock tickers)
- **Description**: Creates graph structures representing stock relationships
- **Requirements**: `pandas`, `numpy`, `yfinance`
- **Duration**: ~2-5 minutes

## Output Data Structure

After running all steps, you will have:

```
data/
├── sp500_history.csv           # Raw downloaded data
├── qlib_format/
│   └── sp500_data.csv          # Qlib-formatted data
├── alpha158/
│   ├── alpha158_features.csv   # 158 technical features
│   └── alpha158_labels.csv     # Target returns
└── graph_data/
    ├── sector_adj_matrix.npy   # Sector graph (N x N)
    ├── money_flow_adj_matrix.npy  # Money flow graph (N x N)
    └── stock_list.npy          # Stock ticker list
```

## Requirements

Install dependencies:
```bash
pip install pandas numpy yfinance tqdm
```

## Troubleshooting

### Download fails for some stocks
- **Issue**: Some tickers may have been delisted or renamed
- **Solution**: The script will continue with available data

### Out of memory during Alpha158 generation
- **Issue**: Processing 500+ stocks with 158 features requires significant RAM
- **Solution**: Close other applications or process in smaller batches

### Graph generation errors
- **Issue**: Missing sector information for some stocks
- **Solution**: Check `fetch_sectors.py` and ensure API access to sector data

## Next Steps

After data preparation is complete, you can proceed to training:

```bash
# Train baseline HSGNN model
python train.py

# Train LLM-enhanced model
python train_llm_enhanced.py --use_llm

# Evaluate and compare results
python evaluate_results.py --baseline outputs/baseline --llm outputs/llm_enhanced
```

## Notes

- **Data frequency**: Daily data is used by default
- **Time range**: Configurable in `step1_download_sp500_data.py` (default: 2020-01-01 to present)
- **Caching**: Graph structures are cached to avoid recomputation
- **Updates**: To update with latest data, re-run Step 1 and subsequent steps
