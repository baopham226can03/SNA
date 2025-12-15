# HSGNN Stock Prediction

Hybrid Structure-aware GNN implementation for S&P 500 stock prediction, based on the paper "Modeling Hybrid Firm Relationships with Graph Neural Networks for Stock Price Movement Prediction".

## 📁 Project Structure

```
.
├── data/                          # Data directory
│   ├── alpha158/                  # Alpha158 features & labels
│   ├── graph_data/               # Graph matrices (sector, money flow, risk)
│   └── qlib_format/              # Qlib formatted data
│
├── outputs/                       # Training outputs
│   └── hsgnn_test/               # Model checkpoints & logs
│
├── dataset.py                    # Dataset loader with temporal windowing
├── model.py                      # HSGNN model (3 modules)
├── train.py                      # Training script
├── inference.py                  # Backtesting & portfolio evaluation
│
├── build_graphs.py               # Build graph matrices from data
├── fetch_sectors.py              # Fetch S&P 500 sector information
├── d_generate_alpha158.py        # Generate Alpha158 features
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preparation (if needed)
```bash
# Fetch sector data
python fetch_sectors.py

# Generate Alpha158 features
python d_generate_alpha158.py

# Build graph matrices
python build_graphs.py
```

### 3. Train Model
```bash
# Quick test (20 epochs, CPU)
python train.py --epochs 20 --batch_size 8 --window_size 150 --output_dir outputs/test

# Full training (100 epochs, recommended with GPU)
python train.py --epochs 100 --batch_size 16 --window_size 300 --output_dir outputs/full
```

### 4. Backtest & Evaluate
```bash
python inference.py --model_path outputs/test/best_model.pt
```

## 📊 Model Architecture

**3 Modules:**
1. **Implicit Graph Learning**: Attention-based graph construction from risk features
2. **Explicit GAT**: Graph Attention Networks on sector relationships  
3. **Hybrid Encoder**: Gated fusion of implicit + explicit representations

**Key Features:**
- Temporal sliding window (150-300 days)
- Rank IC (Spearman) metric
- Early stopping with patience
- TensorBoard logging

## 💻 GPU Training

**Enable GPU (if available):**
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Training will automatically use GPU if available.

## 📈 Expected Performance

- **Rank IC**: 0.03-0.06 on S&P 500 validation set
- **Training time**: 
  - CPU: ~6 min/epoch (100 epochs = 10h)
  - GPU (GTX 1650): ~1.5 min/epoch (100 epochs = 2.5h)
  - GPU (RTX 3080): ~20 sec/epoch (100 epochs = 35min)

## 📄 Paper Reference

"Modeling Hybrid Firm Relationships with Graph Neural Networks for Stock Price Movement Prediction"

Implementation follows paper methodology with public S&P 500 data.
