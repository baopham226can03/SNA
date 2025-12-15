# HSGNN + LLM Dynamic Graph Enhancement

Nâng cấp HSGNN với đồ thị động xây dựng bởi LLM.

## 📁 Cấu trúc Files

### **Hệ thống GỐC (Original HSGNN):**
- `model.py` - HSGNN model gốc từ paper
- `train.py` - Training script gốc
- `dataset.py` - Dataset loader
- `inference.py` - Backtesting

### **Hệ thống NÂNG CẤP (LLM Enhanced):**
- `llm_graph_builder.py` - **Module xây đồ thị động bằng LLM**
- `model_llm_dynamic_graph.py` - **HSGNN với đồ thị động**
- `train_llm_enhanced.py` - **Training script cho version nâng cấp**

---

## 🚀 Quick Start

### **1. Chạy hệ thống GỐC (baseline):**
```bash
python train.py --epochs 20 --batch_size 8 --output_dir outputs/baseline
```

### **2. Chạy hệ thống NÂNG CẤP (không dùng LLM API):**
```bash
# Dùng rule-based dynamic graphs (miễn phí, nhanh)
python train_llm_enhanced.py --epochs 20 --batch_size 8 --use_llm False \
    --output_dir outputs/llm_enhanced_rulebased
```

### **3. Chạy với LLM API (tốt nhất, tốn phí):**
```bash
# Cài thư viện
pip install openai

# Set API key
export OPENAI_API_KEY="sk-..."

# Train
python train_llm_enhanced.py --epochs 20 --batch_size 8 --use_llm True \
    --llm_provider openai --output_dir outputs/llm_enhanced_gpt
```

---

## 🔧 Cách hoạt động

### **Original HSGNN:**
```
Fixed Sector Graph → GAT → Predictions
```

### **LLM Enhanced HSGNN:**
```
Market Context + Stock Info 
    ↓
LLM Reasoning ("Are AAPL and MSFT related?")
    ↓
Dynamic Graph (changes daily)
    ↓
GAT → Predictions
```

---

## 💡 Ưu điểm LLM Dynamic Graph:

1. **Adaptive:** Graph thay đổi theo thời gian, không cố định
2. **Context-aware:** Xét market regime (bull/bear/volatile)
3. **Captures non-obvious relationships:** VD: TSLA ↔ Lithium miners
4. **Explainable:** LLM có thể giải thích tại sao 2 stocks liên quan

---

## 📊 So sánh Performance (dự kiến):

| Model | Validation RankIC | Training Time | Cost |
|-------|-------------------|---------------|------|
| Original HSGNN | ~0.03 | 2h (CPU) | Free |
| + Rule-based Dynamic | ~0.04 | 2h | Free |
| + LLM (GPT-3.5) | ~0.05-0.06 | 2.5h | ~$2-3 |
| + LLM (GPT-4) | ~0.06-0.08 | 3h | ~$10-15 |

---

## 🔑 LLM Providers

### **Option 1: Local (Rule-based) - Miễn phí**
```bash
--use_llm False
```
Dùng sector relationships + market regime rules. Không cần API key.

### **Option 2: OpenAI GPT**
```bash
--use_llm True --llm_provider openai
# Cần: export OPENAI_API_KEY="sk-..."
```
- GPT-3.5-turbo: ~$0.002/request → ~$2 cho full training
- GPT-4: ~$0.03/request → ~$15 cho full training

### **Option 3: Anthropic Claude**
```bash
--use_llm True --llm_provider anthropic
# Cần: export ANTHROPIC_API_KEY="sk-..."
```
- Claude-3-haiku: ~$0.001/request → ~$1 cho full training

---

## 🎯 Hyperparameters

### **Quan trọng nhất:**
- `--use_llm`: True/False - Có dùng LLM không
- `--llm_provider`: 'local', 'openai', 'anthropic'
- `--top_k`: Số cạnh tối đa mỗi stock (default: 10)

### **Model architecture (giống gốc):**
- `--hidden_dim`: 64
- `--num_gat_layers`: 2
- `--num_heads`: 4

---

## 📈 Kết quả & Đánh giá

Sau khi training, so sánh 2 models:

```bash
# Baseline
tensorboard --logdir outputs/baseline/tensorboard

# LLM Enhanced
tensorboard --logdir outputs/llm_enhanced_*/tensorboard
```

Check metrics:
- **Rank IC** (cao hơn = tốt hơn)
- **Validation Loss** (thấp hơn = tốt hơn)
- **Test Rank IC** (generalization)

---

## 🔬 Advanced: Custom LLM Prompts

Chỉnh sửa prompts trong `llm_graph_builder.py`:

```python
# Line ~220
prompt = f"""
Your custom prompt here...
Consider: {custom_factors}
Rate 0-1: ...
"""
```

---

## 💾 Caching

LLM responses được cache tại `data/graph_cache/` để:
- Tránh query lại cùng 1 câu hỏi
- Giảm cost
- Tăng tốc độ training

Xóa cache nếu muốn rebuild:
```bash
rm -rf data/graph_cache/
```

---

## 📝 Notes

1. **First run:** Sẽ chậm vì phải build graphs mới
2. **Subsequent runs:** Nhanh hơn nhờ cache
3. **API cost:** Monitor usage trên dashboard của provider
4. **Rule-based fallback:** Nếu LLM fail, tự động dùng sector graphs

---

## 🐛 Troubleshooting

**LỖI: "API key not found"**
```bash
export OPENAI_API_KEY="your-key-here"
# Hoặc
export ANTHROPIC_API_KEY="your-key-here"
```

**LỖI: "Module 'openai' not found"**
```bash
pip install openai anthropic
```

**Quá chậm/đắt với LLM:**
```bash
# Dùng rule-based thay thế
--use_llm False
```

---

## 📚 Citation

Nếu dùng trong nghiên cứu:

```
Original HSGNN: "Modeling Hybrid Firm Relationships with 
Graph Neural Networks for Stock Price Movement Prediction"

LLM Enhancement: Your work! Add your citation here.
```
