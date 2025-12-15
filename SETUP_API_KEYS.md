# Setup API Keys cho LLM

## 🔑 Cách 1: Dùng file .env (Recommended)

### 1. Copy template:
```bash
cp .env.example .env
```

### 2. Mở `.env` và điền API keys:
```bash
# File: .env
OPENAI_API_KEY=sk-proj-abc123...
ANTHROPIC_API_KEY=sk-ant-xyz789...
```

### 3. File `.env` sẽ KHÔNG bị commit lên Git (đã có trong .gitignore)

---

## 🔑 Cách 2: Set environment variables

### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-proj-abc123..."
$env:ANTHROPIC_API_KEY="sk-ant-xyz789..."
```

### Linux/Mac:
```bash
export OPENAI_API_KEY="sk-proj-abc123..."
export ANTHROPIC_API_KEY="sk-ant-xyz789..."
```

---

## 🚀 Lấy API Keys

### OpenAI (GPT):
1. Đăng ký: https://platform.openai.com/signup
2. Tạo API key: https://platform.openai.com/api-keys
3. Copy key bắt đầu `sk-proj-...`

**Chi phí:**
- GPT-3.5-turbo: $0.002/1k tokens (~$2 cho full training)
- GPT-4: $0.03/1k tokens (~$15 cho full training)

### Anthropic (Claude):
1. Đăng ký: https://console.anthropic.com/
2. Tạo API key: https://console.anthropic.com/settings/keys
3. Copy key bắt đầu `sk-ant-...`

**Chi phí:**
- Claude-3-haiku: $0.001/1k tokens (~$1 cho full training)
- Claude-3-sonnet: $0.015/1k tokens (~$8 cho full training)

---

## ✅ Kiểm tra

```bash
# Cài dependencies
pip install python-dotenv openai anthropic

# Test
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('OpenAI:', os.getenv('OPENAI_API_KEY')[:20] if os.getenv('OPENAI_API_KEY') else 'Not set')"
```

---

## 🛡️ Bảo mật

- ✅ `.env` đã được thêm vào `.gitignore`
- ✅ Không bao giờ commit API keys vào Git
- ✅ Không share `.env` file
- ⚠️ Nếu lỡ commit key: Revoke ngay trên dashboard
- ⚠️ Tắt key khi không dùng để tránh lãng phí

---

## 🔄 Chuyển đổi giữa providers

```bash
# Không dùng LLM (miễn phí)
python train_llm_enhanced.py --use_llm False

# Dùng OpenAI
python train_llm_enhanced.py --use_llm True --llm_provider openai

# Dùng Anthropic  
python train_llm_enhanced.py --use_llm True --llm_provider anthropic
```
