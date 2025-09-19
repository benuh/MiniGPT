# 🤖 Pre-trained Models Guide - MiniGPT

Skip the training time and jump straight to chatting with powerful pre-trained models from top AI companies!

## 🚀 Quick Start

### Option 1: Use AutoTest (Recommended)
```bash
# List available models
python autoTest.py --list-pretrained

# Use a pre-trained model instead of training
python autoTest.py --use-pretrained gpt2
```

### Option 2: Command Line
```bash
# Import a specific model
cd backend
python -m minigpt.pretrained --import gpt2

# Chat with imported model
python -m minigpt.chat
```

### Option 3: Python API
```python
import minigpt

# Import and use in one line
minigpt.quick_import_model("gpt2")
minigpt.quick_chat("backend/checkpoints/pretrained_gpt2.pt")
```

## 🤖 Available Pre-trained Models

| Model Key | Size | Description | Source | License |
|-----------|------|-------------|---------|---------|
| `gpt2` | 124M | OpenAI GPT-2 (small) | HuggingFace | MIT |
| `gpt2-medium` | 355M | OpenAI GPT-2 (medium) | HuggingFace | MIT |
| `gpt2-large` | 774M | OpenAI GPT-2 (large) | HuggingFace | MIT |
| `distilgpt2` | 82M | Distilled GPT-2 (fast) | HuggingFace | Apache 2.0 |
| `gpt-neo-125m` | 125M | EleutherAI GPT-Neo (small) | HuggingFace | MIT |
| `gpt-neo-1.3b` | 1.3B | EleutherAI GPT-Neo (medium) | HuggingFace | MIT |
| `gpt-neo-2.7b` | 2.7B | EleutherAI GPT-Neo (large) | HuggingFace | MIT |
| `gpt-j-6b` | 6B | EleutherAI GPT-J (very large) | HuggingFace | Apache 2.0 |
| `pythia-70m` | 70M | EleutherAI Pythia (tiny) | HuggingFace | Apache 2.0 |
| `pythia-160m` | 160M | EleutherAI Pythia (small) | HuggingFace | Apache 2.0 |

## 💡 Model Recommendations

### 🏃‍♂️ **For Quick Testing**
- **`distilgpt2`** (82M) - Fastest, good quality
- **`pythia-70m`** (70M) - Tiny, very fast

### ⚖️ **Best Balance**
- **`gpt2`** (124M) - Classic, reliable
- **`gpt-neo-125m`** (125M) - Modern architecture

### 🎯 **High Quality**
- **`gpt2-medium`** (355M) - Excellent quality
- **`gpt-neo-1.3b`** (1.3B) - Very high quality

### 🔥 **Maximum Performance** (Requires powerful hardware)
- **`gpt-neo-2.7b`** (2.7B) - Top-tier quality
- **`gpt-j-6b`** (6B) - State-of-the-art (16GB+ RAM recommended)

## 🛠 Usage Methods

### Method 1: AutoTest Integration
```bash
# Complete automation with pre-trained model
python autoTest.py --use-pretrained gpt2

# Preview what will happen
python autoTest.py --use-pretrained gpt2 --dry-run

# Import model only
python autoTest.py --step pretrained --use-pretrained gpt2
```

### Method 2: Direct CLI Commands
```bash
cd backend

# List available models
python -m minigpt.pretrained --list

# Import a model
python -m minigpt.pretrained --import gpt2

# Check imported models
python -m minigpt.pretrained --imported

# Get model info
python -m minigpt.pretrained --info gpt2

# Remove a model
python -m minigpt.pretrained --remove gpt2
```

### Method 3: Web Interface
```bash
# Start API server
cd backend && python -m minigpt.api

# Available endpoints:
# GET /pretrained/available - List available models
# GET /pretrained/imported - List imported models
# POST /pretrained/import/{model_key} - Import a model
# DELETE /pretrained/remove/{model_key} - Remove a model
```

### Method 4: Python Programming
```python
from minigpt.pretrained import PretrainedModelImporter
from minigpt import ChatBot

# Initialize importer
importer = PretrainedModelImporter()

# List available models
models = importer.list_available_models()
print(models.keys())

# Import a model
path = importer.import_model("gpt2")
print(f"Model imported to: {path}")

# Use the model
chatbot = ChatBot(path)
response = chatbot.generate_text("Hello world!")
print(response)
```

## 🔄 API Usage Examples

### List Available Models
```bash
curl http://localhost:8000/pretrained/available
```

```json
{
  "models": {
    "gpt2": {
      "source": "huggingface",
      "size": "124M",
      "description": "OpenAI GPT-2 (small)",
      "license": "MIT"
    }
  }
}
```

### Import a Model
```bash
curl -X POST http://localhost:8000/pretrained/import/gpt2
```

```json
{
  "message": "Started importing gpt2",
  "model_key": "gpt2"
}
```

### Chat with Imported Model
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "max_length": 50}'
```

## 📁 File Organization

### Where Models Are Stored
```
backend/
└── checkpoints/
    ├── pretrained_gpt2.pt          # Imported GPT-2
    ├── pretrained_distilgpt2.pt    # Imported DistilGPT-2
    └── pretrained_cache/           # Download cache
        └── huggingface/            # HuggingFace models cache
```

### Model File Naming
- **Format**: `pretrained_{model_key}.pt`
- **Examples**:
  - `pretrained_gpt2.pt`
  - `pretrained_distilgpt2.pt`
  - `pretrained_gpt-neo-125m.pt`

## ⚡ Performance Comparison

| Model | Size | Import Time | Memory Usage | Quality Score |
|-------|------|-------------|--------------|---------------|
| pythia-70m | 70M | 1 min | 1GB | 6/10 |
| distilgpt2 | 82M | 1 min | 1GB | 7/10 |
| gpt2 | 124M | 2 min | 2GB | 8/10 |
| gpt-neo-125m | 125M | 2 min | 2GB | 8/10 |
| pythia-160m | 160M | 2 min | 2GB | 7/10 |
| gpt2-medium | 355M | 4 min | 4GB | 9/10 |
| gpt2-large | 774M | 8 min | 8GB | 9/10 |
| gpt-neo-1.3b | 1.3B | 12 min | 12GB | 9.5/10 |
| gpt-neo-2.7b | 2.7B | 20 min | 16GB | 9.5/10 |
| gpt-j-6b | 6B | 45 min | 24GB | 10/10 |

## 🚨 System Requirements

### Minimum Requirements
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Models**: distilgpt2, pythia-70m, gpt2

### Recommended Requirements
- **RAM**: 8GB
- **Storage**: 5GB free space
- **Models**: gpt2-medium, gpt-neo-125m

### High-End Requirements
- **RAM**: 16GB+
- **Storage**: 10GB+ free space
- **Models**: gpt-neo-1.3b, gpt-neo-2.7b

### Enthusiast Requirements
- **RAM**: 32GB+
- **Storage**: 20GB+ free space
- **GPU**: Optional but recommended
- **Models**: gpt-j-6b

## 🔧 Troubleshooting

### "transformers not installed"
```bash
pip install transformers
# or
pip install -r backend/requirements.txt
```

### "Out of memory during import"
```bash
# Try a smaller model
python autoTest.py --use-pretrained distilgpt2

# Or free up memory
# Close other applications and try again
```

### "Model import failed"
```bash
# Check internet connection
ping huggingface.co

# Clear cache and retry
rm -rf backend/checkpoints/pretrained_cache
python -m minigpt.pretrained --import gpt2
```

### "Model not found after import"
```bash
# Check imported models
python -m minigpt.pretrained --imported

# Verify file exists
ls -la backend/checkpoints/pretrained_*.pt
```

### "Slow import speed"
```bash
# Import runs in background - be patient
# Check progress in logs
tail -f autotest.log
```

## 📊 Usage Workflows

### Workflow 1: Quick Chat
```bash
# 1. Import model (2 minutes)
python autoTest.py --use-pretrained gpt2 --step pretrained

# 2. Start chatting immediately
cd backend && python -m minigpt.chat
```

### Workflow 2: Full Testing
```bash
# Complete automation with testing
python autoTest.py --use-pretrained gpt2
```

### Workflow 3: Web Interface
```bash
# 1. Import model via API
curl -X POST http://localhost:8000/pretrained/import/gpt2

# 2. Start web interface
./start-all.sh

# 3. Chat at http://localhost:3000
```

### Workflow 4: Multiple Models
```bash
# Import several models for comparison
python -m minigpt.pretrained --import gpt2
python -m minigpt.pretrained --import distilgpt2
python -m minigpt.pretrained --import gpt-neo-125m

# Compare them
python -m minigpt.compare --models checkpoints/pretrained_*.pt
```

## 🎯 Best Practices

### 1. **Start Small**
- Begin with `distilgpt2` or `gpt2` for testing
- Upgrade to larger models once familiar

### 2. **Manage Storage**
- Remove unused models: `python -m minigpt.pretrained --remove model_key`
- Keep cache for faster re-imports

### 3. **Monitor Resources**
- Check memory usage during import
- Close unnecessary applications for large models

### 4. **Network Considerations**
- Models download from HuggingFace (requires internet)
- First import is slower (downloads model)
- Subsequent imports use cache (faster)

### 5. **Quality vs Speed**
- Use `distilgpt2` for fast iterations
- Use `gpt2-medium` for better quality
- Use `gpt-j-6b` for best quality (if hardware allows)

## 🚀 Advanced Usage

### Custom Model Conversion
```python
from minigpt.pretrained import PretrainedModelImporter

# Advanced usage with custom handling
importer = PretrainedModelImporter()

# Get detailed model info
info = importer.get_model_info("gpt2")
print(f"License: {info['license']}")
print(f"Size: {info['size']}")

# Import with progress tracking
path = importer.import_model("gpt2")
print(f"Imported to: {path}")
```

### API Integration
```python
import requests

# Start import via API
response = requests.post("http://localhost:8000/pretrained/import/gpt2")
print(response.json())

# Check import status
models = requests.get("http://localhost:8000/pretrained/imported")
print(models.json())

# Use imported model
chat_response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Hello AI!", "max_length": 100}
)
print(chat_response.json()["response"])
```

## 📋 Quick Reference

### Essential Commands
```bash
# List available models
python autoTest.py --list-pretrained

# Quick start with pre-trained model
python autoTest.py --use-pretrained gpt2

# Import specific model
python -m minigpt.pretrained --import gpt2

# Chat with any imported model
python -m minigpt.chat

# Start web interface
./start-all.sh
```

### Model Selection Guide
- **Learning/Testing**: `distilgpt2` or `gpt2`
- **General Use**: `gpt2` or `gpt2-medium`
- **High Quality**: `gpt-neo-1.3b` or `gpt2-large`
- **Best Quality**: `gpt-j-6b` (requires powerful hardware)

---

**🎉 Now you can skip training and jump straight to chatting with world-class AI models!**

**Ready to start?**
```bash
python autoTest.py --list-pretrained
python autoTest.py --use-pretrained gpt2
```