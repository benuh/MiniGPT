# 🌐 Remote Models Setup Guide - MiniGPT

Access powerful AI models from top companies **without downloading anything locally!**

## 🎯 Quick Start (Free Options)

### **Option 1: HuggingFace (100% Free)**
```bash
# No setup needed! Works immediately
python -m minigpt.chat --remote hf:gpt2
python -m minigpt.chat --remote hf:distilgpt2
```

### **Option 2: Local Streaming (100% Free)**
```bash
# Downloads to memory only, no disk storage
python -m minigpt.chat --remote local:gpt2
python -m minigpt.chat --remote local:distilgpt2
```

## 🚀 **Try It Right Now**

```bash
cd backend

# List all available remote models
python -m minigpt.chat --list-remote

# Start chatting with GPT-2 (free, instant)
python -m minigpt.chat --remote hf:gpt2
```

## 🛠 **Available Model Categories**

### 🟢 **FREE MODELS** (No API Key Needed)

| Model | Command | Provider | Quality | Speed |
|-------|---------|----------|---------|-------|
| GPT-2 | `--remote hf:gpt2` | HuggingFace | Good | Fast |
| DistilGPT-2 | `--remote hf:distilgpt2` | HuggingFace | Good | Very Fast |
| GPT-Neo 125M | `--remote hf:gpt-neo-125m` | HuggingFace | Good | Fast |
| GPT-Neo 1.3B | `--remote hf:gpt-neo-1.3b` | HuggingFace | Excellent | Medium |
| Falcon 7B | `--remote hf:falcon-7b` | HuggingFace | Excellent | Medium |
| Local GPT-2 | `--remote local:gpt2` | Local Memory | Good | Fast |

### 🟡 **PREMIUM MODELS** (API Key Required)

| Model | Command | Cost | Quality |
|-------|---------|------|---------|
| GPT-3.5 Turbo | `--remote openai:gpt-3.5-turbo` | $0.002/1K tokens | Excellent |
| GPT-4 | `--remote openai:gpt-4` | $0.03/1K tokens | Outstanding |
| Claude 3 Haiku | `--remote claude:claude-3-haiku` | $0.00025/1K tokens | Excellent |
| Claude 3 Sonnet | `--remote claude:claude-3-sonnet` | $0.003/1K tokens | Outstanding |
| Claude 3 Opus | `--remote claude:claude-3-opus` | $0.015/1K tokens | World-class |

## 🔧 **Setup Instructions**

### **Free Models (Ready Now)**
```bash
# Works immediately, no setup needed
python -m minigpt.chat --remote hf:gpt2
```

### **Premium Models Setup**

#### **1. OpenAI (ChatGPT)**
```bash
# Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-api-key-here"

# Install client (if not already installed)
pip install openai

# Start chatting
python -m minigpt.chat --remote openai:gpt-3.5-turbo
```

#### **2. Anthropic Claude**
```bash
# Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-api-key-here"

# Install client
pip install anthropic

# Start chatting
python -m minigpt.chat --remote claude:claude-3-sonnet
```

#### **3. HuggingFace (Enhanced)**
```bash
# Optional: Get token from https://huggingface.co/settings/tokens
export HUGGINGFACE_API_TOKEN="your-token-here"

# Removes rate limits and gives access to more models
python -m minigpt.chat --remote hf:gpt-neo-1.3b
```

## 💬 **Usage Examples**

### **Interactive Chat**
```bash
# Free models
python -m minigpt.chat --remote hf:gpt2
python -m minigpt.chat --remote local:distilgpt2

# Premium models
python -m minigpt.chat --remote openai:gpt-3.5-turbo
python -m minigpt.chat --remote claude:claude-3-haiku
```

### **Single Prompts**
```bash
python -m minigpt.chat --remote hf:gpt2 --prompt "Explain AI in simple terms"
python -m minigpt.chat --remote openai:gpt-4 --prompt "Write a poem about coding"
```

### **Web Interface Integration**
The web interface automatically supports remote models:

```bash
# Start API with remote model support
cd backend && python -m minigpt.api

# API endpoints now support:
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "model": "hf:gpt2"}'
```

## 🎮 **Chat Interface Features**

### **Available Commands in Chat**
```
📖 Available commands:
  help        - Show this help
  /models     - List available models
  /info       - Show current model info
  /switch <model> - Switch to different model
  quit/exit/q - Exit chat
```

### **Model Switching Example**
```
👤 You: /switch openai:gpt-4
✅ Switched to openai:gpt-4

👤 You: Hello!
🤖 openai:gpt-4: Hello! I'm GPT-4, a large language model created by OpenAI. How can I assist you today?

👤 You: /switch hf:gpt2
✅ Switched to hf:gpt2

👤 You: Hello!
🤖 hf:gpt2: Hello! How are you doing today?
```

## 📊 **Performance Comparison**

| Model Type | Response Time | Quality | Cost | Storage |
|------------|---------------|---------|------|---------|
| Local Trained | ~2 seconds | Variable | Free | ~500MB |
| HuggingFace API | ~3-5 seconds | Good-Excellent | Free | 0MB |
| Local Memory | ~1-3 seconds | Good | Free | 0MB (temp) |
| OpenAI API | ~1-2 seconds | Excellent | $0.002+ | 0MB |
| Claude API | ~2-3 seconds | Outstanding | $0.0003+ | 0MB |

## 🚨 **Troubleshooting**

### **"Model loading" errors (HuggingFace)**
```bash
# Wait 20 seconds and try again - models take time to load
python -m minigpt.chat --remote hf:gpt2
```

### **"API key not found"**
```bash
# Make sure environment variable is set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Set it in current session
export OPENAI_API_KEY="your-key"
```

### **"Rate limit exceeded"**
```bash
# For HuggingFace - get a free token
export HUGGINGFACE_API_TOKEN="your-token"

# Or try local models
python -m minigpt.chat --remote local:gpt2
```

### **"Import errors"**
```bash
# Install missing packages
pip install transformers torch  # For local models
pip install openai              # For OpenAI
pip install anthropic           # For Claude
```

## 📋 **Quick Reference Commands**

```bash
# List all remote models
python -m minigpt.chat --list-remote

# Chat with free models
python -m minigpt.chat --remote hf:gpt2
python -m minigpt.chat --remote hf:distilgpt2
python -m minigpt.chat --remote local:gpt2

# Chat with premium models (API key required)
python -m minigpt.chat --remote openai:gpt-3.5-turbo
python -m minigpt.chat --remote claude:claude-3-haiku

# Setup help
python -m minigpt.remote --setup

# Single prompt
python -m minigpt.chat --remote hf:gpt2 --prompt "Your question here"
```

## 💰 **Cost Considerations**

### **Free Forever**
- HuggingFace API models
- Local memory models
- Your own trained models

### **Pay Per Use**
- OpenAI: ~$0.002 per 1000 tokens (~750 words)
- Claude: ~$0.0003 per 1000 tokens (~750 words)
- Typical chat costs: $0.01-0.05 per conversation

### **Cost Tips**
1. **Start with free models** to test
2. **Use shorter prompts** to reduce costs
3. **Set up billing alerts** in API dashboards
4. **Mix free and paid** models based on needs

## 🎯 **Best Practices**

### **Model Selection**
- **Quick testing**: `hf:distilgpt2`
- **Good quality**: `hf:gpt2` or `local:gpt2`
- **High quality**: `hf:gpt-neo-1.3b` or `openai:gpt-3.5-turbo`
- **Best quality**: `openai:gpt-4` or `claude:claude-3-opus`

### **Development Workflow**
1. **Prototype** with free HuggingFace models
2. **Test** with local memory models
3. **Polish** with premium models
4. **Deploy** with best model for your budget

### **API Key Security**
- **Never commit** API keys to git
- **Use environment variables** only
- **Rotate keys** regularly
- **Set spending limits** in API dashboards

## 🚀 **Next Steps**

### **Start Free**
```bash
python -m minigpt.chat --remote hf:gpt2
```

### **Compare Models**
```bash
# Try different models and compare responses
python -m minigpt.chat --remote hf:gpt2 --prompt "Tell me a joke"
python -m minigpt.chat --remote hf:gpt-neo-1.3b --prompt "Tell me a joke"
```

### **Integrate with Your App**
```python
from minigpt.remote import RemoteModelManager

manager = RemoteModelManager()
response = manager.generate_text("hf:gpt2", "Hello world!")
print(response)
```

---

**🎉 You now have access to world-class AI models without any downloads or training time!**

**Ready to start?**
```bash
cd backend
python -m minigpt.chat --list-remote
python -m minigpt.chat --remote hf:gpt2
```