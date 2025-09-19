# 🚀 First Time Setup - MiniGPT

Welcome to MiniGPT! This guide will help you get started with training your own personalized language model.

## 🎯 What Makes This Special

**Every user trains their own model!** 🤖

- No pre-trained models in the repository
- Your model learns from your training run
- Each person gets a unique, personalized chatbot
- Models stay local to your machine

## ⚡ Quick Start Options

### 🚀 **Option 1: Instant AI (No Training)**
```bash
cd backend
pip install -e .
python -m minigpt.chat --remote hf:gpt2
```
**Start chatting immediately with free models from HuggingFace, OpenAI, or Claude!**

### 🎯 **Option 2: Complete Automation**
```bash
python autoTest.py
```
This command will:
1. ✅ Check your system requirements
2. 📥 Download training data
3. 🎯 Train your personal model (15-20 minutes)
4. 🧪 Test everything works
5. 🚀 Start your chatbot

### 🎮 **Option 3: Manual Training**
```bash
cd backend
pip install -e .
python -m minigpt.train --config configs/small.yaml
python -m minigpt.chat
```

## 📋 Prerequisites

### System Requirements
- **Python 3.8+**
- **4GB+ RAM** (8GB recommended)
- **2GB free disk space**
- **Node.js 16+** (for web interface)

### Quick System Check
```bash
python quick_setup_check.py
```

## 🛠 Manual Installation (If You Prefer)

### 1. Backend Setup
```bash
cd backend
pip install -e .
```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

### 3. Train Your First Model
```bash
cd backend
python -m minigpt.train --config configs/small.yaml
```

### 4. Chat with Your Model
```bash
python -m minigpt.chat --model checkpoints/best_model.pt
```

## 🎮 Usage After Setup

### 🌐 **Remote Models (Instant)**
```bash
# List all available remote models
python -m minigpt.chat --list-remote

# Chat with free models
python -m minigpt.chat --remote hf:gpt2
python -m minigpt.chat --remote hf:gpt-neo-1.3b

# Chat with premium models (API key required)
export OPENAI_API_KEY="your-key"
python -m minigpt.chat --remote openai:gpt-3.5-turbo
```

### 🎯 **Local Models**
```bash
# Start everything (web interface)
./start-all.sh
# Visit http://localhost:3000

# Command line chat
cd backend
python -m minigpt.chat

# API server
cd backend
python -m minigpt.api
# Visit http://localhost:8000/docs
```

## 📁 What Gets Created Locally

After your first training run:

```
MiniGPT/
├── backend/
│   ├── checkpoints/          # Your trained models 🤖
│   │   ├── best_model.pt     # Your best model
│   │   └── checkpoint_*.pt   # Training checkpoints
│   ├── data/                 # Downloaded training data
│   └── logs/                 # Training logs
├── autotest.log             # Setup logs
└── autotest_report_*.json   # Test results
```

**Note**: These files are git-ignored and stay on your machine only!

## 🔧 Model Configurations

Choose your model size based on your hardware:

### Small Model (Recommended for First Run)
```bash
python autoTest.py --config small
```
- **Training Time**: 10-15 minutes
- **Memory**: 2-4GB RAM
- **Quality**: Good for experimentation

### Medium Model (Better Quality)
```bash
python autoTest.py --config medium
```
- **Training Time**: 30-45 minutes
- **Memory**: 4-8GB RAM
- **Quality**: Noticeably better responses

### Production Model (Best Quality)
```bash
python autoTest.py --config production
```
- **Training Time**: 1-2 hours
- **Memory**: 8GB+ RAM
- **Quality**: High-quality responses

## 🚨 Troubleshooting

### "Command not found"
```bash
# Try python3 instead of python
python3 autoTest.py
```

### "Out of memory"
```bash
# Use smaller model
python autoTest.py --config small
```

### "Port already in use"
```bash
# Kill existing processes
pkill -f "python.*api"
pkill -f "npm start"
```

### Training seems stuck
```bash
# Check logs
tail -f autotest.log
```

## 🎯 What to Expect

### First Training Run
- **Duration**: 15-30 minutes (small model)
- **Data Download**: ~100MB (WikiText dataset)
- **Model Size**: ~160MB (small model)
- **Final Result**: Working chatbot ready to chat!

### Your Model's Personality
Your model will learn from the WikiText dataset but each training run produces unique results:
- Different conversation styles
- Varying response patterns
- Unique "personality" quirks
- Personal model that grows with more training

## 🎉 Success Indicators

You'll know everything worked when you see:

```
🎉 All tests passed! Your MiniGPT setup is ready.

You can now:
  • Start the application: ./start-all.sh
  • Chat with your model: cd backend && python -m minigpt.chat
  • Use web interface: http://localhost:3000
```

## 🔄 Re-training Your Model

Want to improve your model? Simply run training again:

```bash
# Continue training existing model
cd backend
python -m minigpt.train --config configs/small.yaml --resume checkpoints/best_model.pt

# Or start fresh
rm -rf checkpoints/*
python autoTest.py
```

## 💡 Tips for Best Results

1. **Start Small**: Use the small config for your first run
2. **Be Patient**: Training takes time but it's worth it!
3. **Monitor Progress**: Watch the logs to see loss decreasing
4. **Experiment**: Try different configs and see what works best
5. **Have Fun**: Each model is unique - explore what yours can do!

## 🆘 Getting Help

- **Logs**: Check `autotest.log` for detailed information
- **System Check**: Run `python quick_setup_check.py`
- **Preview**: Run `python autoTest.py --dry-run` to see what will happen
- **Step by Step**: Run individual steps with `python autoTest.py --step train`

---

**Ready to train your personal AI? Let's go!** 🚀

```bash
python autoTest.py
```