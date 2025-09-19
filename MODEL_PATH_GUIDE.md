# 📁 Model Path Handling - MiniGPT

## 🎯 Overview

MiniGPT automatically handles model paths so you can run commands from anywhere in the project. Your trained models are always stored in `backend/checkpoints/` but can be accessed consistently regardless of your current directory.

## 📂 Where Models Are Stored

```
MiniGPT/
└── backend/
    └── checkpoints/          # Your trained models live here
        ├── best_model.pt     # Best performing model
        ├── checkpoint_*.pt   # Training checkpoints
        └── latest_model.pt   # Most recent model
```

**Important**: These files are git-ignored, so each user trains their own models locally.

## 🔧 Automatic Path Detection

The system automatically finds your models whether you run commands from:

### From Project Root
```bash
# All of these work from MiniGPT/
python autoTest.py
python backend/src/minigpt/chat.py
python -m minigpt.api  # (if backend is installed)
```

### From Backend Directory
```bash
cd backend
python -m minigpt.chat     # Finds ../checkpoints/ automatically
python -m minigpt.api      # Finds ../checkpoints/ automatically
python -m minigpt.train    # Saves to ../checkpoints/ automatically
```

### From Any Subdirectory
The utilities automatically search for checkpoints in:
1. `./checkpoints/` (current directory)
2. `./backend/checkpoints/` (if running from project root)
3. Creates `./checkpoints/` if neither exists

## 🎮 Usage Examples

### Terminal Chat (No Model Path Needed)
```bash
# Auto-finds best_model.pt or latest checkpoint
cd backend
python -m minigpt.chat

# Or specify a specific model
python -m minigpt.chat --model checkpoints/my_model.pt
```

### Web Interface (Auto-loads Model)
```bash
# From project root
./start-all.sh
# Backend auto-loads best available model

# From backend directory
cd backend
python -m minigpt.api
# Visit http://localhost:8000
```

### API Usage
The API automatically loads the best available model on startup:

```bash
curl http://localhost:8000/model/list
# Shows all available models

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
# Uses auto-loaded model
```

## 🔍 Model Detection Priority

The system finds models in this order:

1. **Best Model**: `best_model.pt` (if exists)
2. **Latest Model**: Most recently modified `.pt` file
3. **No Model**: Shows helpful error message

## 🧪 Testing Path Handling

Run the test script to verify everything works:

```bash
python test_model_paths.py
```

Expected output:
```
🔍 Testing model path detection...
📁 Checkpoints directory: backend/checkpoints
   Exists: True
   Models found: 2
     - best_model.pt (164.4 MB)
     - checkpoint_1000.pt (164.4 MB)

🏆 Best model: backend/checkpoints/best_model.pt
🕐 Latest model: backend/checkpoints/best_model.pt
✅ Path handling test complete!
```

## 🚨 Troubleshooting

### "No trained models found"
```bash
# Train a model first
python autoTest.py
# or
cd backend && python -m minigpt.train --config configs/small.yaml
```

### "Checkpoint not found"
```bash
# Check what models you have
ls -la backend/checkpoints/

# Use the test script to debug
python test_model_paths.py
```

### "Permission denied"
```bash
# Make scripts executable
chmod +x autoTest.py
chmod +x run_autotest.sh
chmod +x test_model_paths.py
```

## 🎯 Developer Notes

### Path Utilities Used

```python
from minigpt.utils import (
    get_checkpoints_dir,    # Find checkpoints directory
    find_best_checkpoint,   # Find best_model.pt or latest
    find_latest_checkpoint, # Find most recent .pt file
    load_checkpoint,        # Load with automatic path resolution
    save_checkpoint         # Save to correct directory
)
```

### Adding New Model Loading Code

When adding new functionality that loads models:

```python
# ✅ Good - Uses utilities
from minigpt.utils import find_best_checkpoint, load_checkpoint

model_path = find_best_checkpoint()
if model_path:
    checkpoint = load_checkpoint(str(model_path), device)

# ❌ Avoid - Hard-coded paths
checkpoint = load_checkpoint("checkpoints/best_model.pt", device)
```

## 📋 Quick Reference

| Command | Working Directory | Model Path |
|---------|------------------|------------|
| `python autoTest.py` | Project root | Auto-detects |
| `cd backend && python -m minigpt.chat` | backend/ | Auto-detects |
| `python -m minigpt.api` | backend/ | Auto-detects |
| `./start-all.sh` | Project root | Auto-detects |

All commands automatically find and use your trained models from `backend/checkpoints/` regardless of where you run them! 🎉