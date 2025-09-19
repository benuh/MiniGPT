# MiniGPT AutoTest Guide 🚀

Complete automation system for MiniGPT training, testing, and deployment.

## Quick Start

### 🚀 **Option 1: Instant AI (No Training)**
```bash
cd backend
pip install -e .
python -m minigpt.chat --remote hf:gpt2
```

### 🎯 **Option 2: Complete Automation**
```bash
# 1. Run quick setup check first
python quick_setup_check.py

# 2. Preview what will be automated
python autoTest.py --dry-run

# 3. Run full automation
python autoTest.py
```

### 🎮 **Option 3: Use Pre-trained Models**
```bash
python autoTest.py --use-pretrained
```

## What AutoTest Does

AutoTest automates the **complete MiniGPT pipeline**:

### 🔧 1. System Check
- ✅ Python 3.8+ verification
- ✅ Required packages (torch, transformers, etc.)
- ✅ Directory structure validation
- ✅ Node.js/npm for frontend

### 📊 2. Data Preparation
- ✅ Downloads WikiText dataset
- ✅ Creates sample data for quick testing
- ✅ Validates data quality
- ✅ Prepares training/validation splits

### 🎯 3. Model Training
- ✅ Backs up existing checkpoints
- ✅ Trains model (3 epochs for testing)
- ✅ Real-time progress monitoring
- ✅ Validates training outputs

### 🔧 4. Backend Testing
- ✅ Model loading verification
- ✅ Chat interface testing
- ✅ API server functionality
- ✅ Model evaluation metrics

### 🌐 5. Frontend Testing
- ✅ npm dependency installation
- ✅ Build process validation
- ✅ Development server testing
- ✅ Frontend-backend integration

### 🔄 6. Integration Testing
- ✅ End-to-end workflow validation
- ✅ Performance benchmarking
- ✅ Comprehensive reporting

## Usage Options

### Run Everything (Recommended)
```bash
python autoTest.py
```

### Preview Mode (Dry Run)
```bash
python autoTest.py --dry-run
```

### Run Specific Steps
```bash
python autoTest.py --step system      # System check only
python autoTest.py --step data        # Data preparation only
python autoTest.py --step train       # Training only
python autoTest.py --step backend     # Backend testing only
python autoTest.py --step frontend    # Frontend testing only
```

### Custom Configuration
```bash
python autoTest.py --config medium    # Use medium model config
python autoTest.py --epochs 5         # Train for 5 epochs
```

## Expected Timeline

| Step | Time | Description |
|------|------|-------------|
| System Check | 30 seconds | Verify dependencies |
| Data Preparation | 2-3 minutes | Download and prepare data |
| Model Training | 5-10 minutes | Train small model (3 epochs) |
| Backend Testing | 3-5 minutes | Test all backend functions |
| Frontend Testing | 5-8 minutes | Build and test frontend |
| Integration | 1-2 minutes | Final validation |
| **Total** | **15-30 minutes** | Complete automation |

## Manual Steps (If You Want to Run Yourself)

After a successful AutoTest run, you can manually repeat any step:

### 1. Manual Data Preparation
```bash
cd backend
python -c "
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
with open('data/manual_data.txt', 'w') as f:
    for example in dataset['train'][:100]:
        f.write(example['text'] + '\n')
print('Data prepared manually')
"
```

### 2. Manual Training
```bash
cd backend
python -m minigpt.train --config configs/small.yaml --epochs 3
```

### 3. Manual Backend Testing
```bash
cd backend

# Test model loading
python -c "
from minigpt.utils import load_checkpoint
from minigpt.model import MiniGPT
checkpoint = load_checkpoint('checkpoints/best_model.pt', 'cpu')
model = MiniGPT(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
print('✅ Model loads successfully')
"

# Test chat
python -m minigpt.chat --model checkpoints/best_model.pt --prompt "Hello test"

# Test API (in separate terminal)
python -m minigpt.api &
curl http://localhost:8000/health

# Test evaluation
python scripts/run_evaluation.py
```

### 4. Manual Frontend Testing
```bash
cd frontend

# Install dependencies
npm install

# Test build
npm run build

# Test dev server (in separate terminal)
npm start
# Visit http://localhost:3000
```

### 5. Manual Integration Testing
```bash
# Start backend
cd backend && python -m minigpt.api &

# Start frontend
cd frontend && npm start &

# Test integration
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt", "max_length": 20}'
```

## File Structure After AutoTest

```
MiniGPT/
├── autoTest.py                    # Main automation script
├── quick_setup_check.py          # Setup verification
├── AUTOTEST_GUIDE.md             # This guide
├── autotest.log                  # Detailed logs
├── autotest_report_*.json        # Test reports
│
├── backend/
│   ├── checkpoints/
│   │   ├── backup_*/             # Checkpoint backups
│   │   ├── checkpoint_*.pt       # Training checkpoints
│   │   └── best_model.pt         # Best model
│   ├── configs/
│   │   └── autotest.yaml         # Test configuration
│   ├── data/
│   │   └── sample_data.txt       # Prepared sample data
│   └── training_progress.json    # Training metrics
│
└── frontend/
    ├── build/                    # Production build
    └── node_modules/             # Dependencies
```

## Understanding the Output

### Success Indicators
```
🔹 STEP: Running: System Check
✅ System Check completed

🔹 STEP: Running: Data Preparation
✅ Data Preparation completed

🔹 STEP: Running: Model Training
ℹ️  Training command: python -m minigpt.train --config configs/autotest.yaml
ℹ️  Epoch 1/3, Loss: 4.23
ℹ️  Epoch 2/3, Loss: 3.87
ℹ️  Epoch 3/3, Loss: 3.52
✅ Model Training completed

🔹 STEP: Running: Backend Testing
ℹ️  Running test: Model Loading
✅ ✓ Model Loading
ℹ️  Running test: Chat Interface
✅ ✓ Chat Interface
✅ Backend Testing completed

📊 Final Report
==================================================
Total Duration: 0:18:42
Steps Passed: 6/6
Success Rate: 100.0%

🎉 All tests passed! Your MiniGPT setup is ready.
```

### Failure Indicators
```
❌ System Check failed
⚠️  Some tests failed. Review the logs and fix issues.

Common fixes:
  • Install missing dependencies: pip install -r requirements.txt
  • Check configuration files in backend/configs/
  • Verify data preparation completed successfully
```

## Troubleshooting

### Common Issues

#### "No module named 'minigpt'"
```bash
cd backend
pip install -e .
```

#### "CUDA out of memory"
```bash
# Reduce batch size in config
python autoTest.py --config small  # Use smaller config
```

#### "Port already in use"
```bash
# Kill existing processes
pkill -f "minigpt.api"
pkill -f "npm start"
```

#### "Node.js not found"
```bash
# Install Node.js
# macOS: brew install node
# Ubuntu: sudo apt install nodejs npm
```

#### Training fails
```bash
# Check logs
tail -f backend/training_autotest.log

# Try with fewer epochs
python autoTest.py --epochs 1
```

### Getting Help

1. **Check logs**: `autotest.log` contains detailed information
2. **Review reports**: `autotest_report_*.json` has step-by-step results
3. **Run setup check**: `python quick_setup_check.py`
4. **Run dry run**: `python autoTest.py --dry-run`

## Advanced Usage

### Custom Configuration
Create your own config file in `backend/configs/`:

```yaml
# backend/configs/my_config.yaml
model:
  vocab_size: 50257
  n_layer: 6
  n_head: 6
  n_embd: 256
  block_size: 512
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 1e-4
  max_epochs: 10
  warmup_steps: 200

data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-103-raw-v1"
  max_length: 512
```

Then run:
```bash
python autoTest.py --config my_config
```

### Continuous Integration
Add to your CI/CD pipeline:

```yaml
# .github/workflows/autotest.yml
name: AutoTest
on: [push, pull_request]

jobs:
  autotest:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - uses: actions/setup-node@v2
      with:
        node-version: '16'
    - name: Run AutoTest
      run: |
        pip install -r backend/requirements.txt
        python autoTest.py --epochs 1
```

### Production Deployment
For production deployment after AutoTest passes:

```bash
# Build for production
cd frontend && npm run build

# Start production services
docker-compose -f docker-compose.prod.yml up -d
```

## Next Steps After Success

Once AutoTest completes successfully:

### 1. Start Using MiniGPT

#### 🌐 **Remote Models (Instant)**
```bash
# List available remote models
python -m minigpt.chat --list-remote

# Chat with free models
python -m minigpt.chat --remote hf:gpt2
python -m minigpt.chat --remote hf:gpt-neo-1.3b

# Premium models (API key required)
export OPENAI_API_KEY="your-key"
python -m minigpt.chat --remote openai:gpt-3.5-turbo
```

#### 🎯 **Local Models**
```bash
# All-in-one startup
./start-all.sh

# Visit web interface
open http://localhost:3000

# Command line chat
cd backend && python -m minigpt.chat
```

### 2. Experiment with Training
```bash
# Train larger model
python -m minigpt.train --config configs/medium.yaml

# Custom training
python -m minigpt.train --config configs/small.yaml --epochs 20
```

### 3. Deploy Your Model
```bash
# Start API server
cd backend && python -m minigpt.api

# Test API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_length": 100}'
```

### 4. Monitor and Improve
```bash
# Evaluate model performance
cd backend && python scripts/run_evaluation.py

# Compare different models
python -m minigpt.compare --models checkpoints/*.pt
```

## Conclusion

AutoTest provides a **complete automation solution** for MiniGPT that:

- ✅ **Validates** your environment setup
- ✅ **Prepares** training data automatically
- ✅ **Trains** models with monitoring
- ✅ **Tests** all components comprehensively
- ✅ **Reports** detailed results
- ✅ **Guides** you to next steps

After a successful run, you'll have a **fully functional MiniGPT system** ready for experimentation, development, or production use.

**Happy automating! 🚀**