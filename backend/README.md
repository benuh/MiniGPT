# MiniGPT Backend

The core MiniGPT implementation with transformer model, training pipeline, and FastAPI server.

## Quick Start

### Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Verify installation:
   ```bash
   python -c "import minigpt; print('✅ MiniGPT installed successfully!')"
   ```

### Running the API Server

Start the FastAPI server for the frontend:
```bash
python -m minigpt.api
```

The API will be available at `http://localhost:8000`

### Training a Model

1. **Quick training** (5 epochs on sample data):
   ```bash
   python -m minigpt.train --config configs/small.yaml --epochs 5
   ```

2. **Interactive chat** with your trained model:
   ```bash
   python -m minigpt.chat
   ```

## Commands

- `python -m minigpt.train` - Start training
- `python -m minigpt.chat` - Interactive chat
- `python -m minigpt.api` - Start API server
- `python -m minigpt.evaluate` - Evaluate model

## Project Structure

```
backend/
├── src/minigpt/           # Main package
│   ├── model.py          # Transformer architecture
│   ├── tokenizer.py      # Text tokenization
│   ├── train.py          # Training pipeline
│   ├── chat.py           # Chat interface
│   ├── api.py            # FastAPI server
│   ├── config.py         # Configuration
│   └── utils.py          # Utilities
├── configs/              # Model configurations
├── data/                 # Training data
├── checkpoints/          # Model checkpoints
└── requirements.txt      # Dependencies
```

See the main project README for detailed documentation.