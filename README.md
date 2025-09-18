# MiniGPT

A minimalist GPT implementation built from scratch for learning and experimentation. This project demonstrates modern ML engineering practices.

## Prerequisites

**Python & Package Manager Setup:**

This project requires Python 3.8+ with pip. Depending on your system, you may need to use `python3` and `pip3` instead of `python` and `pip`.

```bash
# Check your Python installation
python --version   # or python3 --version
pip --version      # or pip3 --version

# If pip is not in PATH, add this to your shell profile (~/.zshrc, ~/.bashrc):
export PATH="$HOME/Library/Python/3.9/bin:$PATH"  # macOS
export PATH="$HOME/.local/bin:$PATH"              # Linux

# Then reload your shell:
source ~/.zshrc  # or source ~/.bashrc
```

**Note:** Throughout this README:

- Replace `python` with `python3` if needed
- Replace `pip` with `pip3` if needed
- If you see "command not found" errors, ensure the Python scripts are in your PATH

## Features

### Core Architecture

- **Transformer Model**: Clean GPT-style implementation with multi-head attention
- **Modular Design**: Separate components for attention, MLP, and transformer blocks
- **Configurable**: Easy to adjust model size and hyperparameters

### Training Pipeline

- **Professional Training Loop**: Complete with validation, checkpointing, and logging
- **Data Loading**: Efficient batching and preprocessing for text data
- **Experiment Tracking**: Integration with Weights & Biases
- **Resumable Training**: Save and resume from checkpoints

### Inference & Chat

- **Interactive Chat Interface**: Real-time conversation with your trained model
- **Flexible Generation**: Configurable temperature, top-k sampling
- **Command System**: Built-in commands for adjusting generation parameters

### 🔧 ML Engineering Best Practices

- **Configuration Management**: YAML-based configs for reproducible experiments
- **Device Agnostic**: Automatic GPU/MPS/CPU detection
- **Comprehensive Utilities**: Parameter counting, checkpointing, metrics
- **Clean Code**: Type hints, docstrings, and modular structure

## Quick Start

### Installation

```bash
git clone https://github.com/benuh/MiniGPT.git
cd MiniGPT

# Install dependencies (use pip3 if pip doesn't work)
pip install -e .
# or
python3 -m pip install -e .
```

**Troubleshooting Installation:**

- If `pip install -e .` fails, try `python3 -m pip install -e .`
- If you get PATH warnings, add the Python bin directory to your PATH (see Prerequisites above)
- On macOS, you may need to install Xcode Command Line Tools: `xcode-select --install`

### Training a Model

```bash
# Train with default small config (use python3 if python doesn't work)
python -m minigpt.train --config configs/small.yaml
# or
python3 -m minigpt.train --config configs/small.yaml

# Resume training from checkpoint
python -m minigpt.train --config configs/small.yaml --resume checkpoints/checkpoint_1000.pt
```

### Chat with Your Model

```bash
# Interactive chat (use python3 if python doesn't work)
python -m minigpt.chat --model checkpoints/best_model.pt
# or
python3 -m minigpt.chat --model checkpoints/best_model.pt

# Single prompt
python -m minigpt.chat --model checkpoints/best_model.pt --prompt "Hello, how are you?"
```

### Quick Test After Installation

```bash
# Verify installation works
python -c "import minigpt; print('MiniGPT installed successfully!')"
# or
python3 -c "import minigpt; print('MiniGPT installed successfully!')"
```

## Project Structure

```
MiniGPT/
├── src/minigpt/           # Main package
│   ├── model.py          # Transformer architecture
│   ├── tokenizer.py      # Text tokenization
│   ├── train.py          # Training pipeline
│   ├── chat.py           # Inference & chat interface
│   ├── config.py         # Configuration management
│   └── utils.py          # Utilities and helpers
├── configs/              # Model configurations
│   └── small.yaml        # Small model config
├── scripts/              # Training and utility scripts
├── tests/                # Unit tests
├── data/                 # Dataset storage
└── checkpoints/          # Model checkpoints
```

## Model Configurations

### Small Model (Default)

- **Parameters**: ~87K
- **Layers**: 4
- **Hidden Size**: 128
- **Attention Heads**: 4
- **Context Length**: 256

Perfect for experimentation and learning on a laptop.

## Configuration

Models are configured via YAML files:

```yaml
model:
  vocab_size: 50257
  n_layer: 4
  n_head: 4
  n_embd: 128
  block_size: 256
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 3e-4
  max_epochs: 10
  weight_decay: 0.01
  gradient_clip: 1.0

data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-2-raw-v1"
  max_length: 256

logging:
  log_interval: 100
  eval_interval: 500
  save_interval: 1000
```

## Chat Commands

The interactive chat interface supports these commands:

- `help` - Show available commands
- `/temp 0.8` - Set sampling temperature
- `/topk 50` - Set top-k value for sampling
- `/len 100` - Set maximum generation length
- `/reset` - Reset conversation history
- `quit` - Exit chat

## Development Roadmap

- [x] Core transformer implementation
- [x] Training pipeline with checkpointing
- [x] Interactive chat interface
- [x] Configuration management
- [x] Model evaluation metrics (perplexity, BLEU)
- [x] Advanced training features (LR scheduling, early stopping)
- [x] Model optimization and quantization
- [x] Model comparison and A/B testing
- [x] REST API server
- [x] Docker containerization
- [x] Continuous improvement automation
- [x] Comprehensive documentation

## Technical Details

### Architecture

- **Attention**: Scaled dot-product with causal masking
- **Position Encoding**: Learned position embeddings
- **Activation**: GELU activation functions
- **Normalization**: Layer normalization (pre-norm)

### Training

- **Optimizer**: AdamW with weight decay
- **Scheduler**: Linear warmup (configurable)
- **Loss**: Cross-entropy on next-token prediction
- **Validation**: Automatic best model selection

### Generation

- **Sampling**: Temperature and top-k sampling
- **Context**: Automatic context window management
- **Stopping**: Configurable max tokens

## Advanced Usage

### Continuous Improvement System

```bash
# Start automated training (runs for 8 hours, then restarts)
chmod +x scripts/start_continuous.sh
./scripts/start_continuous.sh
```

### API Server

```bash
# Start REST API server
python scripts/start_server.py
# or
python3 scripts/start_server.py

# Visit http://localhost:8000/docs for API documentation
```

### Docker Deployment

```bash
# Build and run with Docker
./scripts/docker_build.sh build
./scripts/docker_build.sh run

# Or use docker-compose
docker-compose up -d
```

### Model Evaluation & Comparison

```bash
# Evaluate a trained model
python scripts/run_evaluation.py
# or
python3 scripts/run_evaluation.py

# Compare multiple models
python -m minigpt.compare --models checkpoints/*.pt --plots
```

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - feel free to use this for learning and experimentation!

## Acknowledgments

- Inspired by Andrej Karpathy's educational materials
- Uses the Wikitext dataset for training
