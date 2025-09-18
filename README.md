# MiniGPT 🤖

A minimalist GPT implementation built from scratch for learning and experimentation. This project demonstrates modern ML engineering practices with a clean, educational codebase.

## Features

### 🏗️ Core Architecture
- **Transformer Model**: Clean GPT-style implementation with multi-head attention
- **Modular Design**: Separate components for attention, MLP, and transformer blocks
- **Configurable**: Easy to adjust model size and hyperparameters

### 🚀 Training Pipeline
- **Professional Training Loop**: Complete with validation, checkpointing, and logging
- **Data Loading**: Efficient batching and preprocessing for text data
- **Experiment Tracking**: Integration with Weights & Biases
- **Resumable Training**: Save and resume from checkpoints

### 💬 Inference & Chat
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
git clone https://github.com/benjaminhu/MiniGPT.git
cd MiniGPT
pip install -e .
```

### Training a Model

```bash
# Train with default small config
python -m minigpt.train --config configs/small.yaml

# Resume training from checkpoint
python -m minigpt.train --config configs/small.yaml --resume checkpoints/checkpoint_1000.pt
```

### Chat with Your Model

```bash
# Interactive chat
python -m minigpt.chat --model checkpoints/best_model.pt

# Single prompt
python -m minigpt.chat --model checkpoints/best_model.pt --prompt "Hello, how are you?"
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
- [ ] Model evaluation metrics (perplexity, BLEU)
- [ ] Fine-tuning capabilities
- [ ] Quantization and optimization
- [ ] Web interface
- [ ] Docker containerization
- [ ] Model serving API

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
- Built with PyTorch and Transformers library
- Uses the Wikitext dataset for training

---

**Happy experimenting!** 🚀