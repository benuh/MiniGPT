# MiniGPT Usage Guide 🚀

Complete guide for using MiniGPT - from basic training to advanced deployment.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Training Models](#training-models)
4. [Advanced Training Features](#advanced-training-features)
5. [Model Evaluation](#model-evaluation)
6. [Model Optimization](#model-optimization)
7. [Model Comparison](#model-comparison)
8. [API Deployment](#api-deployment)
9. [Automation & Continuous Improvement](#automation--continuous-improvement)
10. [Docker Deployment](#docker-deployment)
11. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

```bash
git clone https://github.com/benuh/MiniGPT.git
cd MiniGPT
pip install -e .
```

### 30-Second Demo

```bash
# Train a small model (5 minutes)
python -m minigpt.train --config configs/small.yaml

# Chat with your model
python -m minigpt.chat --model checkpoints/best_model.pt

# Evaluate the model
python scripts/run_evaluation.py
```

## Basic Usage

### 1. Training Your First Model

```bash
# Train with default small configuration
python -m minigpt.train --config configs/small.yaml

# Resume training from checkpoint
python -m minigpt.train --config configs/small.yaml --resume checkpoints/checkpoint_1000.pt
```

### 2. Chatting with Your Model

```bash
# Interactive chat
python -m minigpt.chat --model checkpoints/best_model.pt

# Single prompt
python -m minigpt.chat --model checkpoints/best_model.pt --prompt "Hello, how are you?"
```

### 3. Using Python API

```python
import minigpt

# Quick training
trainer = minigpt.quick_train("configs/small.yaml")

# Quick chat
minigpt.quick_chat("checkpoints/best_model.pt")

# Quick evaluation
results = minigpt.quick_evaluate("checkpoints/best_model.pt")
print(f"Perplexity: {results['perplexity']:.2f}")
```

## Training Models

### Configuration Files

Create custom model configurations:

```yaml
# configs/my_model.yaml
model:
  vocab_size: 50257
  n_layer: 6          # Number of transformer layers
  n_head: 6           # Number of attention heads
  n_embd: 256         # Embedding dimension
  block_size: 512     # Context length
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 1e-4
  max_epochs: 15
  warmup_steps: 200
  weight_decay: 0.01
  gradient_clip: 1.0

data:
  dataset_name: "wikitext"
  dataset_config: "wikitext-103-raw-v1"
  max_length: 512

logging:
  log_interval: 50
  eval_interval: 300
  save_interval: 500
  use_wandb: true
```

### Advanced Training Script

```python
from minigpt import Trainer, EarlyStopping, WarmupCosineScheduler

# Create trainer with advanced features
trainer = Trainer("configs/my_model.yaml")

# Add early stopping
early_stopping = EarlyStopping(patience=5, min_delta=0.01)

# Custom training loop with advanced features
for epoch in range(trainer.config['training']['max_epochs']):
    train_loss = trainer.train_epoch()
    val_loss = trainer.validate()

    # Check early stopping
    if early_stopping.step(val_loss, trainer.model):
        print("Early stopping triggered")
        early_stopping.restore_weights(trainer.model)
        break
```

## Advanced Training Features

### Learning Rate Scheduling

```python
from minigpt.schedulers import WarmupCosineScheduler, get_scheduler

# Create custom scheduler
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_steps=100,
    total_steps=1000,
    min_lr_ratio=0.1
)

# Use scheduler factory
scheduler = get_scheduler("warmup_cosine", optimizer,
                         warmup_steps=100, total_steps=1000)
```

### Early Stopping

```python
from minigpt.schedulers import EarlyStopping

# Set up early stopping
early_stopping = EarlyStopping(
    patience=10,        # Wait 10 epochs
    min_delta=0.001,   # Minimum improvement
    mode='min',        # For loss (use 'max' for accuracy)
    restore_best_weights=True
)

# In training loop
if early_stopping.step(val_loss, model):
    print("Training stopped early")
    break
```

### Gradient Clipping

```python
from minigpt.schedulers import GradientClipping

# Different clipping strategies
GradientClipping.clip_grad_norm(model.parameters(), max_norm=1.0)
GradientClipping.clip_grad_value(model.parameters(), clip_value=0.5)
GradientClipping.adaptive_clip_grad(model.parameters(), clip_factor=0.01)
```

## Model Evaluation

### Basic Evaluation

```bash
# Evaluate latest checkpoint
python scripts/run_evaluation.py

# Evaluate specific model
python -m minigpt.evaluate --model checkpoints/my_model.pt --output evaluation.json
```

### Advanced Evaluation

```python
from minigpt import ModelEvaluator, MiniGPT, get_tokenizer
from minigpt.utils import load_checkpoint, get_device

# Load model
device = get_device()
checkpoint = load_checkpoint("checkpoints/best_model.pt", device)
model = MiniGPT(**checkpoint['config']['model']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Create evaluator
tokenizer = get_tokenizer("gpt2")
evaluator = ModelEvaluator(model, tokenizer, device)

# Comprehensive evaluation
test_texts = ["Your test texts here..."]
prompts = ["Test prompts..."]
results = evaluator.comprehensive_evaluation(test_texts, prompts)

# Save detailed report
evaluator.save_evaluation_report(results, "detailed_evaluation.json")
```

### Custom Metrics

```python
# Calculate specific metrics
perplexity = evaluator.calculate_perplexity(test_texts)
bleu_score = evaluator.calculate_bleu_score(references, hypotheses)

# Generation analysis
generation_results = evaluator.generate_and_evaluate(prompts)
print(f"Average length: {generation_results['avg_length']}")
print(f"Unique tokens: {generation_results['unique_token_count']}")
```

## Model Optimization

### Pruning and Quantization

```bash
# Optimize model with command line
python -m minigpt.optimize --model checkpoints/best_model.pt --pruning-ratio 0.2
```

### Advanced Optimization

```python
from minigpt.optimize import ModelOptimizer

# Load model
optimizer = ModelOptimizer(model, device)

# Comprehensive optimization
results = optimizer.comprehensive_optimization(
    pruning_ratio=0.15,
    quantize=True,
    benchmark=True
)

# Save optimized model
optimizer.save_optimized_model("optimized_model.pt", results)

print(f"Size reduction: {results['original_size_mb']:.1f}MB → {results['final_size_mb']:.1f}MB")
print(f"Compression: {results['total_compression_ratio']:.1f}x")
```

### Inference Optimization

```python
# Optimize for inference
optimizer.optimize_for_inference()

# Benchmark performance
benchmark_results = optimizer.benchmark_inference([1, 4, 8, 16])
print(f"Throughput (batch=1): {benchmark_results[1]['throughput_tokens_per_sec']:.0f} tokens/sec")
```

## Model Comparison

### Command Line Comparison

```bash
# Compare multiple models
python -m minigpt.compare \
  --models checkpoints/model_1.pt checkpoints/model_2.pt checkpoints/model_3.pt \
  --names "Small" "Medium" "Large" \
  --plots
```

### Programmatic Comparison

```python
from minigpt.compare import ModelComparator

# Initialize comparator
comparator = ModelComparator()

# Add models to compare
comparator.add_model("Small", "checkpoints/small_model.pt")
comparator.add_model("Medium", "checkpoints/medium_model.pt")
comparator.add_model("Optimized", "checkpoints/optimized_model.pt")

# Run comprehensive comparison
test_texts = ["Test texts for perplexity..."]
prompts = ["Test prompts for generation..."]
results = comparator.comprehensive_comparison(test_texts, prompts)

# Generate visualization
comparator.create_comparison_plots("comparison_plots/")
comparator.save_comparison_report("model_comparison.json")

# Get best model
best_quality = comparator.get_best_model('perplexity')
best_speed = comparator.get_best_model('throughput')
print(f"Best quality: {best_quality}")
print(f"Fastest: {best_speed}")
```

## API Deployment

### Starting the API Server

```bash
# Basic server
python scripts/start_server.py

# Custom configuration
python scripts/start_server.py --host 0.0.0.0 --port 8080 --model checkpoints/best_model.pt
```

### API Usage Examples

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Text generation
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "The future of artificial intelligence",
    "max_length": 100,
    "temperature": 0.8,
    "top_k": 50
})
print(response.json()["generated_text"])

# Chat
response = requests.post("http://localhost:8000/chat", json={
    "message": "Hello, how are you?",
    "max_length": 50
})
print(response.json()["response"])
```

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Automation & Continuous Improvement

### Automated Training

```bash
# Start continuous improvement system
chmod +x scripts/start_continuous.sh
./scripts/start_continuous.sh
```

### Manual Control

```bash
# Start automation with custom settings
python scripts/monitor.py --session-hours 8 --max-restarts 5

# Check status
python scripts/monitor.py --action status

# Manually restart
python scripts/monitor.py --action restart
```

### Advanced Automation

```python
from scripts.auto_train import AutoTrainer

# Custom automation
auto_trainer = AutoTrainer("configs/small.yaml")
auto_trainer.run_continuous_improvement(max_iterations=20, improvement_threshold=0.05)
```

## Docker Deployment

### Quick Docker Setup

```bash
# Build and run
./scripts/docker_build.sh build
./scripts/docker_build.sh run

# Check logs
./scripts/docker_build.sh logs --follow

# Open shell
./scripts/docker_build.sh shell
```

### Docker Compose

```bash
# Full stack deployment
docker-compose up -d

# Scale API instances
docker-compose up -d --scale minigpt-api=3

# View logs
docker-compose logs -f minigpt-api
```

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  minigpt-api:
    image: minigpt:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    environment:
      - MODEL_PATH=/models/production_model.pt
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in config
batch_size: 8  # Instead of 32

# Or use gradient accumulation
gradient_accumulation_steps: 4
```

#### Slow Training
```python
# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits, loss = model(input_ids, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### Model Not Learning
```yaml
# Try different learning rates
learning_rate: 1e-3  # Higher for small models
learning_rate: 1e-5  # Lower for large models

# Add warmup
warmup_steps: 500

# Check gradient clipping
gradient_clip: 1.0
```

### Performance Optimization

#### CPU Training
```python
# Optimize for CPU
torch.set_num_threads(4)  # Adjust based on your CPU
```

#### Memory Optimization
```python
# Use checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    return checkpoint(self.transformer_block, x)
```

### Debugging

#### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Profile Training
```python
from minigpt.schedulers import TrainingProfiler

profiler = TrainingProfiler()
profiler.start("forward_pass")
# ... forward pass code ...
profiler.end("forward_pass")
print(profiler.report())
```

## Advanced Use Cases

### Fine-tuning on Custom Data

```python
from minigpt import Trainer
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        # Custom data processing
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

# Use custom dataset in training
trainer.train_loader = DataLoader(CustomDataset(my_texts, tokenizer, 256))
```

### Multi-GPU Training

```python
# Enable data parallel training
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### Export to ONNX

```python
import torch.onnx

# Export model to ONNX format
dummy_input = torch.randint(0, model.vocab_size, (1, 128))
torch.onnx.export(model, dummy_input, "minigpt.onnx",
                  export_params=True, opset_version=11)
```

## Getting Help

- 📖 **Documentation**: Check this guide and README.md
- 🐛 **Issues**: [GitHub Issues](https://github.com/benuh/MiniGPT/issues)
- 💡 **Features**: Submit feature requests on GitHub
- 🔧 **Configuration**: Check `configs/` directory for examples

## Tips for Success

1. **Start Small**: Begin with `configs/small.yaml` to understand the system
2. **Monitor Training**: Use wandb integration for experiment tracking
3. **Experiment**: Try different architectures and hyperparameters
4. **Optimize**: Use pruning and quantization for deployment
5. **Compare**: Use model comparison tools to track improvements
6. **Automate**: Let the system run continuous improvements
7. **Deploy**: Use Docker and API for production deployment

Happy experimenting with MiniGPT! 🚀