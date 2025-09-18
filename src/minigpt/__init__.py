"""
MiniGPT: A minimalist GPT implementation for learning and experimentation

A comprehensive toolkit for building, training, and deploying small-scale GPT models
with modern ML engineering practices.
"""

__version__ = "1.0.0"
__author__ = "Benjamin Hu"

# Core model and architecture
from .model import MiniGPT, MultiHeadAttention, MLP, Block

# Tokenization
from .tokenizer import SimpleTokenizer, GPT2TokenizerWrapper, get_tokenizer

# Training and data
from .train import Trainer, TextDataset

# Inference and chat
from .chat import ChatBot

# Configuration management
from .config import load_config, save_config, ModelConfig, TrainingConfig, get_default_config

# Utilities
from .utils import (
    get_device,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    format_number,
    estimate_model_size,
    AverageMeter,
    calculate_perplexity,
    get_lr
)

# Advanced training features
from .schedulers import (
    WarmupCosineScheduler,
    LinearScheduler,
    EarlyStopping,
    GradientClipping,
    MetricsTracker,
    TrainingProfiler,
    get_scheduler,
    create_optimizer_with_scheduler
)

# Evaluation and metrics
from .evaluate import ModelEvaluator

# Model optimization
from .optimize import ModelOptimizer

# Model comparison
from .compare import ModelComparator

# API server (optional import)
try:
    from .api import ModelManager
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False
    ModelManager = None

__all__ = [
    # Core components
    "MiniGPT",
    "MultiHeadAttention",
    "MLP",
    "Block",

    # Tokenization
    "SimpleTokenizer",
    "GPT2TokenizerWrapper",
    "get_tokenizer",

    # Training
    "Trainer",
    "TextDataset",

    # Inference
    "ChatBot",

    # Configuration
    "load_config",
    "save_config",
    "ModelConfig",
    "TrainingConfig",
    "get_default_config",

    # Utilities
    "get_device",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "format_number",
    "estimate_model_size",
    "AverageMeter",
    "calculate_perplexity",
    "get_lr",

    # Advanced training
    "WarmupCosineScheduler",
    "LinearScheduler",
    "EarlyStopping",
    "GradientClipping",
    "MetricsTracker",
    "TrainingProfiler",
    "get_scheduler",
    "create_optimizer_with_scheduler",

    # Evaluation
    "ModelEvaluator",

    # Optimization
    "ModelOptimizer",

    # Comparison
    "ModelComparator",

    # API (if available)
    "ModelManager",
]

# Convenience functions
def quick_train(config_path: str = "configs/small.yaml", **kwargs):
    """Quick training with default settings"""
    trainer = Trainer(config_path)
    trainer.train()
    return trainer

def quick_chat(model_path: str):
    """Quick chat interface"""
    chatbot = ChatBot(model_path)
    chatbot.chat_loop()
    return chatbot

def quick_evaluate(model_path: str):
    """Quick model evaluation"""
    import torch
    from .utils import load_checkpoint

    device = get_device()
    checkpoint = load_checkpoint(model_path, device)
    config = checkpoint.get('config', {})

    model = MiniGPT(**config.get('model', {})).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    tokenizer = get_tokenizer("gpt2")
    evaluator = ModelEvaluator(model, tokenizer, device)

    # Simple evaluation with default texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming our world.",
        "The future of artificial intelligence looks bright."
    ]

    prompts = [
        "Hello world",
        "The future of",
        "Once upon a time"
    ]

    return evaluator.comprehensive_evaluation(test_texts, prompts)

# Package info
def get_info():
    """Get package information"""
    info = {
        'name': 'MiniGPT',
        'version': __version__,
        'author': __author__,
        'description': __doc__.split('\n')[1],
        'api_available': _API_AVAILABLE,
        'components': {
            'core': ['MiniGPT', 'Trainer', 'ChatBot'],
            'advanced': ['ModelEvaluator', 'ModelOptimizer', 'ModelComparator'],
            'training': ['WarmupCosineScheduler', 'EarlyStopping', 'MetricsTracker'],
            'utilities': ['get_device', 'load_config', 'count_parameters']
        }
    }
    return info