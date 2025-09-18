"""
MiniGPT: A minimalist GPT implementation for learning and experimentation
"""

__version__ = "0.1.0"
__author__ = "Benjamin Hu"

from .model import MiniGPT, MultiHeadAttention, MLP, Block
from .tokenizer import SimpleTokenizer, GPT2TokenizerWrapper, get_tokenizer
from .train import Trainer, TextDataset
from .chat import ChatBot
from .config import load_config, save_config, ModelConfig, TrainingConfig, get_default_config
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

__all__ = [
    "MiniGPT",
    "MultiHeadAttention",
    "MLP",
    "Block",
    "SimpleTokenizer",
    "GPT2TokenizerWrapper",
    "get_tokenizer",
    "Trainer",
    "TextDataset",
    "ChatBot",
    "load_config",
    "save_config",
    "ModelConfig",
    "TrainingConfig",
    "get_default_config",
    "get_device",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
    "count_parameters",
    "format_number",
    "estimate_model_size",
    "AverageMeter",
    "calculate_perplexity",
    "get_lr"
]