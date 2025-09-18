import os
import torch
import random
import numpy as np
from typing import Dict, Any


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(checkpoint: Dict[str, Any], filename: str):
    """Save model checkpoint"""
    os.makedirs("checkpoints", exist_ok=True)
    filepath = os.path.join("checkpoints", filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filename: str, device: torch.device) -> Dict[str, Any]:
    """Load model checkpoint"""
    filepath = os.path.join("checkpoints", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def estimate_model_size(vocab_size, n_layer, n_head, n_embd):
    """Estimate model size in parameters"""
    # Token embeddings
    token_emb = vocab_size * n_embd

    # Position embeddings (assuming max 1024 positions)
    pos_emb = 1024 * n_embd

    # Transformer layers
    layer_params = 0
    # Multi-head attention
    qkv_proj = 3 * n_embd * n_embd
    out_proj = n_embd * n_embd
    # MLP
    mlp_up = n_embd * (4 * n_embd)
    mlp_down = (4 * n_embd) * n_embd
    # Layer norms
    ln1 = n_embd
    ln2 = n_embd

    layer_params = qkv_proj + out_proj + mlp_up + mlp_down + ln1 + ln2
    total_layer_params = layer_params * n_layer

    # Final layer norm and language model head
    final_ln = n_embd
    lm_head = n_embd * vocab_size

    total_params = token_emb + pos_emb + total_layer_params + final_ln + lm_head
    return total_params


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return torch.exp(loss).item()


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']