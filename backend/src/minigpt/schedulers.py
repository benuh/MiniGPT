"""
Advanced learning rate schedulers and training utilities for MiniGPT
"""

import math
import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any


class WarmupCosineScheduler:
    """
    Cosine annealing scheduler with linear warmup
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        """Update learning rate"""
        self.step_count += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count <= self.warmup_steps:
                # Linear warmup
                lr = base_lr * self.step_count / self.warmup_steps
            else:
                # Cosine annealing
                progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr_ratio * base_lr + (base_lr - self.min_lr_ratio * base_lr) * \
                     0.5 * (1 + math.cos(math.pi * progress))

            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class LinearScheduler:
    """
    Linear learning rate decay
    """
    def __init__(self,
                 optimizer: Optimizer,
                 total_steps: int,
                 min_lr_ratio: float = 0.0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        """Update learning rate"""
        self.step_count += 1
        progress = min(self.step_count / self.total_steps, 1.0)

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = base_lr * (1 - progress * (1 - self.min_lr_ratio))
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting
    """
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/metrics
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.wait_count = 0
        self.stopped_epoch = 0

        self.compare = self._get_compare_fn()

    def _get_compare_fn(self):
        """Get comparison function based on mode"""
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:
            return lambda current, best: current > best + self.min_delta

    def step(self, current_value: float, model=None) -> bool:
        """
        Check if training should be stopped

        Args:
            current_value: Current metric value
            model: Model to save weights from

        Returns:
            True if training should stop, False otherwise
        """
        if self.compare(current_value, self.best_value):
            # Improvement found
            self.best_value = current_value
            self.wait_count = 0

            # Save best weights
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            # No improvement
            self.wait_count += 1

        # Check if should stop
        if self.wait_count >= self.patience:
            self.stopped_epoch = self.wait_count
            return True

        return False

    def restore_weights(self, model):
        """Restore best weights to model"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored best weights from epoch {self.stopped_epoch - self.patience}")


class GradientClipping:
    """
    Advanced gradient clipping utilities
    """
    @staticmethod
    def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
        """Standard gradient norm clipping"""
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

    @staticmethod
    def clip_grad_value(parameters, clip_value: float):
        """Gradient value clipping"""
        return torch.nn.utils.clip_grad_value_(parameters, clip_value)

    @staticmethod
    def adaptive_clip_grad(parameters, clip_factor: float = 0.01, eps: float = 1e-3):
        """
        Adaptive gradient clipping based on parameter norms
        """
        total_norm = 0.0
        param_count = 0

        for p in parameters:
            if p.grad is not None:
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count == 0:
            return 0.0

        total_norm = total_norm ** (1. / 2)
        clip_coef = clip_factor * total_norm / (total_norm + eps)
        clip_coef = min(clip_coef, 1.0)

        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

        return total_norm


class MetricsTracker:
    """
    Track and smooth training metrics
    """
    def __init__(self, smoothing_factor: float = 0.1):
        self.smoothing_factor = smoothing_factor
        self.metrics = {}
        self.history = {}

    def update(self, **kwargs):
        """Update metrics with exponential smoothing"""
        for name, value in kwargs.items():
            if name in self.metrics:
                # Exponential smoothing
                self.metrics[name] = (1 - self.smoothing_factor) * self.metrics[name] + \
                                   self.smoothing_factor * value
            else:
                self.metrics[name] = value

            # Store in history
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

    def get(self, name: str) -> float:
        """Get current smoothed value"""
        return self.metrics.get(name, 0.0)

    def get_history(self, name: str) -> list:
        """Get full history for metric"""
        return self.history.get(name, [])

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.history.clear()


class TrainingProfiler:
    """
    Profile training performance
    """
    def __init__(self):
        self.timings = {}
        self.start_times = {}

    def start(self, name: str):
        """Start timing an operation"""
        import time
        self.start_times[name] = time.time()

    def end(self, name: str):
        """End timing an operation"""
        import time
        if name in self.start_times:
            duration = time.time() - self.start_times[name]

            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)

            del self.start_times[name]
            return duration
        return 0.0

    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        if name not in self.timings:
            return {}

        import numpy as np
        times = self.timings[name]

        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'count': len(times)
        }

    def report(self):
        """Generate profiling report"""
        report = "Training Performance Report\n"
        report += "=" * 40 + "\n"

        for name in self.timings:
            stats = self.get_stats(name)
            if stats:
                report += f"{name:20} | "
                report += f"Mean: {stats['mean']:.3f}s | "
                report += f"Std: {stats['std']:.3f}s | "
                report += f"Count: {stats['count']}\n"

        return report


def get_scheduler(scheduler_type: str, optimizer: Optimizer, **kwargs):
    """
    Factory function to create schedulers
    """
    if scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            kwargs.get('warmup_steps', 100),
            kwargs.get('total_steps', 1000),
            kwargs.get('min_lr_ratio', 0.1)
        )
    elif scheduler_type == "linear":
        return LinearScheduler(
            optimizer,
            kwargs.get('total_steps', 1000),
            kwargs.get('min_lr_ratio', 0.0)
        )
    elif scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('total_steps', 1000),
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_optimizer_with_scheduler(model, config: Dict[str, Any]):
    """
    Create optimizer and scheduler from config
    """
    # Create optimizer
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adamw')

    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.get('lr', 3e-4),
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            betas=optimizer_config.get('betas', (0.9, 0.95))
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.get('lr', 3e-4),
            weight_decay=optimizer_config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Create scheduler
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'warmup_cosine')

    scheduler = get_scheduler(scheduler_type, optimizer, **scheduler_config)

    return optimizer, scheduler