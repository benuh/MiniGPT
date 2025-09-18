import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


class ModelConfig:
    """Model configuration class"""
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.n_layer = kwargs.get('n_layer', 4)
        self.n_head = kwargs.get('n_head', 4)
        self.n_embd = kwargs.get('n_embd', 128)
        self.block_size = kwargs.get('block_size', 256)
        self.dropout = kwargs.get('dropout', 0.1)

    def to_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'block_size': self.block_size,
            'dropout': self.dropout
        }


class TrainingConfig:
    """Training configuration class"""
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.warmup_steps = kwargs.get('warmup_steps', 100)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.gradient_clip = kwargs.get('gradient_clip', 1.0)

    def to_dict(self):
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'gradient_clip': self.gradient_clip
        }


def get_default_config():
    """Get default configuration"""
    return {
        'model': ModelConfig().to_dict(),
        'training': TrainingConfig().to_dict(),
        'data': {
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-2-raw-v1',
            'train_split': 'train',
            'val_split': 'validation',
            'max_length': 256
        },
        'logging': {
            'log_interval': 100,
            'eval_interval': 500,
            'save_interval': 1000,
            'project_name': 'minigpt',
            'run_name': 'default_run',
            'use_wandb': False
        }
    }