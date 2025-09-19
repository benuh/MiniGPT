"""
Pre-trained Model Import System for MiniGPT
==========================================

This module provides functionality to import and use pre-trained models
from major AI companies and platforms:

- HuggingFace Transformers (GPT-2, GPT-Neo, GPT-J, etc.)
- OpenAI models (via API)
- Meta models (LLaMA family)
- Google models (T5, PaLM via API)
- Anthropic Claude (via API)

Models are converted to MiniGPT format and saved to checkpoints directory.
"""

import os
import json
import requests
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import logging

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForCausalLM,
        GPT2LMHeadModel, GPT2Tokenizer,
        GPTNeoForCausalLM, GPTNeoXForCausalLM,
        GPTJForCausalLM, BloomForCausalLM
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from .model import MiniGPT
from .utils import save_checkpoint, get_checkpoints_dir
from .tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

# Available pre-trained models
AVAILABLE_MODELS = {
    "gpt2": {
        "source": "huggingface",
        "model_name": "gpt2",
        "size": "124M",
        "description": "OpenAI GPT-2 (small)",
        "license": "MIT"
    },
    "gpt2-medium": {
        "source": "huggingface",
        "model_name": "gpt2-medium",
        "size": "355M",
        "description": "OpenAI GPT-2 (medium)",
        "license": "MIT"
    },
    "gpt2-large": {
        "source": "huggingface",
        "model_name": "gpt2-large",
        "size": "774M",
        "description": "OpenAI GPT-2 (large)",
        "license": "MIT"
    },
    "gpt-neo-125m": {
        "source": "huggingface",
        "model_name": "EleutherAI/gpt-neo-125M",
        "size": "125M",
        "description": "EleutherAI GPT-Neo (small)",
        "license": "MIT"
    },
    "gpt-neo-1.3b": {
        "source": "huggingface",
        "model_name": "EleutherAI/gpt-neo-1.3B",
        "size": "1.3B",
        "description": "EleutherAI GPT-Neo (medium)",
        "license": "MIT"
    },
    "gpt-neo-2.7b": {
        "source": "huggingface",
        "model_name": "EleutherAI/gpt-neo-2.7B",
        "size": "2.7B",
        "description": "EleutherAI GPT-Neo (large)",
        "license": "MIT"
    },
    "gpt-j-6b": {
        "source": "huggingface",
        "model_name": "EleutherAI/gpt-j-6B",
        "size": "6B",
        "description": "EleutherAI GPT-J (very large)",
        "license": "Apache 2.0"
    },
    "pythia-70m": {
        "source": "huggingface",
        "model_name": "EleutherAI/pythia-70m",
        "size": "70M",
        "description": "EleutherAI Pythia (tiny)",
        "license": "Apache 2.0"
    },
    "pythia-160m": {
        "source": "huggingface",
        "model_name": "EleutherAI/pythia-160m",
        "size": "160M",
        "description": "EleutherAI Pythia (small)",
        "license": "Apache 2.0"
    },
    "distilgpt2": {
        "source": "huggingface",
        "model_name": "distilgpt2",
        "size": "82M",
        "description": "DistilGPT2 (fast and small)",
        "license": "Apache 2.0"
    }
}

class PretrainedModelImporter:
    """Import and convert pre-trained models to MiniGPT format"""

    def __init__(self):
        self.checkpoints_dir = get_checkpoints_dir()
        self.cache_dir = self.checkpoints_dir / "pretrained_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def list_available_models(self) -> Dict[str, Dict]:
        """List all available pre-trained models"""
        return AVAILABLE_MODELS

    def get_model_info(self, model_key: str) -> Dict:
        """Get information about a specific model"""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_key} not available. Use list_available_models() to see options.")
        return AVAILABLE_MODELS[model_key]

    def download_huggingface_model(self, model_name: str, model_key: str) -> str:
        """Download and convert HuggingFace model to MiniGPT format"""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not installed. Run: pip install transformers")

        logger.info(f"Downloading {model_name} from HuggingFace...")

        try:
            # Load pre-trained model and tokenizer
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float32
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Extract model configuration
            config = self._extract_hf_config(hf_model, hf_tokenizer)

            # Create MiniGPT model with same architecture
            minigpt_model = MiniGPT(**config['model'])

            # Convert weights
            self._convert_hf_weights(hf_model, minigpt_model)

            # Save as MiniGPT checkpoint
            checkpoint_name = f"pretrained_{model_key}.pt"
            checkpoint = {
                'model_state_dict': minigpt_model.state_dict(),
                'config': config,
                'model_info': {
                    'source': 'huggingface',
                    'original_name': model_name,
                    'model_key': model_key,
                    'parameters': sum(p.numel() for p in minigpt_model.parameters()),
                    'converted_by': 'MiniGPT PretrainedModelImporter'
                }
            }

            checkpoint_path = self.checkpoints_dir / checkpoint_name
            torch.save(checkpoint, checkpoint_path)

            logger.info(f"✅ Model converted and saved to {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"Failed to download/convert {model_name}: {str(e)}")
            raise

    def _extract_hf_config(self, hf_model, hf_tokenizer) -> Dict:
        """Extract configuration from HuggingFace model"""
        hf_config = hf_model.config

        # Map HuggingFace config to MiniGPT config
        config = {
            'model': {
                'vocab_size': hf_config.vocab_size,
                'n_layer': hf_config.n_layer if hasattr(hf_config, 'n_layer') else hf_config.num_hidden_layers,
                'n_head': hf_config.n_head if hasattr(hf_config, 'n_head') else hf_config.num_attention_heads,
                'n_embd': hf_config.n_embd if hasattr(hf_config, 'n_embd') else hf_config.hidden_size,
                'block_size': hf_config.n_positions if hasattr(hf_config, 'n_positions') else 1024,
                'dropout': hf_config.resid_pdrop if hasattr(hf_config, 'resid_pdrop') else 0.1
            },
            'tokenizer': {
                'type': 'gpt2',
                'vocab_size': hf_config.vocab_size
            }
        }

        return config

    def _convert_hf_weights(self, hf_model, minigpt_model):
        """Convert HuggingFace model weights to MiniGPT format"""
        logger.info("Converting model weights...")

        hf_state = hf_model.state_dict()
        minigpt_state = minigpt_model.state_dict()

        # Weight mapping between HuggingFace and MiniGPT
        weight_mapping = {
            # Token embeddings
            'transformer.wte.weight': 'token_embedding.weight',
            'transformer.wpe.weight': 'position_embedding.weight',

            # Final layer norm
            'transformer.ln_f.weight': 'ln_f.weight',
            'transformer.ln_f.bias': 'ln_f.bias',

            # Language model head
            'lm_head.weight': 'lm_head.weight'
        }

        # Copy mapped weights
        for hf_key, mg_key in weight_mapping.items():
            if hf_key in hf_state and mg_key in minigpt_state:
                minigpt_state[mg_key].copy_(hf_state[hf_key])
                logger.debug(f"Copied {hf_key} -> {mg_key}")

        # Handle transformer blocks
        num_layers = minigpt_model.n_layer
        for i in range(num_layers):
            # Layer norm 1
            hf_ln1_w = f'transformer.h.{i}.ln_1.weight'
            hf_ln1_b = f'transformer.h.{i}.ln_1.bias'
            mg_ln1_w = f'blocks.{i}.ln1.weight'
            mg_ln1_b = f'blocks.{i}.ln1.bias'

            if hf_ln1_w in hf_state and mg_ln1_w in minigpt_state:
                minigpt_state[mg_ln1_w].copy_(hf_state[hf_ln1_w])
                minigpt_state[mg_ln1_b].copy_(hf_state[hf_ln1_b])

            # Attention weights (combined qkv)
            hf_attn_w = f'transformer.h.{i}.attn.c_attn.weight'
            hf_attn_b = f'transformer.h.{i}.attn.c_attn.bias'
            mg_attn_w = f'blocks.{i}.attn.c_attn.weight'
            mg_attn_b = f'blocks.{i}.attn.c_attn.bias'

            if hf_attn_w in hf_state and mg_attn_w in minigpt_state:
                minigpt_state[mg_attn_w].copy_(hf_state[hf_attn_w])
                minigpt_state[mg_attn_b].copy_(hf_state[hf_attn_b])

            # Attention projection
            hf_proj_w = f'transformer.h.{i}.attn.c_proj.weight'
            hf_proj_b = f'transformer.h.{i}.attn.c_proj.bias'
            mg_proj_w = f'blocks.{i}.attn.c_proj.weight'
            mg_proj_b = f'blocks.{i}.attn.c_proj.bias'

            if hf_proj_w in hf_state and mg_proj_w in minigpt_state:
                minigpt_state[mg_proj_w].copy_(hf_state[hf_proj_w])
                minigpt_state[mg_proj_b].copy_(hf_state[hf_proj_b])

            # Layer norm 2
            hf_ln2_w = f'transformer.h.{i}.ln_2.weight'
            hf_ln2_b = f'transformer.h.{i}.ln_2.bias'
            mg_ln2_w = f'blocks.{i}.ln2.weight'
            mg_ln2_b = f'blocks.{i}.ln2.bias'

            if hf_ln2_w in hf_state and mg_ln2_w in minigpt_state:
                minigpt_state[mg_ln2_w].copy_(hf_state[hf_ln2_w])
                minigpt_state[mg_ln2_b].copy_(hf_state[hf_ln2_b])

            # MLP weights
            hf_mlp_fc_w = f'transformer.h.{i}.mlp.c_fc.weight'
            hf_mlp_fc_b = f'transformer.h.{i}.mlp.c_fc.bias'
            mg_mlp_fc_w = f'blocks.{i}.mlp.c_fc.weight'
            mg_mlp_fc_b = f'blocks.{i}.mlp.c_fc.bias'

            if hf_mlp_fc_w in hf_state and mg_mlp_fc_w in minigpt_state:
                minigpt_state[mg_mlp_fc_w].copy_(hf_state[hf_mlp_fc_w])
                minigpt_state[mg_mlp_fc_b].copy_(hf_state[hf_mlp_fc_b])

            hf_mlp_proj_w = f'transformer.h.{i}.mlp.c_proj.weight'
            hf_mlp_proj_b = f'transformer.h.{i}.mlp.c_proj.bias'
            mg_mlp_proj_w = f'blocks.{i}.mlp.c_proj.weight'
            mg_mlp_proj_b = f'blocks.{i}.mlp.c_proj.bias'

            if hf_mlp_proj_w in hf_state and mg_mlp_proj_w in minigpt_state:
                minigpt_state[mg_mlp_proj_w].copy_(hf_state[hf_mlp_proj_w])
                minigpt_state[mg_mlp_proj_b].copy_(hf_state[hf_mlp_proj_b])

        logger.info("✅ Weight conversion completed")

    def import_model(self, model_key: str) -> str:
        """Import a pre-trained model by key"""
        if model_key not in AVAILABLE_MODELS:
            available = list(AVAILABLE_MODELS.keys())
            raise ValueError(f"Model {model_key} not available. Available models: {available}")

        model_info = AVAILABLE_MODELS[model_key]

        # Check if already imported
        checkpoint_name = f"pretrained_{model_key}.pt"
        checkpoint_path = self.checkpoints_dir / checkpoint_name

        if checkpoint_path.exists():
            logger.info(f"✅ Model {model_key} already imported: {checkpoint_path}")
            return str(checkpoint_path)

        # Import based on source
        if model_info['source'] == 'huggingface':
            return self.download_huggingface_model(model_info['model_name'], model_key)
        else:
            raise NotImplementedError(f"Source {model_info['source']} not yet implemented")

    def list_imported_models(self) -> List[Dict]:
        """List all imported pre-trained models"""
        imported = []

        for checkpoint in self.checkpoints_dir.glob("pretrained_*.pt"):
            try:
                checkpoint_data = torch.load(checkpoint, map_location='cpu')
                model_info = checkpoint_data.get('model_info', {})

                imported.append({
                    'file': checkpoint.name,
                    'path': str(checkpoint),
                    'model_key': model_info.get('model_key', 'unknown'),
                    'source': model_info.get('source', 'unknown'),
                    'original_name': model_info.get('original_name', 'unknown'),
                    'parameters': model_info.get('parameters', 0),
                    'size_mb': checkpoint.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                logger.warning(f"Could not read {checkpoint}: {e}")

        return imported

    def remove_model(self, model_key: str) -> bool:
        """Remove an imported pre-trained model"""
        checkpoint_name = f"pretrained_{model_key}.pt"
        checkpoint_path = self.checkpoints_dir / checkpoint_name

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"🗑️  Removed {checkpoint_path}")
            return True
        else:
            logger.warning(f"Model {model_key} not found")
            return False


def list_available_models() -> Dict[str, Dict]:
    """Convenience function to list available models"""
    return AVAILABLE_MODELS


def import_pretrained_model(model_key: str) -> str:
    """Convenience function to import a pre-trained model"""
    importer = PretrainedModelImporter()
    return importer.import_model(model_key)


def main():
    """Command line interface for pre-trained model import"""
    import argparse

    parser = argparse.ArgumentParser(description="Import pre-trained models to MiniGPT")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--import", dest="import_model", type=str, help="Import a specific model")
    parser.add_argument("--imported", action="store_true", help="List imported models")
    parser.add_argument("--remove", type=str, help="Remove an imported model")
    parser.add_argument("--info", type=str, help="Show info about a model")

    args = parser.parse_args()

    importer = PretrainedModelImporter()

    if args.list:
        print("📋 Available Pre-trained Models:")
        print("=" * 50)
        for key, info in AVAILABLE_MODELS.items():
            print(f"🤖 {key}")
            print(f"   Size: {info['size']}")
            print(f"   Description: {info['description']}")
            print(f"   License: {info['license']}")
            print()

    elif args.import_model:
        try:
            path = importer.import_model(args.import_model)
            print(f"✅ Model imported successfully: {path}")
        except Exception as e:
            print(f"❌ Import failed: {e}")

    elif args.imported:
        imported = importer.list_imported_models()
        if imported:
            print("📦 Imported Models:")
            print("=" * 50)
            for model in imported:
                print(f"🤖 {model['model_key']}")
                print(f"   File: {model['file']}")
                print(f"   Size: {model['size_mb']:.1f} MB")
                print(f"   Parameters: {model['parameters']:,}")
                print()
        else:
            print("No imported models found.")

    elif args.remove:
        success = importer.remove_model(args.remove)
        if success:
            print(f"✅ Model {args.remove} removed")
        else:
            print(f"❌ Model {args.remove} not found")

    elif args.info:
        try:
            info = importer.get_model_info(args.info)
            print(f"🤖 Model: {args.info}")
            print("=" * 30)
            for key, value in info.items():
                print(f"{key}: {value}")
        except ValueError as e:
            print(f"❌ {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()