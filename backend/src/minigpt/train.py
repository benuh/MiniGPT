import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import wandb
from typing import Dict, Any

from .model import MiniGPT
from .tokenizer import get_tokenizer
from .config import load_config
from .utils import get_device, save_checkpoint, load_checkpoint


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, max_length=self.max_length)

        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        # Create input and target sequences
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, targets


class Trainer:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.device = get_device()

        self._print_header()

        # Initialize tokenizer
        print("📚 Initializing tokenizer...")
        self.tokenizer = get_tokenizer("gpt2")
        print("   ✅ Tokenizer ready")

        # Initialize model
        print("🧠 Building model architecture...")
        self.model = MiniGPT(**self.config['model']).to(self.device)
        self._print_model_info()

        # Initialize optimizer
        print("⚙️  Setting up optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        print(f"   ✅ AdamW optimizer (LR: {self.config['training']['learning_rate']:.2e})")

        # Initialize data loaders
        print("📊 Preparing datasets...")
        self.train_loader, self.val_loader = self._prepare_data()

        # Initialize wandb logging
        if self.config.get('logging', {}).get('use_wandb', False):
            print("📈 Initializing Weights & Biases...")
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['run_name'],
                config=self.config
            )
            print("   ✅ W&B logging enabled")
        else:
            print("📝 Using local logging only")

        self.global_step = 0
        self.best_val_loss = float('inf')

        print("🚀 Training setup complete!")
        print("=" * 60)

    def _print_header(self):
        """Print training header with configuration info"""
        print("\n" + "=" * 60)
        print("🤖 MiniGPT Training Pipeline")
        print("=" * 60)
        print(f"📱 Device: {self.device}")
        print(f"🏗️  Model Architecture:")
        for key, value in self.config['model'].items():
            print(f"   • {key}: {value}")
        print(f"🎯 Training Configuration:")
        for key, value in self.config['training'].items():
            print(f"   • {key}: {value}")
        print("-" * 60)

    def _print_model_info(self):
        """Print detailed model information"""
        param_count = self.model.count_parameters()
        print(f"   ✅ Model created: {param_count:,} parameters")
        print(f"   📏 Context length: {self.model.block_size}")
        print(f"   🧩 Vocabulary size: {self.model.vocab_size}")
        print(f"   🏗️  Layers: {len(self.model.blocks)}")
        if self.model.blocks:
            print(f"   🔗 Attention heads: {self.model.blocks[0].attn.n_head}")
            print(f"   📐 Embedding dim: {self.model.blocks[0].attn.n_embd}")

    def _prepare_data(self):
        """Prepare training and validation data loaders"""
        data_config = self.config['data']

        # Load dataset
        print(f"   📥 Loading {data_config['dataset_name']} dataset...")
        if data_config['dataset_name'] == "wikitext":
            dataset = load_dataset("wikitext", data_config['dataset_config'])
        else:
            raise ValueError(f"Unsupported dataset: {data_config['dataset_name']}")

        # Process texts
        print("   🔍 Processing text data...")
        train_texts = [item['text'] for item in dataset[data_config['train_split']] if item['text'].strip()]
        val_texts = [item['text'] for item in dataset[data_config['val_split']] if item['text'].strip()]

        print(f"   📊 Dataset statistics:")
        print(f"      • Training texts: {len(train_texts):,}")
        print(f"      • Validation texts: {len(val_texts):,}")
        print(f"      • Max sequence length: {data_config['max_length']}")

        # Create datasets
        train_dataset = TextDataset(train_texts, self.tokenizer, data_config['max_length'])
        val_dataset = TextDataset(val_texts, self.tokenizer, data_config['max_length'])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=2
        )

        print(f"   ✅ Data loaders ready:")
        print(f"      • Training batches: {len(train_loader)}")
        print(f"      • Validation batches: {len(val_loader)}")
        print(f"      • Batch size: {self.config['training']['batch_size']}")

        return train_loader, val_loader

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader,
            desc="   🔄 Training",
            bar_format="      {l_bar}{bar:30}{r_bar}",
            ncols=80
        )

        for batch_idx, (input_ids, targets) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            logits, loss = self.model(input_ids, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )

            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg': f"{total_loss / (batch_idx + 1):.4f}",
                'LR': f"{current_lr:.2e}",
                'Step': f"{self.global_step}"
            })

            # Log to wandb
            if self.config.get('logging', {}).get('use_wandb', False):
                if self.global_step % self.config['logging']['log_interval'] == 0:
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })

            # Validation
            if self.global_step % self.config['logging']['eval_interval'] == 0:
                print(f"\n      🔍 Running validation at step {self.global_step}...")
                val_loss = self.validate()
                self.model.train()

                if val_loss < self.best_val_loss:
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    print(f"      🎉 New best model! Improved by {improvement:.4f}")
                    self.save_checkpoint('best_model.pt')
                else:
                    print(f"      📊 No improvement (best: {self.best_val_loss:.4f})")

            # Save checkpoint
            if self.global_step % self.config['logging']['save_interval'] == 0:
                print(f"      💾 Saving checkpoint at step {self.global_step}")
                self.save_checkpoint(f'checkpoint_{self.global_step}.pt')

        return total_loss / num_batches

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            val_progress = tqdm(
                self.val_loader,
                desc="         Validating",
                bar_format="         {l_bar}{bar:25}{r_bar}",
                ncols=70,
                leave=False
            )
            for input_ids, targets in val_progress:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                logits, loss = self.model(input_ids, targets)
                total_loss += loss.item()

                val_progress.set_postfix({'Loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches

        # Log to wandb
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.log({
                'val_loss': avg_loss,
                'global_step': self.global_step
            })

        print(f"         ✅ Validation complete: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        """Main training loop"""
        print("\n🚀 Starting training process...")
        print("=" * 60)

        import time
        start_time = time.time()

        for epoch in range(self.config['training']['max_epochs']):
            epoch_start = time.time()

            print(f"\n📅 Epoch {epoch + 1}/{self.config['training']['max_epochs']}")
            print("-" * 40)

            train_loss = self.train_epoch()
            val_loss = self.validate()

            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time

            print(f"\n   📊 Epoch {epoch + 1} Summary:")
            print(f"      • Training loss:   {train_loss:.4f}")
            print(f"      • Validation loss: {val_loss:.4f}")
            print(f"      • Best val loss:   {self.best_val_loss:.4f}")
            print(f"      • Epoch time:      {epoch_time:.1f}s")
            print(f"      • Total time:      {total_time/60:.1f}m")

            # Calculate perplexity
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            val_ppl = torch.exp(torch.tensor(val_loss)).item()
            print(f"      • Train perplexity: {train_ppl:.2f}")
            print(f"      • Val perplexity:   {val_ppl:.2f}")

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")
        print(f"   • Total training time: {total_time/60:.1f} minutes")
        print(f"   • Final validation loss: {self.best_val_loss:.4f}")
        print(f"   • Total steps: {self.global_step}")

        print("💾 Saving final model...")
        self.save_checkpoint('final_model.pt')
        print("✅ Final model saved as 'final_model.pt'")
        print("=" * 60)

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        save_checkpoint(checkpoint, filename)
        if "best" in filename:
            print(f"         💎 Best model saved: {filename}")
        elif "final" in filename:
            pass  # Already printed in main loop
        else:
            print(f"         📁 Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = load_checkpoint(filename, self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train MiniGPT model")
    parser.add_argument("--config", type=str, default="configs/small.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")

    args = parser.parse_args()

    trainer = Trainer(args.config)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed training from {args.resume}")

    trainer.train()


if __name__ == "__main__":
    main()