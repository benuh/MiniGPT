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

        # Initialize tokenizer
        self.tokenizer = get_tokenizer("gpt2")

        # Initialize model
        self.model = MiniGPT(**self.config['model']).to(self.device)
        print(f"Model has {self.model.count_parameters():,} parameters")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Initialize data loaders
        self.train_loader, self.val_loader = self._prepare_data()

        # Initialize wandb logging
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['run_name'],
                config=self.config
            )

        self.global_step = 0
        self.best_val_loss = float('inf')

    def _prepare_data(self):
        """Prepare training and validation data loaders"""
        data_config = self.config['data']

        # Load dataset
        if data_config['dataset_name'] == "wikitext":
            dataset = load_dataset("wikitext", data_config['dataset_config'])
        else:
            raise ValueError(f"Unsupported dataset: {data_config['dataset_name']}")

        # Process texts
        train_texts = [item['text'] for item in dataset[data_config['train_split']] if item['text'].strip()]
        val_texts = [item['text'] for item in dataset[data_config['val_split']] if item['text'].strip()]

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

        return train_loader, val_loader

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc="Training")

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
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
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
                val_loss = self.validate()
                self.model.train()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')

            # Save checkpoint
            if self.global_step % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_{self.global_step}.pt')

        return total_loss / num_batches

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for input_ids, targets in tqdm(self.val_loader, desc="Validation"):
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                logits, loss = self.model(input_ids, targets)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches

        # Log to wandb
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.log({
                'val_loss': avg_loss,
                'global_step': self.global_step
            })

        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        """Main training loop"""
        print("Starting training...")

        for epoch in range(self.config['training']['max_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['max_epochs']}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        print("Training completed!")
        self.save_checkpoint('final_model.pt')

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