#!/usr/bin/env python3
"""
Automated training script that continuously improves MiniGPT models
This script handles:
- Progressive model size scaling
- Automatic hyperparameter optimization
- Checkpoint management
- Git integration for version control
- Resilient training with error recovery
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minigpt.config import load_config, save_config, get_default_config
from minigpt.train import Trainer
from minigpt.utils import get_device


class AutoTrainer:
    def __init__(self, base_config_path="configs/small.yaml"):
        self.base_config_path = base_config_path
        self.session_log = []
        self.improvement_history = []
        self.device = get_device()

        # Create directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("experiments", exist_ok=True)

        print(f"🚀 AutoTrainer initialized on {self.device}")

    def log_event(self, event, details=None):
        """Log events with timestamps"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event": event,
            "details": details or {}
        }
        self.session_log.append(log_entry)
        print(f"[{timestamp}] {event}")

        # Save to file
        with open("logs/auto_trainer.log", "a") as f:
            f.write(f"{timestamp}: {event}\n")
            if details:
                f.write(f"  Details: {json.dumps(details, indent=2)}\n")

    def get_next_experiment_name(self):
        """Generate unique experiment name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"

    def create_progressive_config(self, base_config, scale_factor=1.2):
        """Create a slightly larger model configuration"""
        config = base_config.copy()

        # Scale model size
        config['model']['n_layer'] = min(12, int(config['model']['n_layer'] * scale_factor))
        config['model']['n_embd'] = min(512, int(config['model']['n_embd'] * scale_factor))
        config['model']['n_head'] = min(8, max(4, config['model']['n_head']))

        # Adjust training for larger model
        if config['model']['n_embd'] > 256:
            config['training']['batch_size'] = max(8, config['training']['batch_size'] // 2)
            config['training']['learning_rate'] *= 0.8

        return config

    def run_training_experiment(self, config, experiment_name):
        """Run a single training experiment"""
        try:
            # Save experiment config
            exp_dir = f"experiments/{experiment_name}"
            os.makedirs(exp_dir, exist_ok=True)
            config_path = f"{exp_dir}/config.yaml"
            save_config(config, config_path)

            self.log_event(f"Starting experiment {experiment_name}", {
                "model_params": config['model'],
                "training_params": config['training']
            })

            # Initialize trainer
            trainer = Trainer(config_path)

            # Train model
            trainer.train()

            # Get final validation loss
            final_val_loss = trainer.best_val_loss

            self.log_event(f"Experiment {experiment_name} completed", {
                "final_val_loss": final_val_loss,
                "model_params": trainer.model.count_parameters()
            })

            return {
                "experiment_name": experiment_name,
                "config": config,
                "final_val_loss": final_val_loss,
                "model_params": trainer.model.count_parameters(),
                "success": True
            }

        except Exception as e:
            self.log_event(f"Experiment {experiment_name} failed", {
                "error": str(e)
            })
            return {
                "experiment_name": experiment_name,
                "config": config,
                "success": False,
                "error": str(e)
            }

    def optimize_hyperparameters(self, base_config):
        """Simple hyperparameter optimization"""
        best_config = base_config.copy()
        best_loss = float('inf')

        # Try different learning rates
        learning_rates = [1e-4, 3e-4, 5e-4]

        for lr in learning_rates:
            config = base_config.copy()
            config['training']['learning_rate'] = lr
            config['training']['max_epochs'] = 3  # Quick evaluation

            exp_name = f"hp_lr_{lr}_{self.get_next_experiment_name()}"
            result = self.run_training_experiment(config, exp_name)

            if result['success'] and result['final_val_loss'] < best_loss:
                best_loss = result['final_val_loss']
                best_config = config

        return best_config

    def commit_and_push_changes(self, message):
        """Commit changes to git and push to remote"""
        try:
            # Add all changes
            subprocess.run(["git", "add", "."], check=True)

            # Commit with message
            commit_msg = f"{message}\n\n🤖 Generated with AutoTrainer\nTimestamp: {datetime.now().isoformat()}"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)

            # Push to remote
            subprocess.run(["git", "push"], check=True)

            self.log_event("Changes committed and pushed to git", {"message": message})
            return True

        except subprocess.CalledProcessError as e:
            self.log_event("Git operation failed", {"error": str(e)})
            return False

    def run_continuous_improvement(self, max_iterations=10, improvement_threshold=0.1):
        """Main loop for continuous improvement"""
        self.log_event("Starting continuous improvement cycle")

        # Load base configuration
        if os.path.exists(self.base_config_path):
            current_config = load_config(self.base_config_path)
        else:
            current_config = get_default_config()
            save_config(current_config, self.base_config_path)

        best_loss = float('inf')

        for iteration in range(max_iterations):
            self.log_event(f"Starting improvement iteration {iteration + 1}/{max_iterations}")

            # 1. Optimize hyperparameters
            optimized_config = self.optimize_hyperparameters(current_config)

            # 2. Train with optimized config
            exp_name = f"main_{self.get_next_experiment_name()}"
            result = self.run_training_experiment(optimized_config, exp_name)

            if result['success']:
                # 3. Check if this is an improvement
                if result['final_val_loss'] < best_loss - improvement_threshold:
                    best_loss = result['final_val_loss']
                    current_config = result['config']

                    # Save improved config
                    save_config(current_config, self.base_config_path)

                    # Commit improvements
                    commit_msg = f"Improve model performance - Val Loss: {best_loss:.4f}"
                    self.commit_and_push_changes(commit_msg)

                    self.log_event("Model improved and saved", {
                        "new_val_loss": best_loss,
                        "improvement": True
                    })
                else:
                    self.log_event("No significant improvement this iteration")

                # 4. Try scaling up the model for next iteration
                current_config = self.create_progressive_config(current_config)

            # 5. Save progress
            self.save_session_state()

            # Small delay between iterations
            time.sleep(60)

        self.log_event("Continuous improvement cycle completed")

    def save_session_state(self):
        """Save current session state for recovery"""
        state = {
            "session_log": self.session_log,
            "improvement_history": self.improvement_history,
            "timestamp": datetime.now().isoformat()
        }

        with open("logs/session_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def load_session_state(self):
        """Load previous session state for recovery"""
        try:
            with open("logs/session_state.json", "r") as f:
                state = json.load(f)

            self.session_log = state.get("session_log", [])
            self.improvement_history = state.get("improvement_history", [])

            self.log_event("Session state recovered")
            return True

        except FileNotFoundError:
            self.log_event("No previous session state found")
            return False


def main():
    parser = argparse.ArgumentParser(description="Automated MiniGPT training")
    parser.add_argument("--config", type=str, default="configs/small.yaml",
                       help="Base config file")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Maximum training iterations")
    parser.add_argument("--recover", action="store_true",
                       help="Recover from previous session")

    args = parser.parse_args()

    # Initialize auto trainer
    auto_trainer = AutoTrainer(args.config)

    # Recover previous session if requested
    if args.recover:
        auto_trainer.load_session_state()

    try:
        # Run continuous improvement
        auto_trainer.run_continuous_improvement(max_iterations=args.iterations)

    except KeyboardInterrupt:
        auto_trainer.log_event("Training interrupted by user")
        auto_trainer.save_session_state()

    except Exception as e:
        auto_trainer.log_event(f"Unexpected error: {str(e)}")
        auto_trainer.save_session_state()
        raise


if __name__ == "__main__":
    main()