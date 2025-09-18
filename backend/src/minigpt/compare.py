"""
Model comparison and A/B testing utilities for MiniGPT
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from pathlib import Path
import seaborn as sns

from .model import MiniGPT
from .evaluate import ModelEvaluator
from .tokenizer import get_tokenizer
from .utils import get_device, load_checkpoint


class ModelComparator:
    """
    Compare multiple MiniGPT models across various metrics
    """

    def __init__(self, device=None):
        self.device = device or get_device()
        self.tokenizer = get_tokenizer("gpt2")
        self.models = {}
        self.results = {}

    def add_model(self, name: str, checkpoint_path: str):
        """Add a model for comparison"""
        try:
            # Load model
            checkpoint = load_checkpoint(checkpoint_path, self.device)
            config = checkpoint.get('config', {})
            model_config = config.get('model', {})

            model = MiniGPT(**model_config).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.models[name] = {
                'model': model,
                'config': config,
                'checkpoint_path': checkpoint_path,
                'parameters': model.count_parameters()
            }

            print(f"✅ Added model '{name}': {model.count_parameters():,} parameters")

        except Exception as e:
            print(f"❌ Failed to load model '{name}': {str(e)}")

    def compare_perplexity(self, test_texts: List[str]) -> Dict[str, float]:
        """Compare models on perplexity"""
        print("📊 Comparing perplexity...")

        perplexity_results = {}

        for name, model_info in self.models.items():
            evaluator = ModelEvaluator(model_info['model'], self.tokenizer, self.device)
            perplexity = evaluator.calculate_perplexity(test_texts)
            perplexity_results[name] = perplexity

            print(f"  {name}: {perplexity:.2f}")

        return perplexity_results

    def compare_generation_quality(self, prompts: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare models on generation quality"""
        print("✍️ Comparing generation quality...")

        generation_results = {}

        for name, model_info in self.models.items():
            evaluator = ModelEvaluator(model_info['model'], self.tokenizer, self.device)
            results = evaluator.generate_and_evaluate(prompts)
            generation_results[name] = results

            print(f"  {name}: avg_length={results['avg_length']:.1f}, "
                  f"unique_tokens={results['unique_token_count']}, "
                  f"repetition={results['avg_repetition']:.3f}")

        return generation_results

    def benchmark_inference_speed(self, batch_sizes: List[int] = [1, 4, 8]) -> Dict[str, Dict[str, Any]]:
        """Benchmark inference speed for all models"""
        print("⏱️ Benchmarking inference speed...")

        benchmark_results = {}

        for name, model_info in self.models.items():
            print(f"  Benchmarking {name}...")

            # Create dummy input
            dummy_input = torch.randint(0, model_info['model'].vocab_size,
                                      (max(batch_sizes), 128), device=self.device)

            model_benchmarks = {}

            for batch_size in batch_sizes:
                input_batch = dummy_input[:batch_size]

                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model_info['model'](input_batch)

                # Benchmark
                times = []
                for _ in range(20):
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None

                    import time
                    start = time.time()

                    with torch.no_grad():
                        _ = model_info['model'](input_batch)

                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end = time.time()

                    times.append(end - start)

                avg_time = np.mean(times)
                throughput = batch_size * 128 / avg_time  # tokens per second

                model_benchmarks[batch_size] = {
                    'avg_time_ms': avg_time * 1000,
                    'throughput_tokens_per_sec': throughput
                }

            benchmark_results[name] = model_benchmarks

        return benchmark_results

    def comprehensive_comparison(self, test_texts: List[str], prompts: List[str]) -> Dict[str, Any]:
        """Run comprehensive comparison across all metrics"""
        print("🔍 Running comprehensive model comparison...")

        results = {
            'model_info': {},
            'perplexity': {},
            'generation': {},
            'benchmarks': {},
            'summary': {}
        }

        # Collect model info
        for name, model_info in self.models.items():
            results['model_info'][name] = {
                'parameters': model_info['parameters'],
                'vocab_size': model_info['model'].vocab_size,
                'context_length': model_info['model'].block_size,
                'n_layers': len(model_info['model'].blocks),
                'checkpoint_path': model_info['checkpoint_path']
            }

        # Run comparisons
        results['perplexity'] = self.compare_perplexity(test_texts)
        results['generation'] = self.compare_generation_quality(prompts)
        results['benchmarks'] = self.benchmark_inference_speed()

        # Create summary
        for name in self.models.keys():
            results['summary'][name] = {
                'perplexity': results['perplexity'].get(name, float('inf')),
                'avg_generation_length': results['generation'].get(name, {}).get('avg_length', 0),
                'unique_tokens': results['generation'].get(name, {}).get('unique_token_count', 0),
                'throughput_1batch': results['benchmarks'].get(name, {}).get(1, {}).get('throughput_tokens_per_sec', 0),
                'parameters': results['model_info'][name]['parameters']
            }

        self.results = results
        return results

    def create_comparison_plots(self, output_dir: str = "comparison_plots"):
        """Create visualization plots for model comparison"""
        if not self.results:
            print("❌ No comparison results available. Run comprehensive_comparison first.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        model_names = list(self.models.keys())

        # 1. Perplexity comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        perplexities = [self.results['perplexity'][name] for name in model_names]
        bars = ax.bar(model_names, perplexities)
        ax.set_title('Model Perplexity Comparison (Lower is Better)')
        ax.set_ylabel('Perplexity')
        ax.set_xlabel('Model')

        # Add value labels on bars
        for bar, perp in zip(bars, perplexities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{perp:.2f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'perplexity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Inference throughput comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        throughputs = [self.results['summary'][name]['throughput_1batch'] for name in model_names]
        parameters = [self.results['summary'][name]['parameters'] for name in model_names]

        scatter = ax.scatter(parameters, throughputs, s=100, alpha=0.7)

        for i, name in enumerate(model_names):
            ax.annotate(name, (parameters[i], throughputs[i]),
                       xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Model Parameters')
        ax.set_ylabel('Throughput (tokens/sec)')
        ax.set_title('Model Size vs Inference Speed')
        ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(output_path / 'throughput_vs_size.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Generation quality metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Average generation length
        gen_lengths = [self.results['summary'][name]['avg_generation_length'] for name in model_names]
        ax1.bar(model_names, gen_lengths)
        ax1.set_title('Average Generation Length')
        ax1.set_ylabel('Tokens')
        ax1.tick_params(axis='x', rotation=45)

        # Unique tokens
        unique_tokens = [self.results['summary'][name]['unique_tokens'] for name in model_names]
        ax2.bar(model_names, unique_tokens)
        ax2.set_title('Unique Tokens Used')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)

        # Parameters comparison
        ax3.bar(model_names, parameters)
        ax3.set_title('Model Size (Parameters)')
        ax3.set_ylabel('Parameters')
        ax3.set_yscale('log')
        ax3.tick_params(axis='x', rotation=45)

        # Multi-metric radar plot (simplified)
        metrics = ['perplexity_norm', 'throughput_norm', 'unique_tokens_norm']

        # Normalize metrics (invert perplexity so higher is better)
        max_perp = max(perplexities)
        perp_norm = [(max_perp - p) / max_perp for p in perplexities]

        max_throughput = max(throughputs)
        throughput_norm = [t / max_throughput for t in throughputs]

        max_unique = max(unique_tokens)
        unique_norm = [u / max_unique for u in unique_tokens]

        x = np.arange(len(model_names))
        width = 0.25

        ax4.bar(x - width, perp_norm, width, label='Quality (1-norm_perplexity)', alpha=0.8)
        ax4.bar(x, throughput_norm, width, label='Speed (norm_throughput)', alpha=0.8)
        ax4.bar(x + width, unique_norm, width, label='Diversity (norm_unique_tokens)', alpha=0.8)

        ax4.set_xlabel('Model')
        ax4.set_ylabel('Normalized Score')
        ax4.set_title('Multi-Metric Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(output_path / 'generation_quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Comparison plots saved to {output_path}/")

    def save_comparison_report(self, filepath: str = "model_comparison_report.json"):
        """Save detailed comparison report"""
        if not self.results:
            print("❌ No comparison results available. Run comprehensive_comparison first.")
            return

        # Add metadata
        report = {
            'comparison_timestamp': __import__('datetime').datetime.now().isoformat(),
            'device': str(self.device),
            'models_compared': list(self.models.keys()),
            'results': self.results
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Comparison report saved to {filepath}")

    def get_best_model(self, metric: str = 'perplexity') -> str:
        """Get the best performing model based on a metric"""
        if not self.results:
            print("❌ No comparison results available.")
            return None

        if metric == 'perplexity':
            scores = self.results['perplexity']
            best = min(scores, key=scores.get)
        elif metric == 'throughput':
            scores = {name: data['throughput_1batch'] for name, data in self.results['summary'].items()}
            best = max(scores, key=scores.get)
        elif metric == 'diversity':
            scores = {name: data['unique_tokens'] for name, data in self.results['summary'].items()}
            best = max(scores, key=scores.get)
        else:
            print(f"❌ Unknown metric: {metric}")
            return None

        return best


def main():
    """Command-line model comparison tool"""
    import argparse

    parser = argparse.ArgumentParser(description="Compare MiniGPT models")
    parser.add_argument("--models", type=str, nargs='+', required=True,
                       help="Model checkpoint files to compare")
    parser.add_argument("--names", type=str, nargs='+',
                       help="Custom names for models (optional)")
    parser.add_argument("--test-file", type=str,
                       help="File with test texts for perplexity")
    parser.add_argument("--prompts-file", type=str,
                       help="File with prompts for generation testing")
    parser.add_argument("--output", type=str, default="comparison_report.json",
                       help="Output report file")
    parser.add_argument("--plots", action="store_true",
                       help="Generate comparison plots")

    args = parser.parse_args()

    # Initialize comparator
    comparator = ModelComparator()

    # Add models
    model_names = args.names if args.names else [f"model_{i+1}" for i in range(len(args.models))]

    for name, checkpoint in zip(model_names, args.models):
        comparator.add_model(name, checkpoint)

    if not comparator.models:
        print("❌ No models loaded successfully")
        return

    # Load test data
    test_texts = ["The quick brown fox jumps over the lazy dog."] * 10  # Default
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_texts = [line.strip() for line in f.readlines() if line.strip()]

    prompts = ["Hello", "The future of AI", "Once upon a time"]  # Default
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]

    # Run comparison
    results = comparator.comprehensive_comparison(test_texts, prompts[:10])  # Limit prompts

    # Save report
    comparator.save_comparison_report(args.output)

    # Generate plots if requested
    if args.plots:
        comparator.create_comparison_plots()

    # Print summary
    print("\n🏆 Model Ranking Summary:")
    print("-" * 40)

    best_perplexity = comparator.get_best_model('perplexity')
    best_throughput = comparator.get_best_model('throughput')
    best_diversity = comparator.get_best_model('diversity')

    print(f"Best Quality (lowest perplexity): {best_perplexity}")
    print(f"Fastest Inference: {best_throughput}")
    print(f"Most Diverse Generation: {best_diversity}")


if __name__ == "__main__":
    main()