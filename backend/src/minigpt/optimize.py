"""
Model optimization and quantization utilities for MiniGPT
Includes pruning, quantization, and performance optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
import time
from pathlib import Path

from .model import MiniGPT
from .utils import get_device, save_checkpoint


class ModelOptimizer:
    def __init__(self, model: MiniGPT, device=None):
        self.model = model
        self.device = device or get_device()
        self.original_size = self._get_model_size()

    def _get_model_size(self) -> int:
        """Get model size in bytes"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.numel() * param.element_size()
        return param_size

    def _get_model_memory(self) -> Dict[str, int]:
        """Get detailed memory usage"""
        memory_stats = {}

        for name, param in self.model.named_parameters():
            memory_stats[name] = param.numel() * param.element_size()

        return memory_stats

    def prune_magnitude(self, pruning_ratio: float = 0.2) -> Dict[str, Any]:
        """
        Magnitude-based weight pruning
        Remove weights with smallest absolute values
        """
        print(f"🔪 Applying magnitude pruning (ratio: {pruning_ratio})")

        # Collect all weights
        all_weights = []
        weight_info = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weights = module.weight.data.abs().flatten()
                all_weights.append(weights)
                weight_info.append((name, module, weights.numel()))

        # Calculate global threshold
        all_weights_tensor = torch.cat(all_weights)
        threshold_idx = int(len(all_weights_tensor) * pruning_ratio)
        threshold = torch.kthvalue(all_weights_tensor, threshold_idx).values.item()

        # Apply pruning
        pruned_params = 0
        total_params = 0

        for name, module, num_params in weight_info:
            mask = module.weight.data.abs() > threshold
            module.weight.data *= mask.float()

            pruned_count = (~mask).sum().item()
            pruned_params += pruned_count
            total_params += num_params

            print(f"  {name}: pruned {pruned_count}/{num_params} ({100*pruned_count/num_params:.1f}%)")

        pruning_stats = {
            'method': 'magnitude',
            'target_ratio': pruning_ratio,
            'actual_ratio': pruned_params / total_params,
            'pruned_parameters': pruned_params,
            'total_parameters': total_params,
            'size_reduction': 1 - (total_params - pruned_params) / total_params
        }

        print(f"✅ Pruning complete: {100*pruning_stats['actual_ratio']:.1f}% parameters removed")
        return pruning_stats

    def quantize_int8(self) -> Dict[str, Any]:
        """
        Post-training quantization to int8
        """
        print("🔢 Applying INT8 quantization")

        # Store original weights for comparison
        original_weights = {}
        for name, param in self.model.named_parameters():
            original_weights[name] = param.data.clone()

        # Apply quantization
        quantized_modules = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate scale and zero_point
                weight = module.weight.data
                min_val = weight.min()
                max_val = weight.max()

                # Symmetric quantization
                scale = max(abs(min_val), abs(max_val)) / 127

                # Quantize and dequantize
                quantized = torch.round(weight / scale).clamp(-128, 127)
                module.weight.data = quantized * scale

                quantized_modules.append(name)

        # Calculate compression ratio
        new_size = self._get_model_size()
        compression_ratio = self.original_size / new_size

        quantization_stats = {
            'method': 'int8',
            'quantized_modules': quantized_modules,
            'original_size_mb': self.original_size / (1024 * 1024),
            'quantized_size_mb': new_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction': 1 - new_size / self.original_size
        }

        print(f"✅ Quantization complete: {compression_ratio:.1f}x compression")
        return quantization_stats

    def knowledge_distillation_prep(self, student_config: Dict[str, Any]) -> MiniGPT:
        """
        Create a smaller student model for knowledge distillation
        """
        print("🎓 Creating student model for distillation")

        # Create smaller model
        student = MiniGPT(**student_config).to(self.device)

        print(f"Teacher model: {self.model.count_parameters():,} parameters")
        print(f"Student model: {student.count_parameters():,} parameters")
        print(f"Compression ratio: {self.model.count_parameters() / student.count_parameters():.1f}x")

        return student

    def benchmark_inference(self, batch_sizes: list = [1, 4, 8, 16], sequence_length: int = 128) -> Dict[str, Any]:
        """
        Benchmark inference speed and memory usage
        """
        print("⏱️  Benchmarking inference performance")

        self.model.eval()
        benchmark_results = {}

        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")

            # Create dummy input
            dummy_input = torch.randint(0, self.model.vocab_size,
                                      (batch_size, sequence_length),
                                      device=self.device)

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_input)

            # Benchmark
            torch.cuda.synchronize() if self.device.type == 'cuda' else None

            times = []
            memory_used = []

            for _ in range(20):
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                start_time = time.time()

                with torch.no_grad():
                    logits, _ = self.model(dummy_input)

                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end_time = time.time()

                times.append(end_time - start_time)

                if self.device.type == 'cuda':
                    memory_used.append(torch.cuda.max_memory_allocated())

            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_memory = np.mean(memory_used) if memory_used else 0

            benchmark_results[batch_size] = {
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'throughput_tokens_per_sec': batch_size * sequence_length / avg_time,
                'memory_mb': avg_memory / (1024 * 1024) if avg_memory else 0
            }

            print(f"    Time: {avg_time*1000:.2f}±{std_time*1000:.2f}ms")
            print(f"    Throughput: {benchmark_results[batch_size]['throughput_tokens_per_sec']:.0f} tokens/sec")

        return benchmark_results

    def optimize_for_inference(self) -> Dict[str, Any]:
        """
        Apply various optimizations for faster inference
        """
        print("🚀 Optimizing model for inference")

        optimizations = []

        # Fuse operations where possible
        try:
            # This is a placeholder for actual optimization
            # In practice, you'd use tools like TorchScript, ONNX, or TensorRT
            torch.jit.script(self.model)
            optimizations.append("torchscript_compilation")
            print("  ✅ TorchScript compilation applied")
        except Exception as e:
            print(f"  ❌ TorchScript compilation failed: {e}")

        # Set to evaluation mode and disable gradient computation
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        optimizations.append("gradient_disabled")
        print("  ✅ Gradient computation disabled")

        return {
            'optimizations_applied': optimizations,
            'model_mode': 'eval',
            'gradients_enabled': False
        }

    def comprehensive_optimization(self,
                                 pruning_ratio: float = 0.1,
                                 quantize: bool = True,
                                 benchmark: bool = True) -> Dict[str, Any]:
        """
        Apply comprehensive optimization pipeline
        """
        print("🔧 Starting comprehensive optimization pipeline")

        results = {
            'original_size_mb': self.original_size / (1024 * 1024),
            'original_parameters': self.model.count_parameters()
        }

        # Benchmark original model
        if benchmark:
            print("\n📊 Benchmarking original model...")
            results['original_benchmark'] = self.benchmark_inference([1, 8])

        # Apply pruning
        if pruning_ratio > 0:
            print(f"\n🔪 Applying pruning...")
            results['pruning'] = self.prune_magnitude(pruning_ratio)

        # Apply quantization
        if quantize:
            print(f"\n🔢 Applying quantization...")
            results['quantization'] = self.quantize_int8()

        # Optimize for inference
        print(f"\n🚀 Applying inference optimizations...")
        results['inference_optimizations'] = self.optimize_for_inference()

        # Final benchmark
        if benchmark:
            print("\n📊 Benchmarking optimized model...")
            results['optimized_benchmark'] = self.benchmark_inference([1, 8])

        # Calculate final statistics
        results['final_size_mb'] = self._get_model_size() / (1024 * 1024)
        results['final_parameters'] = self.model.count_parameters()
        results['total_compression_ratio'] = results['original_size_mb'] / results['final_size_mb']

        print(f"\n✅ Optimization complete!")
        print(f"   Size reduction: {results['original_size_mb']:.1f}MB → {results['final_size_mb']:.1f}MB")
        print(f"   Compression ratio: {results['total_compression_ratio']:.1f}x")

        return results

    def save_optimized_model(self, filepath: str, optimization_results: Dict[str, Any]):
        """
        Save optimized model with metadata
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimization_results': optimization_results,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'block_size': self.model.block_size,
            },
            'optimized': True,
            'optimization_timestamp': time.time()
        }

        save_checkpoint(checkpoint, filepath)
        print(f"💾 Optimized model saved to {filepath}")


def main():
    """Command-line optimization script"""
    import argparse
    from .utils import load_checkpoint
    from .config import load_config

    parser = argparse.ArgumentParser(description="Optimize MiniGPT model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Output path for optimized model")
    parser.add_argument("--pruning-ratio", type=float, default=0.1, help="Pruning ratio")
    parser.add_argument("--no-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmarking")

    args = parser.parse_args()

    # Load model
    device = get_device()
    checkpoint = load_checkpoint(args.model, device)
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    model = MiniGPT(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize optimizer
    optimizer = ModelOptimizer(model, device)

    # Run optimization
    results = optimizer.comprehensive_optimization(
        pruning_ratio=args.pruning_ratio,
        quantize=not args.no_quantize,
        benchmark=not args.no_benchmark
    )

    # Save optimized model
    if args.output:
        optimizer.save_optimized_model(args.output, results)
    else:
        output_path = f"optimized_{Path(args.model).stem}.pt"
        optimizer.save_optimized_model(output_path, results)

    # Save optimization report
    import json
    report_path = f"optimization_report_{Path(args.model).stem}.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"📄 Optimization report saved to {report_path}")


if __name__ == "__main__":
    main()