#!/usr/bin/env python3
"""
Quick evaluation script for trained models
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minigpt.evaluate import ModelEvaluator
from minigpt.model import MiniGPT
from minigpt.tokenizer import get_tokenizer
from minigpt.utils import load_checkpoint, get_device


def create_sample_prompts():
    """Create sample prompts for evaluation"""
    return [
        "The quick brown fox",
        "In a galaxy far, far away",
        "Once upon a time",
        "The future of artificial intelligence",
        "Climate change is",
        "The most important invention",
        "Life is like",
        "Technology has changed",
        "Education in the 21st century",
        "The key to happiness"
    ]


def create_sample_test_texts():
    """Create sample test texts for perplexity calculation"""
    return [
        "The cat sat on the mat and looked around the room carefully.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "The weather today is sunny with a chance of rain in the afternoon.",
        "Books have been a source of knowledge and entertainment for centuries.",
        "Cooking is both an art and a science that brings people together.",
        "Space exploration continues to reveal new mysteries about our universe.",
        "Music has the power to evoke emotions and memories from our past.",
        "The internet has revolutionized how we communicate and share information.",
        "Exercise is important for maintaining both physical and mental health.",
        "Travel broadens our perspective and helps us understand different cultures."
    ]


def main():
    print("🔍 MiniGPT Model Evaluation")
    print("=" * 40)

    # Check for available models
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("❌ No checkpoints directory found")
        print("   Please train a model first using: python -m minigpt.train")
        return

    # Find available checkpoints
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        print("❌ No model checkpoints found")
        print("   Please train a model first using: python -m minigpt.train")
        return

    # Use the most recent checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    print(f"📂 Using checkpoint: {latest_checkpoint.name}")

    try:
        # Load model
        device = get_device()
        print(f"🔧 Using device: {device}")

        checkpoint = load_checkpoint(latest_checkpoint.name, device)
        config = checkpoint.get('config', {})
        model_config = config.get('model', {})

        model = MiniGPT(**model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"🤖 Model loaded: {model.count_parameters():,} parameters")

        # Load tokenizer
        tokenizer = get_tokenizer("gpt2")

        # Initialize evaluator
        evaluator = ModelEvaluator(model, tokenizer, device)

        # Run evaluation
        test_texts = create_sample_test_texts()
        prompts = create_sample_prompts()

        print("\n📊 Running evaluation...")
        results = evaluator.comprehensive_evaluation(test_texts, prompts)

        # Display results
        print("\n" + "=" * 50)
        print("📈 EVALUATION RESULTS")
        print("=" * 50)

        print(f"📊 Perplexity: {results.get('perplexity', 'N/A'):.2f}")

        if 'generation' in results:
            gen = results['generation']
            print(f"📝 Average generation length: {gen.get('avg_length', 0):.1f} tokens")
            print(f"🔤 Unique tokens used: {gen.get('unique_token_count', 0)}")
            print(f"🔄 Repetition score: {gen.get('avg_repetition', 0):.3f} (lower is better)")

        print(f"\n🏗️  Model Architecture:")
        stats = results.get('model_stats', {})
        print(f"   • Parameters: {stats.get('parameters', 'N/A'):,}")
        print(f"   • Layers: {stats.get('n_layers', 'N/A')}")
        print(f"   • Attention heads: {stats.get('n_heads', 'N/A')}")
        print(f"   • Embedding dimension: {stats.get('n_embd', 'N/A')}")
        print(f"   • Context length: {stats.get('block_size', 'N/A')}")

        # Show some sample generations
        if 'generation' in results:
            print(f"\n💬 Sample Generations:")
            print("-" * 50)
            for i, gen in enumerate(results['generation']['generations'][:3]):
                print(f"{i+1}. Prompt: \"{gen['prompt']}\"")
                print(f"   Response: \"{gen['response'][:100]}{'...' if len(gen['response']) > 100 else ''}\"")
                print()

        # Save detailed report
        report_file = f"evaluation_report_{latest_checkpoint.stem}.json"
        evaluator.save_evaluation_report(results, report_file)
        print(f"📄 Detailed report saved to: {report_file}")

    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()