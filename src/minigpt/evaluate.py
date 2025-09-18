"""
Evaluation metrics and utilities for MiniGPT models
"""

import math
import torch
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

from .model import MiniGPT
from .tokenizer import get_tokenizer
from .utils import get_device


class ModelEvaluator:
    def __init__(self, model: MiniGPT, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or get_device()
        self.model.eval()

    def calculate_perplexity(self, texts: List[str], batch_size: int = 8) -> float:
        """Calculate perplexity on a list of texts"""
        total_log_likelihood = 0
        total_tokens = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
                batch_texts = texts[i:i + batch_size]
                batch_tokens = []

                for text in batch_texts:
                    tokens = self.tokenizer.encode(text)
                    if len(tokens) > self.model.block_size:
                        tokens = tokens[:self.model.block_size]
                    batch_tokens.append(tokens)

                # Pad sequences
                max_len = max(len(tokens) for tokens in batch_tokens)
                padded_tokens = []
                for tokens in batch_tokens:
                    padded = tokens + [0] * (max_len - len(tokens))
                    padded_tokens.append(padded)

                input_ids = torch.tensor(padded_tokens, device=self.device)

                # Calculate loss for each sequence
                for j, tokens in enumerate(batch_tokens):
                    if len(tokens) < 2:
                        continue

                    seq_input = input_ids[j:j+1, :len(tokens)-1]
                    seq_target = input_ids[j:j+1, 1:len(tokens)]

                    logits, loss = self.model(seq_input, seq_target)

                    total_log_likelihood += loss.item() * (len(tokens) - 1)
                    total_tokens += len(tokens) - 1

        if total_tokens == 0:
            return float('inf')

        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = math.exp(avg_log_likelihood)
        return perplexity

    def calculate_bleu_score(self, references: List[str], hypotheses: List[str]) -> float:
        """Calculate BLEU score (simplified version)"""
        def get_ngrams(text: str, n: int) -> Dict[str, int]:
            words = text.split()
            ngrams = defaultdict(int)
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                ngrams[ngram] += 1
            return dict(ngrams)

        def calculate_precision(ref_ngrams: Dict[str, int], hyp_ngrams: Dict[str, int]) -> float:
            if not hyp_ngrams:
                return 0.0

            overlap = 0
            total = 0
            for ngram, count in hyp_ngrams.items():
                overlap += min(count, ref_ngrams.get(ngram, 0))
                total += count

            return overlap / total if total > 0 else 0.0

        # Calculate precision for n-grams (n=1 to 4)
        precisions = []
        for n in range(1, 5):
            total_precision = 0
            for ref, hyp in zip(references, hypotheses):
                ref_ngrams = get_ngrams(ref.lower(), n)
                hyp_ngrams = get_ngrams(hyp.lower(), n)
                precision = calculate_precision(ref_ngrams, hyp_ngrams)
                total_precision += precision

            avg_precision = total_precision / len(references) if references else 0
            precisions.append(avg_precision)

        # Geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0

        geometric_mean = np.exp(np.mean(np.log(precisions)))

        # Brevity penalty
        ref_len = sum(len(ref.split()) for ref in references)
        hyp_len = sum(len(hyp.split()) for hyp in hypotheses)

        if hyp_len >= ref_len:
            bp = 1.0
        else:
            bp = np.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0

        bleu = bp * geometric_mean
        return bleu

    def generate_and_evaluate(self, prompts: List[str], max_length: int = 50) -> Dict[str, Any]:
        """Generate text for prompts and calculate basic metrics"""
        results = {
            'generations': [],
            'avg_length': 0,
            'unique_tokens': set(),
            'repetition_score': 0
        }

        total_length = 0
        repetition_scores = []

        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating text"):
                # Encode prompt
                input_ids = self.tokenizer.encode(prompt)
                input_tensor = torch.tensor([input_ids], device=self.device)

                # Generate
                generated = self.model.generate(
                    input_tensor,
                    max_new_tokens=max_length,
                    temperature=0.8,
                    top_k=50
                )

                # Decode
                generated_text = self.tokenizer.decode(generated[0].tolist())
                response = generated_text[len(prompt):].strip()

                results['generations'].append({
                    'prompt': prompt,
                    'response': response,
                    'full_text': generated_text
                })

                # Calculate metrics
                tokens = response.split()
                total_length += len(tokens)
                results['unique_tokens'].update(tokens)

                # Repetition score (simple n-gram repetition)
                if len(tokens) > 3:
                    bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
                    unique_bigrams = set(bigrams)
                    repetition = 1 - len(unique_bigrams) / len(bigrams) if bigrams else 0
                    repetition_scores.append(repetition)

        results['avg_length'] = total_length / len(prompts) if prompts else 0
        results['unique_token_count'] = len(results['unique_tokens'])
        results['avg_repetition'] = np.mean(repetition_scores) if repetition_scores else 0

        return results

    def comprehensive_evaluation(self, test_texts: List[str], prompts: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        print("🔍 Running comprehensive model evaluation...")

        results = {}

        # Perplexity
        print("📊 Calculating perplexity...")
        perplexity = self.calculate_perplexity(test_texts[:100])  # Limit for speed
        results['perplexity'] = perplexity

        # Generation evaluation
        if prompts:
            print("✍️ Evaluating text generation...")
            generation_results = self.generate_and_evaluate(prompts[:20])
            results['generation'] = generation_results

        # Model statistics
        print("📈 Gathering model statistics...")
        results['model_stats'] = {
            'parameters': self.model.count_parameters(),
            'vocab_size': self.model.vocab_size,
            'block_size': self.model.block_size,
            'n_layers': len(self.model.blocks),
            'n_heads': self.model.blocks[0].attn.n_head if self.model.blocks else 0,
            'n_embd': self.model.blocks[0].attn.n_embd if self.model.blocks else 0
        }

        return results

    def save_evaluation_report(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file"""
        import json
        from datetime import datetime

        report = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': results,
            'model_info': results.get('model_stats', {}),
            'summary': {
                'perplexity': results.get('perplexity', 'N/A'),
                'avg_generation_length': results.get('generation', {}).get('avg_length', 'N/A'),
                'unique_tokens': results.get('generation', {}).get('unique_token_count', 'N/A'),
                'repetition_score': results.get('generation', {}).get('avg_repetition', 'N/A')
            }
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📋 Evaluation report saved to {filepath}")


def main():
    """Command-line evaluation script"""
    import argparse
    from .utils import load_checkpoint
    from .config import load_config

    parser = argparse.ArgumentParser(description="Evaluate MiniGPT model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--test-file", type=str, help="Path to test text file")
    parser.add_argument("--prompts-file", type=str, help="Path to prompts file")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output report file")

    args = parser.parse_args()

    # Load model
    device = get_device()
    checkpoint = load_checkpoint(args.model, device)

    if args.config:
        config = load_config(args.config)
    else:
        config = checkpoint.get('config', {})

    model_config = config.get('model', {})
    model = MiniGPT(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load tokenizer
    tokenizer = get_tokenizer("gpt2")

    # Initialize evaluator
    evaluator = ModelEvaluator(model, tokenizer, device)

    # Load test data
    test_texts = []
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_texts = [line.strip() for line in f.readlines() if line.strip()]

    prompts = []
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]

    # Run evaluation
    results = evaluator.comprehensive_evaluation(test_texts, prompts)

    # Save report
    evaluator.save_evaluation_report(results, args.output)

    # Print summary
    print("\n📊 Evaluation Summary:")
    print(f"   Perplexity: {results.get('perplexity', 'N/A'):.2f}")
    if 'generation' in results:
        gen = results['generation']
        print(f"   Avg Generation Length: {gen.get('avg_length', 0):.1f} tokens")
        print(f"   Unique Tokens: {gen.get('unique_token_count', 0)}")
        print(f"   Repetition Score: {gen.get('avg_repetition', 0):.3f}")


if __name__ == "__main__":
    main()