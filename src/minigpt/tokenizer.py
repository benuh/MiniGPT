import os
import pickle
from typing import List, Optional
from transformers import GPT2Tokenizer


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.special_tokens = {
            '<|endoftext|>': 50256,
            '<|pad|>': 0,
        }

    def train(self, texts: List[str]) -> None:
        """Train a simple character-level tokenizer"""
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        # Create vocabulary
        vocab = sorted(list(all_chars))

        # Add special tokens
        for token, idx in self.special_tokens.items():
            if token not in vocab:
                vocab.insert(0, token)

        # Create encoder/decoder mappings
        self.encoder = {ch: i for i, ch in enumerate(vocab)}
        self.decoder = {i: ch for i, ch in enumerate(vocab)}

        # Truncate if needed
        if len(self.encoder) > self.vocab_size:
            vocab = vocab[:self.vocab_size]
            self.encoder = {ch: i for i, ch in enumerate(vocab)}
            self.decoder = {i: ch for i, ch in enumerate(vocab)}

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        return [self.encoder.get(char, self.encoder.get('<|endoftext|>', 0)) for char in text]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        return ''.join([self.decoder.get(token_id, '') for token_id in token_ids])

    def save(self, path: str) -> None:
        """Save tokenizer to file"""
        data = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.encoder = data['encoder']
        self.decoder = data['decoder']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']


class GPT2TokenizerWrapper:
    """Wrapper around HuggingFace GPT2 tokenizer for compatibility"""

    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """Encode text to token IDs"""
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        if max_length is not None:
            encoded = encoded[:max_length]
        return encoded

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: List[str], max_length: int, padding: bool = True) -> dict:
        """Batch encode texts with padding"""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=True,
            return_tensors="pt"
        )


def get_tokenizer(tokenizer_type: str = "gpt2", **kwargs):
    """Factory function to get tokenizer"""
    if tokenizer_type == "simple":
        return SimpleTokenizer(**kwargs)
    elif tokenizer_type == "gpt2":
        return GPT2TokenizerWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")