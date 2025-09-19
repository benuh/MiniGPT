"""
Remote Model Access System for MiniGPT
======================================

Access pre-trained models from major AI companies without local downloads:
- HuggingFace Inference API (free tier available)
- OpenAI API (GPT-3.5, GPT-4)
- Anthropic Claude API
- Google Gemini API
- Local HuggingFace models (streaming, no disk save)

Usage:
    python -m minigpt.chat --remote hf:gpt2
    python -m minigpt.chat --remote openai:gpt-3.5-turbo
    python -m minigpt.chat --remote claude:claude-3-sonnet
"""

import os
import json
import time
import asyncio
import requests
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    HF_LOCAL_AVAILABLE = True
except ImportError:
    HF_LOCAL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Available remote models
REMOTE_MODELS = {
    # HuggingFace Inference API (Free tier available)
    "hf:gpt2": {
        "provider": "huggingface",
        "model_name": "gpt2",
        "size": "124M",
        "cost": "Free",
        "description": "OpenAI GPT-2 via HuggingFace API"
    },
    "hf:distilgpt2": {
        "provider": "huggingface",
        "model_name": "distilgpt2",
        "size": "82M",
        "cost": "Free",
        "description": "Distilled GPT-2 via HuggingFace API"
    },
    "hf:gpt-neo-125m": {
        "provider": "huggingface",
        "model_name": "EleutherAI/gpt-neo-125M",
        "size": "125M",
        "cost": "Free",
        "description": "EleutherAI GPT-Neo via HuggingFace API"
    },
    "hf:gpt-neo-1.3b": {
        "provider": "huggingface",
        "model_name": "EleutherAI/gpt-neo-1.3B",
        "size": "1.3B",
        "cost": "Free",
        "description": "EleutherAI GPT-Neo 1.3B via HuggingFace API"
    },
    "hf:falcon-7b": {
        "provider": "huggingface",
        "model_name": "tiiuae/falcon-7b-instruct",
        "size": "7B",
        "cost": "Free",
        "description": "Falcon 7B Instruct via HuggingFace API"
    },

    # OpenAI API (Requires API key & payment)
    "openai:gpt-3.5-turbo": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "size": "Unknown",
        "cost": "$0.002/1K tokens",
        "description": "OpenAI GPT-3.5 Turbo"
    },
    "openai:gpt-4": {
        "provider": "openai",
        "model_name": "gpt-4",
        "size": "Unknown",
        "cost": "$0.03/1K tokens",
        "description": "OpenAI GPT-4"
    },
    "openai:gpt-4-turbo": {
        "provider": "openai",
        "model_name": "gpt-4-turbo-preview",
        "size": "Unknown",
        "cost": "$0.01/1K tokens",
        "description": "OpenAI GPT-4 Turbo"
    },

    # Anthropic Claude API (Requires API key & payment)
    "claude:claude-3-haiku": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "size": "Unknown",
        "cost": "$0.00025/1K tokens",
        "description": "Anthropic Claude 3 Haiku (fastest)"
    },
    "claude:claude-3-sonnet": {
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "size": "Unknown",
        "cost": "$0.003/1K tokens",
        "description": "Anthropic Claude 3 Sonnet (balanced)"
    },
    "claude:claude-3-opus": {
        "provider": "anthropic",
        "model_name": "claude-3-opus-20240229",
        "size": "Unknown",
        "cost": "$0.015/1K tokens",
        "description": "Anthropic Claude 3 Opus (most capable)"
    },

    # Local HuggingFace (No disk save, load in memory)
    "local:gpt2": {
        "provider": "local_hf",
        "model_name": "gpt2",
        "size": "124M",
        "cost": "Free (local compute)",
        "description": "GPT-2 loaded locally in memory"
    },
    "local:distilgpt2": {
        "provider": "local_hf",
        "model_name": "distilgpt2",
        "size": "82M",
        "cost": "Free (local compute)",
        "description": "DistilGPT-2 loaded locally in memory"
    },

    # Backup simple models that should always work
    "simple:demo": {
        "provider": "demo",
        "model_name": "demo",
        "size": "0B",
        "cost": "Free",
        "description": "Simple demo model (always works)"
    }
}

class RemoteModelClient:
    """Base class for remote model clients"""

    def __init__(self, model_key: str):
        self.model_key = model_key
        self.model_info = REMOTE_MODELS.get(model_key, {})
        self.provider = self.model_info.get("provider")
        self.model_name = self.model_info.get("model_name")

        if not self.model_info:
            raise ValueError(f"Unknown remote model: {model_key}")

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8, **kwargs) -> str:
        """Generate text - to be implemented by subclasses"""
        raise NotImplementedError

class HuggingFaceClient(RemoteModelClient):
    """HuggingFace Inference API client"""

    def __init__(self, model_key: str):
        super().__init__(model_key)
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

        if not self.api_token:
            logger.warning("No HUGGINGFACE_API_TOKEN found. Using public API (may have rate limits)")

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8, **kwargs) -> str:
        """Generate text using HuggingFace Inference API"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"

            # For public models, try a simple text generation format first
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": temperature,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            }

            logger.info(f"Making request to {self.api_url}")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)

            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")

            if response.status_code == 503:
                # Model is loading, wait and retry
                logger.info("Model is loading, waiting...")
                time.sleep(20)
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"API response: {result}")

                if isinstance(result, list) and len(result) > 0:
                    generated = result[0].get("generated_text", "")
                    # Remove the input prompt from the response if it's included
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated or "No response generated"
                elif isinstance(result, dict):
                    generated = result.get("generated_text", "")
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated or "No response generated"
                else:
                    return str(result)

            elif response.status_code == 401:
                # Try without authentication for public models
                logger.info("401 error, trying without authentication...")
                headers_no_auth = {"Content-Type": "application/json"}
                response = requests.post(self.api_url, headers=headers_no_auth, json=payload, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated = result[0].get("generated_text", "")
                        if generated.startswith(prompt):
                            generated = generated[len(prompt):].strip()
                        return generated or "No response generated"
                    elif isinstance(result, dict):
                        generated = result.get("generated_text", "")
                        if generated.startswith(prompt):
                            generated = generated[len(prompt):].strip()
                        return generated or "No response generated"
                elif response.status_code == 401:
                    # HuggingFace API might be requiring authentication now
                    # Fall back to local model instead
                    logger.info("HuggingFace API requires authentication. Falling back to local model...")
                    return self._fallback_to_local(prompt, max_length, temperature)

                error_msg = f"HuggingFace API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            else:
                # For any other error, try fallback to local model
                if response.status_code == 401 or response.status_code == 403:
                    logger.info(f"API authentication failed ({response.status_code}). Falling back to local model...")
                    return self._fallback_to_local(prompt, max_length, temperature)

                error_msg = f"HuggingFace API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"

        except Exception as e:
            logger.error(f"HuggingFace generation error: {str(e)}")
            # Also try fallback on general exceptions
            try:
                return self._fallback_to_local(prompt, max_length, temperature)
            except Exception as fallback_error:
                return f"Error: {str(e)}. Fallback also failed: {str(fallback_error)}"

    def _fallback_to_local(self, prompt: str, max_length: int, temperature: float) -> str:
        """Fallback to local HuggingFace model when API fails"""
        try:
            if not HF_LOCAL_AVAILABLE:
                return "Error: HuggingFace API unavailable and local transformers not installed. Please install: pip install transformers torch"

            logger.info(f"Loading local {self.model_name} model...")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            model = model.to(device)

            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text or "No response generated"

        except Exception as e:
            logger.error(f"Local fallback error: {str(e)}")
            return f"Error: Local fallback failed - {str(e)}. Try: pip install transformers torch"

class OpenAIClient(RemoteModelClient):
    """OpenAI API client"""

    def __init__(self, model_key: str):
        super().__init__(model_key)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")

        openai.api_key = self.api_key

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8, **kwargs) -> str:
        """Generate text using OpenAI API"""
        try:
            if "gpt-3.5" in self.model_name or "gpt-4" in self.model_name:
                # Use ChatCompletion API
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            else:
                # Use Completion API for older models
                response = openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_length,
                    temperature=temperature
                )
                return response.choices[0].text.strip()

        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            return f"Error: {str(e)}"

class AnthropicClient(RemoteModelClient):
    """Anthropic Claude API client"""

    def __init__(self, model_key: str):
        super().__init__(model_key)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8, **kwargs) -> str:
        """Generate text using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_length,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Anthropic generation error: {str(e)}")
            return f"Error: {str(e)}"

class LocalHuggingFaceClient(RemoteModelClient):
    """Local HuggingFace model client (in-memory, no disk save)"""

    def __init__(self, model_key: str):
        super().__init__(model_key)
        if not HF_LOCAL_AVAILABLE:
            raise ImportError("Transformers not installed. Run: pip install transformers torch")

        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer in memory"""
        try:
            logger.info(f"Loading {self.model_name} in memory...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"✅ {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {str(e)}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8, **kwargs) -> str:
        """Generate text using local model"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the new tokens
            generated_tokens = outputs[0][inputs.shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Local generation error: {str(e)}")
            return f"Error: {str(e)}"

class DemoClient(RemoteModelClient):
    """Simple demo client that always works"""

    def __init__(self, model_key: str):
        super().__init__(model_key)

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8, **kwargs) -> str:
        """Generate a simple demo response"""
        responses = [
            f"Hello! You said: '{prompt}'. This is a demo response from MiniGPT.",
            f"Thanks for your message: '{prompt}'. I'm a simple demo model working perfectly!",
            f"I received: '{prompt}'. The demo model is functioning correctly. Try using a real model like 'local:gpt2' for better responses!",
            f"Demo response to: '{prompt}'. Everything is working! You can now use real AI models through the remote system."
        ]
        import random
        return random.choice(responses)

class RemoteModelManager:
    """Manage remote model access"""

    def __init__(self):
        self.clients = {}

    def get_client(self, model_key: str) -> RemoteModelClient:
        """Get or create a client for the model"""
        if model_key not in self.clients:
            model_info = REMOTE_MODELS.get(model_key, {})
            provider = model_info.get("provider")

            if provider == "huggingface":
                self.clients[model_key] = HuggingFaceClient(model_key)
            elif provider == "openai":
                self.clients[model_key] = OpenAIClient(model_key)
            elif provider == "anthropic":
                self.clients[model_key] = AnthropicClient(model_key)
            elif provider == "local_hf":
                self.clients[model_key] = LocalHuggingFaceClient(model_key)
            elif provider == "demo":
                self.clients[model_key] = DemoClient(model_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        return self.clients[model_key]

    def list_models(self) -> Dict[str, Dict]:
        """List all available remote models"""
        return REMOTE_MODELS

    def generate_text(self, model_key: str, prompt: str, max_length: int = 100,
                     temperature: float = 0.8, **kwargs) -> str:
        """Generate text using specified remote model"""
        client = self.get_client(model_key)
        return client.generate_text(prompt, max_length, temperature, **kwargs)

class RemoteChatBot:
    """Chat interface for remote models"""

    def __init__(self, model_key: str):
        self.model_key = model_key
        self.manager = RemoteModelManager()
        self.model_info = REMOTE_MODELS.get(model_key, {})

        if not self.model_info:
            available = list(REMOTE_MODELS.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available}")

        # Initialize the client
        try:
            self.client = self.manager.get_client(model_key)
            print(f"🌐 Remote model loaded: {model_key}")
            print(f"📊 Provider: {self.model_info['provider']}")
            print(f"💰 Cost: {self.model_info['cost']}")
            print(f"📝 Description: {self.model_info['description']}")
        except Exception as e:
            print(f"❌ Failed to initialize {model_key}: {str(e)}")
            raise

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8) -> str:
        """Generate text from prompt"""
        return self.manager.generate_text(self.model_key, prompt, max_length, temperature)

    def chat_loop(self):
        """Interactive chat loop"""
        print(f"\n🌐 Remote Chat with {self.model_key}")
        print("=" * 50)
        print("Type 'quit', 'exit', or 'q' to end")
        print("Type 'help' for commands")
        print("=" * 50)

        conversation_history = ""

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue

                if not user_input:
                    continue

                # Generate response
                print(f"\n🤖 {self.model_key}: ", end="", flush=True)

                # For chat models, use conversation format
                if self.model_info['provider'] in ['openai', 'anthropic']:
                    response = self.generate_text(user_input, max_length=150, temperature=0.8)
                else:
                    # For completion models, build conversation context
                    full_prompt = f"{conversation_history}Human: {user_input}\nAssistant:"
                    response = self.generate_text(full_prompt, max_length=100, temperature=0.8)

                print(response)

                # Update conversation history for completion models
                if self.model_info['provider'] not in ['openai', 'anthropic']:
                    conversation_history += f"Human: {user_input}\nAssistant: {response}\n"

                    # Keep conversation manageable
                    if len(conversation_history) > 1000:
                        lines = conversation_history.split('\n')
                        conversation_history = '\n'.join(lines[-10:])

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def _show_help(self):
        """Show help information"""
        print("\n📖 Available commands:")
        print("  help        - Show this help")
        print("  /models     - List available models")
        print("  /info       - Show current model info")
        print("  /switch <model> - Switch to different model")
        print("  quit/exit/q - Exit chat")

    def _handle_command(self, command: str):
        """Handle special commands"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == '/models':
            print("\n🌐 Available Remote Models:")
            for key, info in REMOTE_MODELS.items():
                cost_color = "🟢" if "Free" in info['cost'] else "🟡"
                print(f"  {cost_color} {key} - {info['description']} ({info['cost']})")

        elif cmd == '/info':
            print(f"\n📊 Current Model: {self.model_key}")
            for key, value in self.model_info.items():
                print(f"  {key}: {value}")

        elif cmd == '/switch' and len(parts) > 1:
            new_model = parts[1]
            if new_model in REMOTE_MODELS:
                try:
                    self.__init__(new_model)
                    print(f"✅ Switched to {new_model}")
                except Exception as e:
                    print(f"❌ Failed to switch: {e}")
            else:
                print(f"❌ Unknown model: {new_model}")

        else:
            print("❌ Unknown command. Type 'help' for available commands")

def list_remote_models() -> Dict[str, Dict]:
    """List all available remote models"""
    return REMOTE_MODELS

def create_remote_chat(model_key: str) -> RemoteChatBot:
    """Create a remote chat bot"""
    return RemoteChatBot(model_key)

def main():
    """Command line interface for remote models"""
    import argparse

    parser = argparse.ArgumentParser(description="Access remote AI models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--chat", type=str, help="Start chat with model (e.g., hf:gpt2)")
    parser.add_argument("--prompt", type=str, help="Single prompt to model")
    parser.add_argument("--model", type=str, default="hf:gpt2", help="Model to use")
    parser.add_argument("--setup", action="store_true", help="Show setup instructions")

    args = parser.parse_args()

    if args.setup:
        print("🔧 Remote Models Setup Guide")
        print("=" * 40)
        print("\n1. HuggingFace (Free):")
        print("   export HUGGINGFACE_API_TOKEN=your_token")
        print("   (Optional - works without token but with rate limits)")
        print("\n2. OpenAI (Paid):")
        print("   export OPENAI_API_KEY=your_api_key")
        print("   pip install openai")
        print("\n3. Anthropic Claude (Paid):")
        print("   export ANTHROPIC_API_KEY=your_api_key")
        print("   pip install anthropic")
        print("\n4. Local HuggingFace (Free):")
        print("   pip install transformers torch")
        print("   (No API key needed)")
        return

    if args.list:
        print("🌐 Available Remote Models:")
        print("=" * 50)

        by_provider = {}
        for key, info in REMOTE_MODELS.items():
            provider = info['provider']
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append((key, info))

        for provider, models in by_provider.items():
            print(f"\n📡 {provider.upper()}:")
            for key, info in models:
                cost_color = "🟢" if "Free" in info['cost'] else "🟡"
                print(f"  {cost_color} {key}")
                print(f"     {info['description']}")
                print(f"     Cost: {info['cost']}")

        print(f"\nUsage: python -m minigpt.remote --chat hf:gpt2")
        return

    if args.chat:
        try:
            chatbot = RemoteChatBot(args.chat)
            chatbot.chat_loop()
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try: python -m minigpt.remote --setup")

    elif args.prompt:
        try:
            manager = RemoteModelManager()
            response = manager.generate_text(args.model, args.prompt)
            print(f"Model: {args.model}")
            print(f"Prompt: {args.prompt}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()