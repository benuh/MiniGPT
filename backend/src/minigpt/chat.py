import torch
import argparse
from typing import Optional

from .model import MiniGPT
from .tokenizer import get_tokenizer
from .config import load_config
from .utils import get_device, load_checkpoint, find_best_checkpoint


class ChatBot:
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.device = get_device()

        # Load checkpoint
        checkpoint = load_checkpoint(model_path, self.device)
        self.config = checkpoint.get('config', {})

        # Override with external config if provided
        if config_path:
            external_config = load_config(config_path)
            self.config.update(external_config)

        # Initialize tokenizer
        self.tokenizer = get_tokenizer("gpt2")

        # Initialize and load model
        model_config = self.config.get('model', {})
        self.model = MiniGPT(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded with {self.model.count_parameters():,} parameters")
        print(f"Using device: {self.device}")

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8, top_k: int = 50):
        """Generate text based on a prompt"""
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_tensor,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k
            )

        # Decode the generated text
        generated_text = self.tokenizer.decode(generated[0].tolist())
        return generated_text

    def chat_loop(self):
        """Interactive chat loop"""
        print("\n> MiniGPT Chat Interface")
        print("=" * 50)
        print("Type 'quit', 'exit', or 'q' to end the conversation")
        print("Type 'help' for available commands")
        print("=" * 50)

        conversation_history = ""

        while True:
            try:
                user_input = input("\n=d You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n=K Goodbye!")
                    break

                if user_input.lower() == 'help':
                    self.show_help()
                    continue

                if user_input.lower().startswith('/'):
                    self.handle_command(user_input)
                    continue

                if not user_input:
                    continue

                # Add to conversation history
                conversation_history += f"Human: {user_input}\nAssistant: "

                # Generate response
                print("\n> MiniGPT: ", end="", flush=True)
                response = self.generate_text(
                    conversation_history,
                    max_length=50,
                    temperature=0.8,
                    top_k=50
                )

                # Extract just the assistant's response
                response_lines = response.split('\n')
                assistant_response = ""
                for line in response_lines:
                    if line.startswith("Assistant:"):
                        assistant_response = line.replace("Assistant:", "").strip()
                        break

                if not assistant_response:
                    # Fallback: take the generated text after the prompt
                    assistant_response = response[len(conversation_history):].split('\n')[0].strip()

                print(assistant_response)

                # Update conversation history
                conversation_history += assistant_response + "\n"

                # Keep conversation history manageable
                if len(conversation_history) > 1000:
                    lines = conversation_history.split('\n')
                    conversation_history = '\n'.join(lines[-10:])

            except KeyboardInterrupt:
                print("\n\n=K Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nL Error: {e}")

    def show_help(self):
        """Show help information"""
        print("\n📖 Available commands:")
        print("  help        - Show this help message")
        print("  /temp <val> - Set temperature (0.1-2.0)")
        print("  /topk <val> - Set top-k value (1-100)")
        print("  /len <val>  - Set max generation length")
        print("  /reset      - Reset conversation history")
        print("  quit/exit/q - Exit the chat")

    def handle_command(self, command: str):
        """Handle special commands"""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == '/temp' and len(parts) == 2:
            try:
                temp = float(parts[1])
                if 0.1 <= temp <= 2.0:
                    self.temperature = temp
                    print(f" Temperature set to {temp}")
                else:
                    print("L Temperature must be between 0.1 and 2.0")
            except ValueError:
                print("L Invalid temperature value")

        elif cmd == '/topk' and len(parts) == 2:
            try:
                topk = int(parts[1])
                if 1 <= topk <= 100:
                    self.top_k = topk
                    print(f" Top-k set to {topk}")
                else:
                    print("L Top-k must be between 1 and 100")
            except ValueError:
                print("L Invalid top-k value")

        elif cmd == '/len' and len(parts) == 2:
            try:
                length = int(parts[1])
                if 1 <= length <= 200:
                    self.max_length = length
                    print(f" Max length set to {length}")
                else:
                    print("L Max length must be between 1 and 200")
            except ValueError:
                print("L Invalid length value")

        elif cmd == '/reset':
            print(" Conversation history reset")

        else:
            print("L Unknown command. Type 'help' for available commands")


def main():
    parser = argparse.ArgumentParser(description="Chat with MiniGPT model")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model checkpoint (auto-finds latest if not specified)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Single prompt mode (non-interactive)")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling")

    args = parser.parse_args()

    # Find model if not specified
    model_path = args.model
    if not model_path:
        best_checkpoint = find_best_checkpoint()
        if not best_checkpoint:
            print("❌ No trained models found!")
            print("Please train a model first:")
            print("  python -m minigpt.train --config configs/small.yaml")
            print("  or run: python autoTest.py")
            return
        model_path = str(best_checkpoint)
        print(f"📁 Using model: {model_path}")

    # Initialize chatbot
    chatbot = ChatBot(model_path, args.config)

    if args.prompt:
        # Single prompt mode
        response = chatbot.generate_text(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        # Interactive chat mode
        chatbot.chat_loop()


if __name__ == "__main__":
    main()