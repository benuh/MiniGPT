#!/usr/bin/env python3
"""
Start the MiniGPT API server
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minigpt.api import run_server


def main():
    parser = argparse.ArgumentParser(description="Start MiniGPT API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to (default: 8000)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes (default: 1)")
    parser.add_argument("--model", type=str,
                       help="Specific model checkpoint to load")

    args = parser.parse_args()

    print("🚀 Starting MiniGPT API Server")
    print("=" * 40)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    if args.model:
        print(f"Model: {args.model}")
    print()
    print("📖 API Documentation:")
    print(f"  Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"  ReDoc: http://{args.host}:{args.port}/redoc")
    print()
    print("🔌 Example API calls:")
    print(f"  Health: curl http://{args.host}:{args.port}/health")
    print(f"  Generate: curl -X POST http://{args.host}:{args.port}/generate \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"prompt\": \"Hello world\", \"max_length\": 50}}'")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 40)

    try:
        run_server(args.host, args.port, args.workers)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")


if __name__ == "__main__":
    main()