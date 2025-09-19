#!/usr/bin/env python3
"""
Test script to verify model path handling works from different directories
"""

import sys
import os
from pathlib import Path

# Add backend src to path
sys.path.insert(0, 'backend/src')

try:
    from minigpt.utils import get_checkpoints_dir, find_best_checkpoint, find_latest_checkpoint

    print("🔍 Testing model path detection...")
    print(f"Current working directory: {Path.cwd()}")
    print()

    # Test checkpoints directory detection
    checkpoints_dir = get_checkpoints_dir()
    print(f"📁 Checkpoints directory: {checkpoints_dir}")
    print(f"   Exists: {checkpoints_dir.exists()}")

    if checkpoints_dir.exists():
        models = list(checkpoints_dir.glob("*.pt"))
        print(f"   Models found: {len(models)}")
        for model in models:
            print(f"     - {model.name} ({model.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print("   No models found")

    print()

    # Test best model detection
    best_model = find_best_checkpoint()
    if best_model:
        print(f"🏆 Best model: {best_model}")
    else:
        print("🏆 No best model found")

    # Test latest model detection
    latest_model = find_latest_checkpoint()
    if latest_model:
        print(f"🕐 Latest model: {latest_model}")
    else:
        print("🕐 No latest model found")

    print()

    # Test from different working directories
    print("🧪 Testing from backend directory...")
    os.chdir('backend')

    checkpoints_dir_backend = get_checkpoints_dir()
    print(f"📁 Checkpoints directory (from backend/): {checkpoints_dir_backend}")
    print(f"   Exists: {checkpoints_dir_backend.exists()}")

    best_model_backend = find_best_checkpoint()
    if best_model_backend:
        print(f"🏆 Best model (from backend/): {best_model_backend}")
    else:
        print("🏆 No best model found (from backend/)")

    # Test command line usage
    print()
    print("✅ Path handling test complete!")
    print()
    print("You can now use:")
    print("  cd backend && python -m minigpt.chat")
    print("  cd backend && python -m minigpt.api")
    print("  python autoTest.py")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the MiniGPT project directory")
except Exception as e:
    print(f"❌ Error: {e}")