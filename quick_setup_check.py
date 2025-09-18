#!/usr/bin/env python3
"""
Quick Setup Check for MiniGPT
=============================
This script performs a rapid check of your MiniGPT setup to ensure
everything is ready for training and testing.

Usage: python quick_setup_check.py
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False

def check_directories():
    """Check required directories"""
    dirs = ["backend", "frontend", "backend/configs", "backend/src"]
    all_good = True

    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")
            all_good = False

    return all_good

def check_configs():
    """Check configuration files"""
    configs = ["backend/configs/small.yaml", "backend/configs/medium.yaml"]
    all_good = True

    for config in configs:
        if os.path.exists(config):
            print(f"✅ {config}")
        else:
            print(f"❌ Missing: {config}")
            all_good = False

    return all_good

def check_backend_install():
    """Check if backend is installed"""
    try:
        result = subprocess.run([
            sys.executable, "-c", "import minigpt; print('Backend installed')"
        ], cwd="backend", capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Backend package installed")
            return True
        else:
            print("❌ Backend not installed. Run: cd backend && pip install -e .")
            return False
    except:
        print("❌ Backend installation check failed")
        return False

def check_node_npm():
    """Check Node.js and npm"""
    try:
        node_result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        npm_result = subprocess.run(['npm', '--version'], capture_output=True, text=True)

        if node_result.returncode == 0 and npm_result.returncode == 0:
            print(f"✅ Node.js {node_result.stdout.strip()}")
            print(f"✅ npm {npm_result.stdout.strip()}")
            return True
        else:
            print("❌ Node.js/npm not found")
            return False
    except:
        print("❌ Node.js/npm check failed")
        return False

def main():
    print("🔍 MiniGPT Quick Setup Check")
    print("=" * 40)

    checks = [
        ("Python Version", check_python),
        ("Directory Structure", check_directories),
        ("Configuration Files", check_configs),
        ("Backend Installation", check_backend_install),
        ("Node.js/npm", check_node_npm)
    ]

    results = []
    for name, check_func in checks:
        print(f"\n📋 {name}:")
        result = check_func()
        results.append(result)

    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print("🎉 All checks passed! You're ready to run autoTest.py")
        print("\nNext steps:")
        print("  python autoTest.py --dry-run    # Preview what will run")
        print("  python autoTest.py              # Run full automation")
    else:
        print(f"⚠️  {passed}/{total} checks passed. Fix issues above first.")

    return passed == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)