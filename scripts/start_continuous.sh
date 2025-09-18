#!/bin/bash
# Start continuous improvement system for MiniGPT

set -e

echo "🚀 Starting MiniGPT Continuous Improvement System"
echo "================================================"

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: Please run this script from the MiniGPT root directory"
    exit 1
fi

# Create necessary directories
mkdir -p logs monitoring checkpoints experiments

# Set up Python environment if needed
if [ ! -d "venv" ]; then
    echo "📦 Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
else
    source venv/bin/activate
fi

# Install additional dependencies if needed
echo "📋 Checking dependencies..."
pip install -q PyYAML wandb

# Make scripts executable
chmod +x scripts/monitor.py
chmod +x scripts/auto_train.py

# Start the monitoring system
echo "🤖 Starting automated training monitor..."
echo "   - Session duration: 8 hours"
echo "   - Max restarts: 5"
echo "   - Cooldown between sessions: 30 minutes"
echo ""
echo "Press Ctrl+C to stop the system gracefully"
echo ""

# Run the monitor
python scripts/monitor.py \
    --action start \
    --session-hours 8 \
    --max-restarts 5

echo "🛑 Continuous improvement system stopped"