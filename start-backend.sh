#!/bin/bash

echo "🚀 Starting MiniGPT Backend..."
echo "================================"

# Navigate to backend directory
cd backend

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f ".installed" ]; then
    echo "📦 Installing dependencies..."
    pip install -e .
    touch .installed
fi

# Start the API server
echo "🌐 Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python -m minigpt.api