#!/bin/bash

echo "🎨 Starting MiniGPT Frontend..."
echo "==============================="

# Navigate to frontend directory
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
fi

# Start the development server
echo "🌐 Starting React development server..."
echo "Frontend will be available at: http://localhost:3001"
echo "Press Ctrl+C to stop"
echo ""

PORT=3001 npm run dev