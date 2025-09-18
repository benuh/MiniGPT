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

# Check if Tailwind packages are installed correctly
echo "🔧 Checking Tailwind CSS installation..."
if [ ! -d "node_modules/tailwindcss" ]; then
    echo "📦 Installing Tailwind CSS..."
    npm install -D tailwindcss@3.4.16 postcss autoprefixer @tailwindcss/forms @tailwindcss/typography
fi

# Clear cache to ensure fresh build
echo "🧹 Clearing build cache..."
rm -rf .next build dist

# Start the development server
echo "🌐 Starting React development server..."
echo "Frontend will be available at: http://localhost:3001"
echo "Press Ctrl+C to stop"
echo ""

PORT=3001 npm run dev