#!/bin/bash

echo "🤖 Starting MiniGPT Full Stack..."
echo "=================================="
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend stopped"
    fi
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

echo "🚀 Starting backend..."
./start-backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

echo ""
echo "🎨 Starting frontend..."
./start-frontend.sh &
FRONTEND_PID=$!

echo ""
echo "✅ Both services are starting..."
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔌 Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for background processes
wait