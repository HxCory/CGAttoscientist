#!/bin/bash

# PhD Thesis RAG Assistant - Quick Start Script
# This script starts both backend and frontend services

set -e

echo "=================================="
echo "PhD Thesis RAG Assistant Launcher"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit
}

trap cleanup EXIT INT TERM

# Start Backend
echo "🚀 Starting Backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please copy .env.example to .env and add your ANTHROPIC_API_KEY"
    echo ""
    read -p "Press Enter to continue anyway (or Ctrl+C to exit)..."
fi

# Install dependencies if needed
pip install -q -r requirements.txt

# Start backend in background
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID) at http://localhost:8000"

cd ..

# Start Frontend
echo ""
echo "🎨 Starting Frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Start frontend in background
npm run dev &
FRONTEND_PID=$!
echo "✓ Frontend started (PID: $FRONTEND_PID) at http://localhost:5173"

cd ..

echo ""
echo "=================================="
echo "✅ Both services are running!"
echo "=================================="
echo ""
echo "📱 Open your browser to: http://localhost:5173"
echo "📊 Backend API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
