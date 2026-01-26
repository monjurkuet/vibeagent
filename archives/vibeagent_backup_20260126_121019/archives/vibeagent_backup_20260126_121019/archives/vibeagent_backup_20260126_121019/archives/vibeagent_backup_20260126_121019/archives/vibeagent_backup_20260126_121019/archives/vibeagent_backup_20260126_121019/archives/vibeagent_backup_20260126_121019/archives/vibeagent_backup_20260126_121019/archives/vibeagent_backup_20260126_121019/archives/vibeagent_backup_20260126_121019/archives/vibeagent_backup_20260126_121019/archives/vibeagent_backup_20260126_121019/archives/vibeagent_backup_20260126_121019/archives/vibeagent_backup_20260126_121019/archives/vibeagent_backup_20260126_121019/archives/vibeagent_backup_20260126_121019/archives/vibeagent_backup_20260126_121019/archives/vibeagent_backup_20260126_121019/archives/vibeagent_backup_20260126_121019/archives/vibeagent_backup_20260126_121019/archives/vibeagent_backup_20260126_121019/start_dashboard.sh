#!/bin/bash

# VibeAgent Dashboard Startup Script

echo "🚀 Starting VibeAgent Dashboard..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Load port configuration
if [ -f "config/ports.json" ]; then
    API_PORT=$(python3 -c "import json; print(json.load(open('config/ports.json'))['ports']['api'])")
    FRONTEND_PORT=$(python3 -c "import json; print(json.load(open('config/ports.json'))['ports']['frontend'])")
else
    API_PORT=8001
    FRONTEND_PORT=3000
fi

# Check if API port is in use
if lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $API_PORT is already in use. Trying alternative port..."
    API_PORT=9000
fi

# Check if frontend port is in use
if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $FRONTEND_PORT is already in use. Trying alternative port..."
    FRONTEND_PORT=9001
fi

echo "📡 Starting FastAPI backend on port $API_PORT..."
cd api
python main.py &
API_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Update frontend API URL
sed -i "s|const API_URL = 'http://localhost:[0-9]*'|const API_URL = 'http://localhost:$API_PORT'|" frontend/index.html

# Start frontend
echo "🌐 Starting frontend on port $FRONTEND_PORT..."
cd frontend
python -m http.server $FRONTEND_PORT &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Dashboard started!"
echo "   Frontend: http://localhost:$FRONTEND_PORT"
echo "   API: http://localhost:$API_PORT"
echo "   API Docs: http://localhost:$API_PORT/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Handle cleanup
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait