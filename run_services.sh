#!/bin/bash

# ============================================================
# 🚀 Pneumonia Detection Project - Services Runner
# ============================================================
# This script starts all services for the project:
#   1. FastAPI Backend (api.py) on port 8000
#   2. Frontend HTTP Server on port 3000
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project directory (where this script is located)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Port configuration
API_PORT=${API_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-3000}

# Function to print colored messages
print_header() {
    echo -e "\n${CYAN}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on a port
kill_port() {
    local port=$1
    local pid=$(lsof -Pi :$port -sTCP:LISTEN -t 2>/dev/null)
    if [ -n "$pid" ]; then
        print_warning "Killing existing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
}

# Cleanup function to stop all services
cleanup() {
    echo ""
    print_header "🛑 Stopping all services..."
    
    # Kill backend
    if [ -n "$API_PID" ]; then
        print_info "Stopping API server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null || true
    fi
    
    # Kill frontend
    if [ -n "$FRONTEND_PID" ]; then
        print_info "Stopping Frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Additional cleanup for any remaining processes on our ports
    kill_port $API_PORT
    kill_port $FRONTEND_PORT
    
    print_success "All services stopped."
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Print startup banner
print_header "🫁 PNEUMONIA DETECTION - SERVICES RUNNER"

echo -e "${YELLOW}Project Directory:${NC} $PROJECT_DIR"
echo -e "${YELLOW}API Port:${NC} $API_PORT"
echo -e "${YELLOW}Frontend Port:${NC} $FRONTEND_PORT"
echo ""

# Step 1: Check and activate virtual environment
print_info "Checking virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    print_success "Virtual environment activated"
else
    print_warning "Virtual environment not found. Using system Python."
fi

# Step 2: Check if ports are in use and free them
print_info "Checking if ports are available..."

if check_port $API_PORT; then
    print_warning "Port $API_PORT is in use."
    read -p "Kill existing process on port $API_PORT? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill_port $API_PORT
    else
        print_error "Please free port $API_PORT and try again."
        exit 1
    fi
fi

if check_port $FRONTEND_PORT; then
    print_warning "Port $FRONTEND_PORT is in use."
    read -p "Kill existing process on port $FRONTEND_PORT? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill_port $FRONTEND_PORT
    else
        print_error "Please free port $FRONTEND_PORT and try again."
        exit 1
    fi
fi

print_success "Ports are available"

# Step 3: Start the API backend
print_header "🔧 Starting Backend API Server..."
print_info "Running: python3 api.py"

PORT=$API_PORT python3 api.py &
API_PID=$!
print_success "API server started (PID: $API_PID)"

# Wait for API to be ready
print_info "Waiting for API to initialize..."
sleep 3

# Check if API is running
if ! kill -0 $API_PID 2>/dev/null; then
    print_error "API server failed to start!"
    exit 1
fi

# Step 4: Start the Frontend server
print_header "🌐 Starting Frontend Server..."
print_info "Running: python3 -m http.server $FRONTEND_PORT (from frontend/)"

cd frontend
python3 -m http.server $FRONTEND_PORT &
FRONTEND_PID=$!
cd ..
print_success "Frontend server started (PID: $FRONTEND_PID)"

# Wait a moment for frontend to start
sleep 2

# Check if Frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    print_error "Frontend server failed to start!"
    exit 1
fi

# Step 5: Print success message with links
print_header "✅ ALL SERVICES RUNNING!"

echo -e "${GREEN}┌─────────────────────────────────────────────────────────────┐${NC}"
echo -e "${GREEN}│                    🎉 SERVICES READY!                      │${NC}"
echo -e "${GREEN}├─────────────────────────────────────────────────────────────┤${NC}"
echo -e "${GREEN}│                                                             │${NC}"
echo -e "${GREEN}│  🌐 Frontend:    ${CYAN}http://localhost:$FRONTEND_PORT${GREEN}                  │${NC}"
echo -e "${GREEN}│  🔧 API:         ${CYAN}http://localhost:$API_PORT${GREEN}                     │${NC}"
echo -e "${GREEN}│  📚 API Docs:    ${CYAN}http://localhost:$API_PORT/docs${GREEN}                │${NC}"
echo -e "${GREEN}│                                                             │${NC}"
echo -e "${GREEN}├─────────────────────────────────────────────────────────────┤${NC}"
echo -e "${GREEN}│  Press ${RED}Ctrl+C${GREEN} to stop all services                        │${NC}"
echo -e "${GREEN}└─────────────────────────────────────────────────────────────┘${NC}"
echo ""

# Step 6: Wait for API health check
print_info "Waiting for API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
        print_success "API is healthy and ready!"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Step 7: Auto-open browser (optional)
if [ "$1" != "--no-browser" ]; then
    print_info "Opening browser..."
    
    # Try different methods to open browser (cross-platform)
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:$FRONTEND_PORT" 2>/dev/null &
    elif command -v open &> /dev/null; then
        open "http://localhost:$FRONTEND_PORT" 2>/dev/null &
    elif command -v start &> /dev/null; then
        start "http://localhost:$FRONTEND_PORT" 2>/dev/null &
    elif [ -n "$BROWSER" ]; then
        $BROWSER "http://localhost:$FRONTEND_PORT" 2>/dev/null &
    else
        # For WSL - try to open in Windows browser
        if grep -qi microsoft /proc/version 2>/dev/null; then
            cmd.exe /c start "http://localhost:$FRONTEND_PORT" 2>/dev/null &
        else
            print_warning "Could not auto-open browser. Please open manually:"
            echo -e "  ${CYAN}http://localhost:$FRONTEND_PORT${NC}"
        fi
    fi
fi

print_success "Ready! Upload an X-ray image to get predictions."
echo ""

# Keep script running and wait for both processes
wait $API_PID $FRONTEND_PID
