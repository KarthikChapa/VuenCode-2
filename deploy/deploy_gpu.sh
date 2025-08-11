#!/bin/bash

# VuenCode Phase 2 - GPU Server Deployment Script
# Deploy VuenCode with GPU optimization and competition-grade performance

set -e

echo "🚀 Starting VuenCode Phase 2 GPU Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    print_status "GPU detected successfully"
else
    print_warning "nvidia-smi not found. GPU acceleration may not be available."
fi

# Create virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install GPU requirements
print_status "Installing GPU-optimized dependencies..."
if [ -f "docker/requirements-gpu.txt" ]; then
    pip install -r docker/requirements-gpu.txt
else
    print_warning "GPU requirements file not found, using local requirements"
    pip install -r docker/requirements-local.txt
fi

# Install additional GPU packages
print_status "Installing CUDA-specific packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorrt
pip install nvidia-ml-py3

# Set environment variables
print_status "Configuring environment variables..."
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
export FORCE_CUDA=1

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f "configs/competition.env" ]; then
        cp configs/competition.env .env
        print_status "Created .env from competition template"
    else
        print_error "No environment configuration found"
        exit 1
    fi
fi

# Source environment variables
source .env

# Check required environment variables
if [ -z "$GOOGLE_API_KEY" ]; then
    print_error "GOOGLE_API_KEY not set. Please add it to .env file"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start the application
print_status "Starting VuenCode Phase 2 server..."
nohup python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop uvloop \
    --log-config utils/logging_config.json > logs/vuencode.log 2>&1 &

# Get the PID
PID=$!
echo $PID > logs/vuencode.pid

print_status "VuenCode server started with PID: $PID"
print_status "Logs are being written to: logs/vuencode.log"

# Wait a moment for startup
sleep 5

# Check if the service is running
if curl -s http://localhost:8000/health > /dev/null; then
    print_status "✅ VuenCode Phase 2 is running successfully!"
    print_status "Health endpoint: http://localhost:8000/health"
    print_status "API endpoint: http://localhost:8000/infer"
    print_status ""
    print_status "To view logs: tail -f logs/vuencode.log"
    print_status "To stop service: kill \$(cat logs/vuencode.pid)"
else
    print_error "❌ Service failed to start. Check logs/vuencode.log for details"
    exit 1
fi

print_status "🎯 GPU deployment complete! Ready for competition."