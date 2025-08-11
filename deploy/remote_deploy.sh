#!/bin/bash
# VuenCode Phase 2 Remote GPU Deployment Script
# Automatically deploys the complete system to remote GPU server

set -e  # Exit on any error

echo "🚀 VuenCode Phase 2 Remote GPU Deployment Starting..."
echo "======================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/KarthikChapa/VuenCode-2.git"
PROJECT_DIR="$HOME/VuenCode"
PYTHON_VERSION="3.11"
PORT=8000

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on GPU server
print_status "Checking system requirements..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    print_warning "No NVIDIA GPU detected, continuing with CPU-only deployment"
fi

# Check Python version
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    print_warning "Using system Python3 instead of Python 3.11"
else
    print_error "Python 3 not found. Please install Python 3.11 or later."
    exit 1
fi

print_success "Using Python: $($PYTHON_CMD --version)"

# Step 1: Clean up any existing installation
print_status "Cleaning up existing installation..."
if [ -d "$PROJECT_DIR" ]; then
    print_status "Removing existing project directory..."
    rm -rf "$PROJECT_DIR"
fi

# Step 2: Clone the latest repository
print_status "Cloning VuenCode repository..."
git clone "$REPO_URL" "$PROJECT_DIR"
cd "$PROJECT_DIR"
print_success "Repository cloned successfully"

# Step 3: Set up Python virtual environment
print_status "Setting up Python virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate
print_success "Virtual environment created and activated"

# Step 4: Upgrade pip and install build tools
print_status "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Step 5: Install PyTorch with CUDA support (if GPU available)
print_status "Installing PyTorch with optimal configuration..."
if command -v nvidia-smi &> /dev/null; then
    print_status "Installing PyTorch with CUDA support..."
    # Install PyTorch with CUDA 12.1 support (most common on cloud GPUs)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    print_success "PyTorch with CUDA installed"
else
    print_status "Installing PyTorch CPU-only version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    print_success "PyTorch CPU-only installed"
fi

# Step 6: Install core dependencies
print_status "Installing core dependencies..."
pip install -r docker/requirements-local.txt
print_success "Dependencies installed"

# Step 7: Install additional GPU-optimized packages
print_status "Installing additional GPU-optimized packages..."
pip install accelerate transformers optimum[onnxruntime-gpu] 2>/dev/null || pip install accelerate transformers optimum
print_success "Additional packages installed"

# Step 8: Create required directories
print_status "Creating required directories..."
mkdir -p logs cache data/videos temp
print_success "Directories created"

# Step 9: Set environment variables for GPU deployment
print_status "Configuring environment variables..."
export DEPLOYMENT_MODE=gpu
export USE_GPU_ACCELERATION=true
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Create environment file
cat > configs/gpu.env << EOF
# GPU Deployment Configuration
DEPLOYMENT_MODE=gpu
USE_GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
API_HOST=0.0.0.0
API_PORT=8000

# Performance Optimization
TARGET_LATENCY_MS=400
MAX_FRAMES_PER_VIDEO=64
GPU_MEMORY_OPTIMIZATION=true
ENABLE_TENSORRT=false
BATCH_SIZE=4

# Gemini API Configuration (update with your key)
GEMINI_API_KEY=your_api_key_here
GEMINI_FLASH_MODEL=gemini-2.0-flash-exp
GEMINI_PRO_MODEL=gemini-1.5-pro

# Caching and Fallback
ENABLE_CACHING=true
CACHE_TTL_HOURS=24
FALLBACK_ENABLED=true
EOF

print_success "Environment configuration created"

# Step 10: Test Python imports and system functionality
# Ensure PYTHON_CMD is set
PYTHON_CMD=${PYTHON_CMD:-python3}

print_status "Testing system functionality..."
$PYTHON_CMD -c "
import sys
sys.path.append('.')
try:
    from utils.config import get_config
    from api.main import app
    print('✅ All imports successful')
    
    # Test GPU availability
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.get_device_name()}')
        print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    else:
        print('⚠️  CUDA not available, using CPU')
        
    print('✅ System test passed')
except Exception as e:
    print(f'❌ System test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "System functionality test passed"
else
    print_error "System test failed. Check dependencies."
    exit 1
fi

# Step 11: Start the VuenCode server
print_status "Starting VuenCode server..."
print_status "Server will start on port $PORT"
print_status "Health check: http://localhost:$PORT/health"
print_status "API endpoint: http://localhost:$PORT/infer"

# Create systemd service file for automatic startup (optional)
cat > vuencode.service << EOF
[Unit]
Description=VuenCode Phase 2 GPU Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
Environment=PYTHONPATH=$PROJECT_DIR
Environment=DEPLOYMENT_MODE=gpu
Environment=USE_GPU_ACCELERATION=true
Environment=CUDA_VISIBLE_DEVICES=0
Environment=NVIDIA_VISIBLE_DEVICES=all
ExecStart=$PROJECT_DIR/venv/bin/python -m uvicorn VuenCode.api.main:app --host 0.0.0.0 --port $PORT --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

print_status "Systemd service file created (optional)"

# Start the server
print_success "🎉 VuenCode Phase 2 deployment completed successfully!"
echo "======================================================="
echo "🚀 Starting VuenCode server..."
echo "📊 Monitor logs: tail -f logs/app.log"
echo "🔍 Health check: curl http://localhost:$PORT/health"
echo "🎯 API endpoint: http://localhost:$PORT/infer"
echo "======================================================="

# Start the server with proper environment
exec $PYTHON_CMD -m uvicorn VuenCode.api.main:app --host 0.0.0.0 --port $PORT --workers 1