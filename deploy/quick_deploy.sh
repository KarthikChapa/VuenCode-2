#!/bin/bash

# VuenCode Phase 2 - One-Click GPU Deployment Script
# Complete deployment with ngrok tunnel setup

set -e

echo "🚀 VuenCode Phase 2 - Quick GPU Deployment Starting..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on GPU server
if [ ! -f "/usr/bin/nvidia-smi" ] && [ ! -f "/usr/local/cuda/bin/nvcc" ]; then
    print_warning "CUDA tools not detected. Continuing with CPU-only deployment."
fi

# Step 1: Update system and install prerequisites
print_status "📦 Installing system prerequisites..."
sudo apt update -qq
sudo apt install -y curl wget git python3-pip python3-venv htop

# Step 2: Install ngrok if not present
if ! command -v ngrok &> /dev/null; then
    print_status "🌐 Installing ngrok..."
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
    sudo apt update && sudo apt install ngrok
fi

# Step 3: Set up ngrok auth token
print_status "🔑 Configuring ngrok authentication..."
if [ -z "$NGROK_AUTHTOKEN" ]; then
    print_error "NGROK_AUTHTOKEN is not set. Please export NGROK_AUTHTOKEN and re-run."
    exit 1
fi
ngrok config add-authtoken "$NGROK_AUTHTOKEN"

# Step 4: Clone repository if not present
if [ ! -d "VuenCode-2" ]; then
    print_status "📥 Cloning VuenCode repository..."
    git clone https://github.com/KarthikChapa/VuenCode-2.git
fi

cd VuenCode-2

# Step 5: Set up environment
print_status "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    cp configs/competition.env .env
    print_warning "Please edit .env file and add your GOOGLE_API_KEY"
    print_status "Example: export GOOGLE_API_KEY=\"your_api_key_here\""
fi

# Step 6: Deploy VuenCode
print_status "🚀 Deploying VuenCode Phase 2..."
chmod +x deploy/deploy_gpu.sh
chmod +x deploy/setup_ngrok.sh
./deploy/deploy_gpu.sh

# Wait for service to start
sleep 10

# Step 7: Setup ngrok tunnel
print_status "🌐 Setting up ngrok tunnel..."
./deploy/setup_ngrok.sh

print_status "✅ VuenCode Phase 2 deployment completed!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎯 VuenCode Phase 2 is now live and accessible worldwide!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
print_status "📋 Next Steps:"
echo "  1. Check logs/ngrok_url.txt for your public URL"
echo "  2. Test the /health endpoint"
echo "  3. Submit your endpoint URL to the competition"
echo ""
print_status "📊 Monitoring:"
echo "  - Service logs: tail -f logs/vuencode.log"
echo "  - Ngrok logs: tail -f logs/ngrok.log"
echo "  - GPU usage: watch -n 1 nvidia-smi"
echo "  - Ngrok dashboard: http://localhost:4040"
echo ""
print_status "🛑 To stop services:"
echo "  - Stop VuenCode: kill \$(cat logs/vuencode.pid)"
echo "  - Stop ngrok: kill \$(cat logs/ngrok.pid)"