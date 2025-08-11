#!/bin/bash

# VuenCode Phase 2 - One-Command Remote Deployment
# Copy and paste this entire script into your remote GPU server

echo "🚀 VuenCode Phase 2 - One-Command Deployment Starting..."

# Update system and install prerequisites
apt update -qq && apt install -y curl wget git python3-pip python3-venv htop nvidia-smi unzip

# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list
apt update && apt install ngrok

# Set up ngrok auth
ngrok config add-authtoken 318E2nqW697Tr4Leg6cJgLETIXD_oFmZ4zyvNdVwhs8c1JxL

# Clone repository
cd /root
git clone https://github.com/KarthikChapa/VuenCode-2.git
cd VuenCode-2

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r docker/requirements-local.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper

# Set up environment
cp configs/competition.env .env
echo 'export GOOGLE_API_KEY="AIzaSyA4LCrRRlopK-lFxPNBq6WSMFQ3_PacDc8"' >> .env
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
source .env

# Create logs directory
mkdir -p logs

# Start VuenCode server
echo "Starting VuenCode server..."
nohup python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 > logs/vuencode.log 2>&1 &
echo $! > logs/vuencode.pid

# Wait for server to start
sleep 15

# Test server
echo "Testing server..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ VuenCode server is running"
else
    echo "❌ Server failed to start - check logs/vuencode.log"
    exit 1
fi

# Start ngrok tunnel
echo "Starting ngrok tunnel..."
nohup ngrok http 8000 --basic-auth "competition:vuencode2024" > logs/ngrok.log 2>&1 &
echo $! > logs/ngrok.pid

# Wait for tunnel
sleep 10

# Get public URL
echo "Getting public URL..."
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnels = data.get('tunnels', [])
    if tunnels:
        print(tunnels[0]['public_url'])
except:
    print('')
")

if [ ! -z "$PUBLIC_URL" ]; then
    echo ""
    echo "🎉 VuenCode Phase 2 Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "📍 PUBLIC ENDPOINTS:"
    echo "   Health Check: $PUBLIC_URL/health"
    echo "   Video Analysis: $PUBLIC_URL/infer"
    echo ""
    echo "🔐 AUTHENTICATION:"
    echo "   Username: competition"
    echo "   Password: vuencode2024"
    echo ""
    echo "🧪 TEST COMMANDS:"
    echo "   curl -u competition:vuencode2024 $PUBLIC_URL/health"
    echo ""
    echo "🎯 COMPETITION SUBMISSION URL:"
    echo "   $PUBLIC_URL"
    echo ""
    echo "📊 MONITORING:"
    echo "   Server logs: tail -f logs/vuencode.log"
    echo "   Ngrok logs: tail -f logs/ngrok.log"
    echo "   GPU usage: watch -n 1 nvidia-smi"
    echo "   Ngrok dashboard: http://localhost:4040"
    echo ""
    echo "🛑 TO STOP:"
    echo "   kill \$(cat logs/vuencode.pid)"
    echo "   kill \$(cat logs/ngrok.pid)"
    echo ""
    
    # Save endpoint info
    cat > competition_endpoint.txt << EOF
VuenCode Phase 2 Competition Endpoint
====================================
URL: $PUBLIC_URL
Username: competition
Password: vuencode2024
Health: $PUBLIC_URL/health
Infer: $PUBLIC_URL/infer

Test Command:
curl -u competition:vuencode2024 $PUBLIC_URL/health

Deployed: $(date)
EOF
    
    echo "✅ Endpoint details saved to competition_endpoint.txt"
    echo "🚀 Ready for competition submission!"
    
    # Test the endpoint
    echo ""
    echo "Testing public endpoint..."
    if curl -u competition:vuencode2024 -s $PUBLIC_URL/health > /dev/null; then
        echo "✅ Public endpoint is working correctly!"
    else
        echo "⚠️  Public endpoint test failed - check authentication"
    fi
    
else
    echo "❌ Failed to get ngrok public URL"
    echo "Check logs/ngrok.log for details"
fi

echo ""
echo "=========================================="
echo "VuenCode Phase 2 deployment completed!"