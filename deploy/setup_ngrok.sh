#!/bin/bash

# VuenCode Phase 2 - Ngrok Tunnel Setup Script
# Create secure tunnel for competition access

set -e

echo "🌐 Setting up Ngrok tunnel for VuenCode Phase 2..."

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

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    print_error "Ngrok is not installed. Please install it first:"
    echo "curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null"
    echo "echo 'deb https://ngrok-agent.s3.amazonaws.com buster main' | sudo tee /etc/apt/sources.list.d/ngrok.list"
    echo "sudo apt update && sudo apt install ngrok"
    exit 1
fi

# Check if VuenCode service is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    print_error "VuenCode service is not running on port 8000"
    print_status "Please start VuenCode first: ./deploy/deploy_gpu.sh"
    exit 1
fi

print_status "VuenCode service detected on port 8000"

# Set ngrok auth token (if not already set)
NGROK_TOKEN="318E2nqW697Tr4Leg6cJgLETIXD_oFmZ4zyvNdVwhs8c1JxL"
print_status "Configuring ngrok auth token..."
ngrok config add-authtoken $NGROK_TOKEN

# Create ngrok config file
print_status "Creating ngrok configuration..."
cat > ~/.ngrok2/ngrok.yml << EOF
version: "2"
authtoken: $NGROK_TOKEN
tunnels:
  vuencode:
    proto: http
    addr: 8000
    hostname: 
    bind_tls: true
    inspect: false
    metadata: "VuenCode Phase 2 Competition Endpoint"
    basic_auth:
      - "competition:vuencode2024"
  vuencode-health:
    proto: http
    addr: 8000
    hostname: 
    bind_tls: true
    inspect: false
    metadata: "VuenCode Health Check"
EOF

print_status "Starting ngrok tunnel..."

# Start ngrok in background
nohup ngrok start --all --config ~/.ngrok2/ngrok.yml > logs/ngrok.log 2>&1 &
NGROK_PID=$!
echo $NGROK_PID > logs/ngrok.pid

print_status "Ngrok started with PID: $NGROK_PID"

# Wait for ngrok to establish tunnel
print_status "Waiting for tunnel to establish..."
sleep 10

# Get tunnel URLs
print_status "Retrieving tunnel information..."

# Get public URL from ngrok API
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnels = data.get('tunnels', [])
    for tunnel in tunnels:
        if tunnel.get('name') == 'vuencode':
            print(tunnel['public_url'])
            break
    else:
        print('Tunnel not found')
except:
    print('Error parsing ngrok response')
")

if [[ $NGROK_URL == *"https://"* ]]; then
    print_status "✅ Ngrok tunnel established successfully!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}🌐 VuenCode Phase 2 Competition Endpoints${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${BLUE}Public API Endpoint:${NC}"
    echo "  🔗 $NGROK_URL/infer"
    echo ""
    echo -e "${BLUE}Health Check Endpoint:${NC}"
    echo "  🔗 $NGROK_URL/health"
    echo ""
    echo -e "${BLUE}Authentication:${NC}"
    echo "  Username: competition"
    echo "  Password: vuencode2024"
    echo ""
    echo -e "${BLUE}Example Usage:${NC}"
    echo "  curl -u competition:vuencode2024 $NGROK_URL/health"
    echo "  curl -u competition:vuencode2024 -X POST $NGROK_URL/infer -H 'Content-Type: application/json' -d '{\"video_url\":\"your_video_url\"}'"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    print_status "Tunnel logs: tail -f logs/ngrok.log"
    print_status "To stop tunnel: kill \$(cat logs/ngrok.pid)"
    print_status "Ngrok web interface: http://localhost:4040"
    
    # Save URL to file for reference
    echo "$NGROK_URL" > logs/ngrok_url.txt
    print_status "Public URL saved to: logs/ngrok_url.txt"
    
else
    print_error "❌ Failed to establish ngrok tunnel"
    print_status "Check logs/ngrok.log for details"
    cat logs/ngrok.log
    exit 1
fi

print_status "🚀 VuenCode Phase 2 is now accessible worldwide!"