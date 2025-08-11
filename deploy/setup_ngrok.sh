#!/bin/bash
# VuenCode Ngrok Tunnel Setup Script
# Creates secure public endpoints for remote access

set -e

echo "🌐 Setting up Ngrok tunnel for VuenCode..."
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
VUENCODE_PORT=8000
NGROK_CONFIG_PATH="$HOME/.ngrok2/ngrok.yml"

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    print_status "Installing ngrok..."
    
    # Download and install ngrok
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Installing ngrok for Linux..."
        wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
        tar xzf ngrok-v3-stable-linux-amd64.tgz
        sudo mv ngrok /usr/local/bin/
        rm ngrok-v3-stable-linux-amd64.tgz
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "Installing ngrok for macOS..."
        brew install ngrok/ngrok/ngrok
    else
        print_error "Unsupported OS. Please install ngrok manually from https://ngrok.com/download"
        exit 1
    fi
    
    print_success "Ngrok installed successfully"
else
    print_success "Ngrok is already installed"
fi

# Check if ngrok is authenticated
if ! ngrok config check &> /dev/null; then
    print_warning "Ngrok not authenticated. Please run: ngrok config add-authtoken YOUR_TOKEN"
    echo "Get your auth token from: https://dashboard.ngrok.com/get-started/your-authtoken"
    
    read -p "Enter your ngrok auth token: " NGROK_TOKEN
    if [ -n "$NGROK_TOKEN" ]; then
        ngrok config add-authtoken "$NGROK_TOKEN"
        print_success "Ngrok authenticated successfully"
    else
        print_error "No auth token provided. Exiting."
        exit 1
    fi
fi

# Create ngrok configuration
print_status "Creating ngrok configuration..."
mkdir -p "$(dirname "$NGROK_CONFIG_PATH")"

cat > "$NGROK_CONFIG_PATH" << 'EOF'
version: '2'
authtoken: YOUR_AUTH_TOKEN_HERE

tunnels:
  vuencode-api:
    proto: http
    addr: 8000
    bind_tls: true
    hostname: vuencode-api.ngrok.io
    inspect: false
  
  vuencode-health:
    proto: http
    addr: 8000
    bind_tls: true
    subdomain: vuencode-health
    inspect: true

log_level: info
log_format: term
EOF

print_success "Ngrok configuration created"

# Function to start VuenCode server in background
start_vuencode_server() {
    print_status "Starting VuenCode server on port $VUENCODE_PORT..."
    
    cd "$(dirname "$0")/.."
    source venv/bin/activate
    
    export DEPLOYMENT_MODE=gpu
    export USE_GPU_ACCELERATION=true
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    # Start server in background
    nohup python -m uvicorn VuenCode.api.main:app --host 0.0.0.0 --port $VUENCODE_PORT --workers 1 > logs/server.log 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    print_status "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s "http://localhost:$VUENCODE_PORT/health" > /dev/null 2>&1; then
            print_success "VuenCode server started successfully (PID: $SERVER_PID)"
            echo $SERVER_PID > server.pid
            return 0
        fi
        sleep 2
        echo -n "."
    done
    
    print_error "Server failed to start within 60 seconds"
    return 1
}

# Function to start ngrok tunnel
start_ngrok_tunnel() {
    print_status "Starting ngrok tunnel..."
    
    # Start ngrok in background
    nohup ngrok start vuencode-api > logs/ngrok.log 2>&1 &
    NGROK_PID=$!
    echo $NGROK_PID > ngrok.pid
    
    # Wait for ngrok to start
    print_status "Waiting for ngrok tunnel to establish..."
    sleep 10
    
    # Get tunnel information
    TUNNEL_INFO=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null || echo '{"tunnels":[]}')
    
    if echo "$TUNNEL_INFO" | jq -e '.tunnels[0]' > /dev/null 2>&1; then
        PUBLIC_URL=$(echo "$TUNNEL_INFO" | jq -r '.tunnels[0].public_url')
        LOCAL_URL=$(echo "$TUNNEL_INFO" | jq -r '.tunnels[0].config.addr')
        
        print_success "Ngrok tunnel established successfully!"
        echo "======================================================"
        echo "🌐 PUBLIC API ENDPOINT: $PUBLIC_URL"
        echo "🏠 LOCAL ENDPOINT: http://localhost:$VUENCODE_PORT"
        echo "======================================================"
        echo "📊 Health Check: $PUBLIC_URL/health"
        echo "🎯 Inference API: $PUBLIC_URL/infer"
        echo "🔍 Ngrok Dashboard: http://localhost:4040"
        echo "======================================================"
        
        # Save endpoint information
        cat > endpoint_info.json << EOF
{
  "public_url": "$PUBLIC_URL",
  "local_url": "http://localhost:$VUENCODE_PORT",
  "health_endpoint": "$PUBLIC_URL/health",
  "inference_endpoint": "$PUBLIC_URL/infer",
  "ngrok_dashboard": "http://localhost:4040",
  "timestamp": "$(date -Iseconds)",
  "server_pid": "$SERVER_PID",
  "ngrok_pid": "$NGROK_PID"
}
EOF
        
        print_success "Endpoint information saved to endpoint_info.json"
        
        # Test the endpoints
        print_status "Testing endpoints..."
        
        # Test health endpoint
        if curl -s "$PUBLIC_URL/health" > /dev/null; then
            print_success "✅ Health endpoint is responding"
        else
            print_warning "⚠️  Health endpoint test failed"
        fi
        
        return 0
    else
        print_error "Failed to establish ngrok tunnel"
        return 1
    fi
}

# Function to create test script
create_test_script() {
    cat > test_api.sh << 'EOF'
#!/bin/bash
# VuenCode API Test Script

ENDPOINT_INFO="endpoint_info.json"

if [ ! -f "$ENDPOINT_INFO" ]; then
    echo "❌ Endpoint info not found. Run setup_ngrok.sh first."
    exit 1
fi

PUBLIC_URL=$(jq -r '.public_url' "$ENDPOINT_INFO")
HEALTH_URL=$(jq -r '.health_endpoint' "$ENDPOINT_INFO")
INFERENCE_URL=$(jq -r '.inference_endpoint' "$ENDPOINT_INFO")

echo "🧪 Testing VuenCode API endpoints..."
echo "===================================="

# Test health endpoint
echo "🔍 Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s "$HEALTH_URL")
if [ $? -eq 0 ]; then
    echo "✅ Health endpoint: OK"
    echo "$HEALTH_RESPONSE" | jq .
else
    echo "❌ Health endpoint: Failed"
fi

echo ""
echo "🎯 Inference endpoint: $INFERENCE_URL"
echo "📊 Use this for video analysis requests"
echo ""
echo "Example curl command:"
echo "curl -X POST '$INFERENCE_URL' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"video_url\": \"https://example.com/video.mp4\","
echo "    \"query\": \"What happens in this video?\","
echo "    \"quality\": \"auto\""
echo "  }'"
EOF

    chmod +x test_api.sh
    print_success "Test script created: test_api.sh"
}

# Function to create stop script
create_stop_script() {
    cat > stop_services.sh << 'EOF'
#!/bin/bash
# Stop VuenCode services

echo "🛑 Stopping VuenCode services..."

# Stop server
if [ -f server.pid ]; then
    SERVER_PID=$(cat server.pid)
    if kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID
        echo "✅ VuenCode server stopped (PID: $SERVER_PID)"
    fi
    rm -f server.pid
fi

# Stop ngrok
if [ -f ngrok.pid ]; then
    NGROK_PID=$(cat ngrok.pid)
    if kill -0 $NGROK_PID 2>/dev/null; then
        kill $NGROK_PID
        echo "✅ Ngrok tunnel stopped (PID: $NGROK_PID)"
    fi
    rm -f ngrok.pid
fi

# Clean up
rm -f endpoint_info.json

echo "🎉 All services stopped successfully"
EOF

    chmod +x stop_services.sh
    print_success "Stop script created: stop_services.sh"
}

# Main execution
main() {
    # Create logs directory
    mkdir -p logs
    
    # Start VuenCode server
    if start_vuencode_server; then
        # Start ngrok tunnel
        if start_ngrok_tunnel; then
            # Create utility scripts
            create_test_script
            create_stop_script
            
            print_success "🎉 VuenCode deployment completed successfully!"
            echo ""
            echo "🎯 Your VuenCode API is now publicly accessible!"
            echo "📁 Check endpoint_info.json for all URLs"
            echo "🧪 Run ./test_api.sh to test the API"
            echo "🛑 Run ./stop_services.sh to stop all services"
            echo ""
            echo "🔄 Services will continue running in the background"
            echo "📊 Monitor logs: tail -f logs/server.log"
            echo "🌐 Ngrok dashboard: http://localhost:4040"
            
            # Keep script running to show logs
            echo ""
            echo "Press Ctrl+C to stop monitoring (services will continue running)"
            echo "Real-time server logs:"
            echo "======================================================"
            tail -f logs/server.log
        else
            print_error "Failed to start ngrok tunnel"
            exit 1
        fi
    else
        print_error "Failed to start VuenCode server"
        exit 1
    fi
}

# Handle script interruption
trap 'echo ""; echo "🔄 Services are still running in the background"; echo "🛑 Use ./stop_services.sh to stop them"; exit 0' INT

# Run main function
main