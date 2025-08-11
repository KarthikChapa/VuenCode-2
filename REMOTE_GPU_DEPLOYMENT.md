# VuenCode Phase 2 - Remote GPU Server Deployment Guide

## Current SSH Connection
```bash
ssh -i C:\Users\karth\.ssh\vuencode_team08_id_ed25519 -p 44152 root@38.128.232.8
```

## Step-by-Step Deployment on Remote GPU Server

### 1. System Setup and Prerequisites
```bash
# Update system
apt update && apt upgrade -y

# Install essential packages
apt install -y curl wget git python3-pip python3-venv htop nvidia-smi unzip

# Verify GPU availability
nvidia-smi

# Check Python version
python3 --version
```

### 2. Install Ngrok
```bash
# Download and install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list
apt update && apt install ngrok

# Set up ngrok authentication
ngrok config add-authtoken 318E2nqW697Tr4Leg6cJgLETIXD_oFmZ4zyvNdVwhs8c1JxL

# Verify ngrok installation
ngrok version
```

### 3. Clone VuenCode Repository
```bash
# Clone the repository
git clone https://github.com/KarthikChapa/VuenCode-2.git
cd VuenCode-2

# Check repository contents
ls -la
```

### 4. Environment Setup
```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r docker/requirements-local.txt

# Install GPU-specific packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper
```

### 5. Configure Environment Variables
```bash
# Create environment file
cp configs/competition.env .env

# Edit environment file (add your Gemini API key)
nano .env

# Add this line to .env:
# export GOOGLE_API_KEY="your_gemini_api_key_here"

# Set GPU environment variables
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
```

### 6. Start VuenCode Server
```bash
# Make sure we're in the VuenCode-2 directory
cd /root/VuenCode-2

# Source environment variables
source .env

# Create logs directory
mkdir -p logs

# Start the server in background
nohup python3 -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 > logs/vuencode.log 2>&1 &

# Get the process ID
echo $! > logs/vuencode.pid

# Wait for server to start
sleep 10

# Test server locally
curl http://localhost:8000/health
```

### 7. Start Ngrok Tunnel
```bash
# Start ngrok tunnel with authentication
nohup ngrok http 8000 --basic-auth "competition:vuencode2024" > logs/ngrok.log 2>&1 &

# Get ngrok process ID
echo $! > logs/ngrok.pid

# Wait for tunnel to establish
sleep 5

# Get the public URL
curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnels = data.get('tunnels', [])
    for tunnel in tunnels:
        print('Public URL:', tunnel['public_url'])
        print('Health Check:', tunnel['public_url'] + '/health')
        print('Inference Endpoint:', tunnel['public_url'] + '/infer')
        break
except:
    print('Error getting tunnel URL')
"
```

### 8. Test Public Endpoint
```bash
# Get the public URL (replace with actual URL from step 7)
PUBLIC_URL="https://your-ngrok-url.ngrok.io"

# Test health endpoint
curl -u competition:vuencode2024 $PUBLIC_URL/health

# Test with a sample video (optional)
curl -u competition:vuencode2024 -X POST $PUBLIC_URL/infer \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"}'
```

### 9. Monitoring and Logs
```bash
# View VuenCode logs
tail -f logs/vuencode.log

# View ngrok logs
tail -f logs/ngrok.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check running processes
ps aux | grep python
ps aux | grep ngrok

# View ngrok web interface (if port forwarding is set up)
# http://localhost:4040
```

### 10. Stopping Services
```bash
# Stop VuenCode server
kill $(cat logs/vuencode.pid)

# Stop ngrok tunnel
kill $(cat logs/ngrok.pid)

# Or stop all related processes
pkill -f "uvicorn api.main:app"
pkill -f "ngrok http"
```

## Quick Deployment Script
For convenience, you can also use the deployment script:

```bash
# Make deployment script executable
chmod +x deploy/deploy_gpu.sh
chmod +x deploy/setup_ngrok.sh

# Run deployment
./deploy/deploy_gpu.sh

# In a separate terminal, setup ngrok
./deploy/setup_ngrok.sh
```

## Competition Submission
Once deployed, you'll get a public URL like:
- **Health Check**: `https://xxxx.ngrok.io/health`
- **Inference**: `https://xxxx.ngrok.io/infer`
- **Authentication**: Username: `competition`, Password: `vuencode2024`

Submit the ngrok URL to the competition platform.

## Troubleshooting

### Server won't start:
```bash
# Check logs
tail -20 logs/vuencode.log

# Check if port is in use
netstat -tlnp | grep 8000

# Kill existing processes
pkill -f uvicorn
```

### Ngrok issues:
```bash
# Check ngrok status
curl http://localhost:4040/api/tunnels

# Restart ngrok
pkill -f ngrok
ngrok http 8000 --basic-auth "competition:vuencode2024" &
```

### GPU not detected:
```bash
# Check CUDA
nvidia-smi
nvcc --version

# Check PyTorch GPU support
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"