# VuenCode Phase 2 - Remote GPU Deployment Guide

## Server Information
- **OS**: Ubuntu (minimized)
- **Server Name**: vuencode
- **Ngrok Token**: `318E2nqW697Tr4Leg6cJgLETIXD_oFmZ4zyvNdVwhs8c1JxL`
- **GPU**: CUDA-enabled (assumed)

## Deployment Steps

### 1. Initial Server Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y git curl wget python3-pip python3-venv htop nvidia-smi

# Verify GPU availability
nvidia-smi

# Install Docker (for containerized deployment)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Clone VuenCode Repository
```bash
# Clone the repository
git clone https://github.com/KarthikChapa/VuenCode-2.git
cd VuenCode-2

# Make deployment script executable
chmod +x deploy/deploy_gpu.sh
chmod +x deploy/setup_ngrok.sh
```

### 3. Install Ngrok
```bash
# Download and install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Authenticate ngrok
ngrok config add-authtoken 318E2nqW697Tr4Leg6cJgLETIXD_oFmZ4zyvNdVwhs8c1JxL
```

### 4. Environment Setup
```bash
# Set up environment variables
export GOOGLE_API_KEY="your_gemini_api_key_here"
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VISIBLE_DEVICES=0

# Create GPU environment configuration
cp configs/competition.env .env
```

### 5. Deploy VuenCode
```bash
# Option A: Direct Python deployment
./deploy/deploy_gpu.sh

# Option B: Docker deployment (recommended)
docker-compose -f docker/docker-compose-gpu.yml up -d

# Start ngrok tunnel
./deploy/setup_ngrok.sh
```

### 6. Verify Deployment
```bash
# Check service status
curl http://localhost:8000/health

# Test with ngrok URL
curl https://your-ngrok-url.ngrok.io/health
```

## Performance Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check logs
docker logs vuencode-gpu
# or
tail -f logs/vuencode.log
```

## Troubleshooting
- If CUDA not detected: Check nvidia-smi and CUDA installation
- If port conflicts: Modify port in docker-compose-gpu.yml
- If ngrok fails: Verify auth token and internet connection
- If OOM errors: Adjust batch sizes in competition.env

## Security Notes
- Ngrok tunnel is public - ensure proper authentication
- Monitor resource usage to prevent abuse
- Consider IP whitelisting for production use