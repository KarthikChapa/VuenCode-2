# VuenCode Phase 2 - Remote GPU Deployment Guide

## 🚀 Quick Start - One-Click Deployment

### Step 1: Deploy to Remote GPU Server

```bash
# On your remote GPU server, run this single command:
curl -sSL https://raw.githubusercontent.com/KarthikChapa/VuenCode-2/master/deploy/remote_deploy.sh | bash
```

This will:
- ✅ Clone the latest VuenCode repository
- ✅ Set up Python virtual environment
- ✅ Install all dependencies with GPU support
- ✅ Configure environment variables
- ✅ Test system functionality
- ✅ Start the VuenCode server on port 8000

### Step 2: Set up Public Access (Optional)

```bash
# Inside the VuenCode directory, run:
./deploy/setup_ngrok.sh
```

This will:
- ✅ Install and configure ngrok
- ✅ Create secure public tunnel
- ✅ Provide public API endpoints
- ✅ Generate test scripts

---

## 📋 Manual Deployment Instructions

If you prefer manual deployment, follow these steps:

### Prerequisites
- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPU with CUDA support (recommended)
- Python 3.11+ 
- Git
- 8GB+ RAM, 20GB+ disk space

### 1. Clone Repository
```bash
git clone https://github.com/KarthikChapa/VuenCode-2.git
cd VuenCode-2
```

### 2. Set up Python Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies
```bash
# For GPU systems with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only systems
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r docker/requirements-local.txt
```

### 4. Configure Environment
```bash
# Create required directories
mkdir -p logs cache data/videos temp

# Set environment variables
export DEPLOYMENT_MODE=gpu
export USE_GPU_ACCELERATION=true
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 5. Start VuenCode Server
```bash
python -m uvicorn VuenCode.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## 🌐 Public Endpoint Setup

### Option 1: Using Ngrok (Recommended for Testing)

1. **Install ngrok:**
   ```bash
   # Linux
   wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
   tar xzf ngrok-v3-stable-linux-amd64.tgz
   sudo mv ngrok /usr/local/bin/
   
   # macOS
   brew install ngrok/ngrok/ngrok
   ```

2. **Authenticate ngrok:**
   ```bash
   ngrok config add-authtoken YOUR_TOKEN_HERE
   ```
   Get your token from: https://dashboard.ngrok.com/get-started/your-authtoken

3. **Start tunnel:**
   ```bash
   # Run our automated script
   ./deploy/setup_ngrok.sh
   
   # Or manually
   ngrok http 8000
   ```

### Option 2: Using Cloud Provider Load Balancer

For production deployments, configure your cloud provider's load balancer:

**AWS Application Load Balancer:**
- Target Group: Port 8000
- Health Check: `/health`
- Protocol: HTTP/HTTPS

**Google Cloud Load Balancer:**
- Backend Service: Port 8000
- Health Check: `/health`
- Protocol: HTTP(S)

**Azure Application Gateway:**
- Backend Pool: Port 8000
- Health Probe: `/health`
- Protocol: HTTP/HTTPS

---

## 📊 API Endpoints

Once deployed, your VuenCode API will be available at:

### Health Check
```bash
GET /health

# Response
{
  "status": "healthy",
  "deployment_mode": "gpu",
  "capabilities": {
    "vst_compression": true,
    "multimodal_fusion": true,
    "audio_processing": true
  },
  "performance_summary": {
    "avg_latency_ms": 420.5,
    "p95_latency_ms": 680.2,
    "success_rate": 0.995
  }
}
```

### Video Inference
```bash
POST /infer
Content-Type: application/json

{
  "video_url": "https://example.com/video.mp4",
  "query": "What are the main activities in this video?",
  "category": "action_recognition",
  "quality": "auto",
  "max_frames": 32
}

# Response (Plain Text)
Video Summary with Timeline:
1. 00:00:00 – Initial scene setup with primary subjects positioned in frame
2. 00:00:05 – Main activity begins with object interaction and movement
3. 00:00:15 – Progressive action development with environmental changes
4. 00:00:25 – Conclusion of sequence with final positioning
```

---

## 🧪 Testing Your Deployment

### 1. Quick Health Check
```bash
curl http://your-endpoint/health
```

### 2. Test Video Analysis
```bash
curl -X POST "http://your-endpoint/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
    "query": "Describe what happens in this video",
    "quality": "auto"
  }'
```

### 3. Performance Test
```bash
# Test with multiple concurrent requests
for i in {1..5}; do
  curl -X POST "http://your-endpoint/infer" \
    -H "Content-Type: application/json" \
    -d '{"video_url":"https://example.com/video.mp4","query":"Test"}' &
done
wait
```

---

## 🔧 Configuration Options

### Environment Variables
```bash
# Deployment Configuration
DEPLOYMENT_MODE=gpu                    # gpu, local, competition
USE_GPU_ACCELERATION=true             # Enable GPU processing
CUDA_VISIBLE_DEVICES=0                # GPU device ID
TARGET_LATENCY_MS=400                 # Target response time

# API Configuration  
API_HOST=0.0.0.0                      # Server host
API_PORT=8000                         # Server port
MAX_FRAMES_PER_VIDEO=64               # Frame extraction limit

# Gemini API (update with your keys)
GEMINI_API_KEY=your_key_here          # Required for real AI processing
GEMINI_FLASH_MODEL=gemini-2.0-flash-exp
GEMINI_PRO_MODEL=gemini-1.5-pro

# Performance Tuning
GPU_MEMORY_OPTIMIZATION=true          # Enable memory optimization
BATCH_SIZE=4                          # Processing batch size
ENABLE_CACHING=true                   # Enable response caching
```

### Quality Settings
- `fast`: Minimal processing, <200ms latency
- `balanced`: Good quality/speed trade-off, <400ms latency
- `high`: Maximum quality, <800ms latency
- `auto`: Intelligent selection based on query complexity

---

## 📈 Performance Monitoring

### Built-in Metrics
```bash
GET /metrics
```

Returns comprehensive performance data:
- Request latency (avg, p95, p99)
- Throughput (requests/second)  
- Error rates and success rates
- GPU utilization and memory usage
- Model usage statistics (Flash vs Pro)

### Log Files
- `logs/server.log` - Application logs
- `logs/ngrok.log` - Tunnel logs (if using ngrok)
- `logs/performance.json` - Performance metrics

### Monitoring Dashboard
If using ngrok, access the dashboard at:
- http://localhost:4040 - Ngrok tunnel dashboard

---

## 🔒 Security Considerations

### For Production Deployments:

1. **API Authentication:**
   ```bash
   # Add API key validation (recommended)
   export API_KEY_REQUIRED=true
   export VALID_API_KEYS="key1,key2,key3"
   ```

2. **Rate Limiting:**
   ```bash
   # Configure rate limits
   export RATE_LIMIT_REQUESTS=100        # Requests per minute
   export RATE_LIMIT_CONCURRENT=10       # Concurrent requests
   ```

3. **Network Security:**
   - Use HTTPS only
   - Configure firewall rules
   - Use VPN or private networks
   - Enable CORS properly

4. **Content Filtering:**
   ```bash
   export ENABLE_CONTENT_FILTER=true     # Filter inappropriate content
   export MAX_VIDEO_SIZE_MB=100          # Limit video size
   ```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Not Available**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**2. Import Errors**
```bash
# Verify PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Test imports
python -c "from VuenCode.api.main import app; print('✅ Import successful')"
```

**3. Port Already in Use**
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process if needed
sudo kill -9 <PID>

# Or use different port
python -m uvicorn VuenCode.api.main:app --host 0.0.0.0 --port 8001
```

**4. Memory Issues**
```bash
# Enable memory optimization
export GPU_MEMORY_OPTIMIZATION=true

# Reduce batch size
export BATCH_SIZE=1

# Limit max frames
export MAX_FRAMES_PER_VIDEO=32
```

**5. Slow Performance**
```bash
# Check GPU utilization
nvidia-smi

# Enable performance mode
export PERFORMANCE_MODE=competition
export TARGET_LATENCY_MS=200

# Use Flash model only
export FORCE_FLASH_MODEL=true
```

### Logs and Debugging
```bash
# View real-time logs
tail -f logs/server.log

# Check system resources
htop
nvidia-smi -l 1

# Test specific endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:8000/metrics
```

---

## 📞 Support

### Getting Help
- 📧 Create GitHub issue for bugs/features
- 💬 Check logs in `logs/` directory
- 🔍 Review error messages carefully
- 📊 Check system resources (RAM, GPU, disk)

### Performance Optimization
- Use GPU deployment for best performance
- Enable caching for repeated requests
- Use appropriate quality settings
- Monitor system resources
- Consider load balancing for high traffic

---

## 🎯 Ready for Competition

Your VuenCode Phase 2 system is now deployed with:

✅ **Sub-400ms Latency** - Optimized processing pipeline
✅ **GPU Acceleration** - CUDA-optimized PyTorch models  
✅ **Multimodal Processing** - Video + Audio + Text analysis
✅ **VST Compression** - Efficient long-video handling
✅ **Intelligent Fallback** - Three-tier reliability system
✅ **Competition Endpoints** - `/health` and `/infer` ready
✅ **Performance Monitoring** - Real-time metrics and logging
✅ **Public Access** - Secure tunneling with ngrok

**🚀 Your VuenCode API is competition-ready!**