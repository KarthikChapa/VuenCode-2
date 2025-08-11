# PowerShell script for VuenCode Phase 2 remote GPU deployment
# This script demonstrates the deployment process for Windows users

Write-Host "🚀 VuenCode Phase 2 Remote GPU Deployment Script" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

Write-Host "`n📋 Deployment Steps Overview:" -ForegroundColor Yellow
Write-Host "1. Clone VuenCode-2 repository to remote GPU server"
Write-Host "2. Install Python dependencies and CUDA environment"
Write-Host "3. Configure environment variables and API keys"
Write-Host "4. Start the FastAPI server with GPU acceleration"
Write-Host "5. Set up ngrok tunnel for public endpoint access"

Write-Host "`n🔧 Required Information:" -ForegroundColor Cyan
Write-Host "- Remote GPU server SSH access (username@hostname)"
Write-Host "- Gemini API key for video processing"
Write-Host "- Ngrok auth token for public tunneling"

Write-Host "`n📝 Manual Deployment Commands for Remote GPU Server:" -ForegroundColor Magenta
Write-Host "# 1. Connect to your GPU server via SSH"
Write-Host "ssh username@your-gpu-server.com"
Write-Host ""
Write-Host "# 2. Clone the latest VuenCode Phase 2 code"
Write-Host "git clone https://github.com/KarthikChapa/VuenCode-2.git"
Write-Host "cd VuenCode-2"
Write-Host ""
Write-Host "# 3. Run the automated deployment script"
Write-Host "chmod +x deploy/remote_deploy.sh"
Write-Host "./deploy/remote_deploy.sh"
Write-Host ""
Write-Host "# 4. Set up public endpoint with ngrok"
Write-Host "chmod +x deploy/setup_ngrok.sh"
Write-Host "./deploy/setup_ngrok.sh"

Write-Host "`n🌐 Alternative: Using Cloud Platform Deployment" -ForegroundColor Green

# Google Colab deployment
Write-Host "`n🔥 Google Colab Deployment (Recommended):" -ForegroundColor Yellow
$colabCode = @"
# Run this in Google Colab for instant GPU deployment
!git clone https://github.com/KarthikChapa/VuenCode-2.git
%cd VuenCode-2

# Install dependencies
!pip install -r docker/requirements-gpu.txt

# Set environment variables
import os
os.environ['GEMINI_API_KEY'] = 'your-gemini-api-key-here'
os.environ['DEPLOYMENT_MODE'] = 'competition'
os.environ['USE_GPU_ACCELERATION'] = 'true'

# Start the server
!python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Install and setup ngrok for public access
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"🌐 VuenCode Phase 2 Public Endpoint: {public_url}")
"@

Write-Host $colabCode -ForegroundColor White

# Hugging Face Spaces deployment
Write-Host "`n🤗 Hugging Face Spaces Deployment:" -ForegroundColor Yellow
Write-Host "1. Go to https://huggingface.co/spaces"
Write-Host "2. Create a new Space with Gradio SDK and GPU"
Write-Host "3. Upload VuenCode repository files"
Write-Host "4. Add app.py file for Gradio interface"
Write-Host "5. Set environment variables in Space settings"

Write-Host "`n📱 Quick Test Commands:" -ForegroundColor Cyan
Write-Host "# Test health endpoint"
Write-Host "curl https://your-endpoint.ngrok.io/health"
Write-Host ""
Write-Host "# Test inference endpoint"
Write-Host 'curl -X POST https://your-endpoint.ngrok.io/infer -H "Content-Type: application/json" -d "{\"query\": \"What do you see?\", \"video_data\": null}"'

Write-Host "`n✅ Deployment Complete!" -ForegroundColor Green
Write-Host "Your VuenCode Phase 2 system will be running with:" -ForegroundColor White
Write-Host "- 🎯 Competition-grade performance (sub-600ms target)"
Write-Host "- 🚀 Phase 2 capabilities: VST + Multimodal + Audio"
Write-Host "- 🔗 Public API endpoint for competition use"
Write-Host "- 📊 Real-time performance monitoring"

Write-Host "`n🔍 Monitoring & Logs:" -ForegroundColor Cyan
Write-Host "- Health endpoint: /health"
Write-Host "- Metrics endpoint: /metrics (if enabled)"
Write-Host "- Server logs: Check terminal output"
Write-Host "- Performance data: Exported to performance_final.json"

Write-Host "`n🎯 Ready for Competition Submission!" -ForegroundColor Green