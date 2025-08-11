#!/usr/bin/env python3
"""
VuenCode Phase 2 - Instant Google Colab GPU Deployment
=======================================================

One-click deployment script for Google Colab with GPU acceleration.
Creates public endpoint via ngrok for competition submission.

Usage:
1. Open Google Colab (colab.research.google.com)
2. Change runtime to GPU (Runtime -> Change runtime type -> GPU)
3. Run this script in a single cell
4. Enter your API keys when prompted
5. Get public endpoint URL for competition

Requirements:
- Google Colab with GPU runtime
- Gemini API key (https://ai.google.dev/)
- Ngrok auth token (https://ngrok.com/)
"""

import os
import sys
import time
import subprocess
import threading
import json
from pathlib import Path

def setup_environment():
    """Setup the Colab environment for VuenCode deployment."""
    print("🚀 VuenCode Phase 2 - Instant GPU Deployment")
    print("=" * 50)
    
    # Check GPU availability
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        print("✅ GPU detected and ready")
    except:
        print("⚠️  No GPU detected - performance may be limited")
    
    # Clone repository
    if not Path("VuenCode-2").exists():
        print("📦 Cloning VuenCode Phase 2 repository...")
        subprocess.run(["git", "clone", "https://github.com/KarthikChapa/VuenCode-2.git"], check=True)
    
    os.chdir("VuenCode-2")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "docker/requirements-gpu.txt"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok", "fastapi", "uvicorn", "python-multipart"], check=True)
    
    print("✅ Environment setup complete!")

def configure_api_keys():
    """Configure API keys and environment variables."""
    print("\n🔧 Configuration Setup")
    print("-" * 30)
    
    # Get API keys
    try:
        from getpass import getpass
        api_key = getpass("🔑 Enter your Gemini API key: ")
        ngrok_token = getpass("🌐 Enter your ngrok auth token: ")
    except:
        # Fallback for environments without getpass
        api_key = input("🔑 Enter your Gemini API key: ")
        ngrok_token = input("🌐 Enter your ngrok auth token: ")
    
    # Set environment variables
    env_vars = {
        'GEMINI_API_KEY': api_key,
        'NGROK_AUTH_TOKEN': ngrok_token,
        'DEPLOYMENT_MODE': 'competition',
        'USE_GPU_ACCELERATION': 'true',
        'TARGET_LATENCY_MS': '600',
        'MAX_FRAMES_PER_VIDEO': '32',
        'API_HOST': '0.0.0.0',
        'API_PORT': '8000'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ Configuration complete!")

def start_server():
    """Start the VuenCode FastAPI server."""
    print("\n🚀 Starting VuenCode Phase 2 server...")
    
    def run_server():
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "1"
        ])
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("⏳ Waiting for server startup...")
    time.sleep(15)
    
    # Test local health endpoint
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Server running - Status: {health['status']}")
            print(f"🚀 Phase 2 capabilities: VST={health['capabilities']['vst_compression']}, "
                  f"Multimodal={health['capabilities']['multimodal_fusion']}, "
                  f"Audio={health['capabilities']['audio_processing']}")
        else:
            print(f"⚠️  Server health check failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Server health check error: {e}")
    
    return server_thread

def create_public_endpoint():
    """Create public endpoint using ngrok."""
    print("\n🌐 Creating public endpoint...")
    
    try:
        from pyngrok import ngrok, conf
        
        # Set ngrok auth token
        conf.get_default().auth_token = os.environ['NGROK_AUTH_TOKEN']
        
        # Create public tunnel
        public_tunnel = ngrok.connect(8000)
        public_url = str(public_tunnel.public_url)
        
        print(f"🎯 VuenCode Phase 2 Public Endpoint: {public_url}")
        
        return public_url
        
    except Exception as e:
        print(f"❌ Failed to create public endpoint: {e}")
        return None

def test_endpoints(public_url):
    """Test the competition endpoints."""
    if not public_url:
        return
    
    print(f"\n🧪 Testing competition endpoints...")
    print("=" * 40)
    
    try:
        import requests
        
        # Test health endpoint
        print("📊 Testing health endpoint...")
        response = requests.get(f"{public_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health check passed - {health['status']}")
            process_time = response.headers.get('X-Process-Time-MS', 'N/A')
            print(f"⚡ Response time: {process_time}ms")
        
        # Test inference endpoint
        print("\n🧠 Testing inference endpoint...")
        payload = {
            "query": "What capabilities does this video understanding system have?",
            "video_data": None,
            "category": "general_understanding"
        }
        
        response = requests.post(f"{public_url}/infer", json=payload, timeout=30)
        if response.status_code == 200:
            result = response.text
            print(f"✅ Inference test passed")
            print(f"📝 Response preview: {result[:100]}...")
            process_time = response.headers.get('X-Process-Time-MS', 'N/A')
            print(f"⚡ Response time: {process_time}ms")
        else:
            print(f"⚠️  Inference test failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Endpoint testing failed: {e}")

def main():
    """Main deployment function."""
    try:
        # Setup environment
        setup_environment()
        
        # Configure API keys
        configure_api_keys()
        
        # Start server
        server_thread = start_server()
        
        # Create public endpoint
        public_url = create_public_endpoint()
        
        if public_url:
            # Test endpoints
            test_endpoints(public_url)
            
            # Display final results
            print("\n" + "=" * 60)
            print("🏆 VuenCode Phase 2 Competition Deployment: COMPLETE")
            print("=" * 60)
            print(f"🌐 Public API Endpoint: {public_url}")
            print(f"📊 Health Check: {public_url}/health")
            print(f"🧠 Inference API: {public_url}/infer")
            print("\n🎯 READY FOR COMPETITION SUBMISSION!")
            print(f"📋 Submit this URL: {public_url}")
            print("\n⚡ Server will run as long as this session is active.")
            print("🔄 Keep this Colab notebook open to maintain the endpoint.")
            
            # Keep running
            print("\n🔄 Monitoring endpoints... (Press Ctrl+C to stop)")
            try:
                while True:
                    time.sleep(60)
                    # Optional: Add periodic health checks here
            except KeyboardInterrupt:
                print("\n🛑 Deployment stopped by user")
        else:
            print("❌ Failed to create public endpoint - check your ngrok token")
            
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()