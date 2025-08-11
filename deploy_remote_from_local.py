#!/usr/bin/env python3
"""
VuenCode Phase 2 - Remote GPU Deployment from Local Machine
Deploy VuenCode on remote GPU server via SSH and get public endpoint
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path

def print_status(message):
    print(f"[INFO] {message}")

def print_error(message):
    print(f"[ERROR] {message}")

def print_success(message):
    print(f"[SUCCESS] ✅ {message}")

# Remote server configuration
REMOTE_HOST = "38.128.232.8"
REMOTE_PORT = "44152"
SSH_KEY = r"C:\Users\karth\.ssh\vuencode_team08_id_ed25519"
NGROK_TOKEN = "318E2nqW697Tr4Leg6cJgLETIXD_oFmZ4zyvNdVwhs8c1JxL"

def run_ssh_command(command, capture_output=True):
    """Run command on remote server via SSH"""
    ssh_cmd = [
        "ssh", 
        "-i", SSH_KEY,
        "-p", REMOTE_PORT,
        "-o", "StrictHostKeyChecking=no",
        f"root@{REMOTE_HOST}",
        command
    ]
    
    try:
        if capture_output:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(ssh_cmd, timeout=300)
            return result.returncode, "", ""
    except subprocess.TimeoutExpired:
        print_error("Command timed out")
        return 1, "", "Timeout"
    except Exception as e:
        print_error(f"SSH command failed: {e}")
        return 1, "", str(e)

def deploy_vuencode_remote():
    """Deploy VuenCode on remote server"""
    
    print("🚀 VuenCode Phase 2 - Remote GPU Deployment from Local")
    print("=" * 60)
    
    # Step 1: Test SSH connection
    print_status("Testing SSH connection to remote GPU server...")
    returncode, stdout, stderr = run_ssh_command("echo 'SSH connection successful'")
    
    if returncode != 0:
        print_error(f"SSH connection failed: {stderr}")
        return None
    
    print_success("SSH connection established")
    
    # Step 2: Check if deployment already exists
    print_status("Checking for existing deployment...")
    returncode, stdout, stderr = run_ssh_command("ls /root/VuenCode-2")
    
    if returncode == 0:
        print_status("Existing deployment found, updating...")
        # Pull latest changes
        run_ssh_command("cd /root/VuenCode-2 && git pull origin master")
    else:
        print_status("No existing deployment, performing fresh install...")
        
        # Step 3: System setup
        print_status("Setting up system prerequisites...")
        setup_cmd = """
        apt update -qq && 
        apt install -y curl wget git python3-pip python3-venv htop nvidia-smi unzip
        """
        run_ssh_command(setup_cmd)
        
        # Step 4: Install ngrok
        print_status("Installing ngrok...")
        ngrok_cmd = f"""
        curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null &&
        echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list &&
        apt update && apt install ngrok &&
        ngrok config add-authtoken {NGROK_TOKEN}
        """
        run_ssh_command(ngrok_cmd)
        
        # Step 5: Clone repository
        print_status("Cloning VuenCode repository...")
        clone_cmd = """
        cd /root &&
        git clone https://github.com/KarthikChapa/VuenCode-2.git &&
        cd VuenCode-2
        """
        run_ssh_command(clone_cmd)
    
    # Step 6: Set up Python environment
    print_status("Setting up Python environment...")
    python_setup_cmd = """
    cd /root/VuenCode-2 &&
    python3 -m venv venv &&
    source venv/bin/activate &&
    pip install --upgrade pip &&
    pip install -r docker/requirements-local.txt &&
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 &&
    pip install openai-whisper
    """
    run_ssh_command(python_setup_cmd)
    
    # Step 7: Configure environment
    print_status("Configuring environment...")
    
    # Ask for Gemini API key
    print("\n" + "="*50)
    gemini_key = input("Enter your Gemini API Key: ").strip()
    print("="*50)
    
    env_setup_cmd = f"""
    cd /root/VuenCode-2 &&
    cp configs/competition.env .env &&
    echo 'export GOOGLE_API_KEY="{gemini_key}"' >> .env &&
    export CUDA_VISIBLE_DEVICES=0 &&
    export NVIDIA_VISIBLE_DEVICES=all
    """
    run_ssh_command(env_setup_cmd)
    
    # Step 8: Stop any existing services
    print_status("Stopping any existing services...")
    cleanup_cmd = """
    pkill -f "uvicorn api.main:app" || true &&
    pkill -f "ngrok http" || true &&
    sleep 3
    """
    run_ssh_command(cleanup_cmd)
    
    # Step 9: Start VuenCode server
    print_status("Starting VuenCode Phase 2 server...")
    server_cmd = f"""
    cd /root/VuenCode-2 &&
    source venv/bin/activate &&
    source .env &&
    mkdir -p logs &&
    nohup venv/bin/python api/main_standalone.py > logs/vuencode.log 2>&1 &
    echo $! > logs/vuencode.pid
    """
    run_ssh_command(server_cmd)
    
    # Step 10: Wait for server to start
    print_status("Waiting for server to initialize...")
    time.sleep(15)
    
    # Check server health
    print_status("Testing server health...")
    health_cmd = "cd /root/VuenCode-2 && curl -s http://localhost:8000/health"
    returncode, stdout, stderr = run_ssh_command(health_cmd)
    
    if returncode != 0:
        print_error("Server health check failed")
        # Get logs
        log_cmd = "cd /root/VuenCode-2 && tail -20 logs/vuencode.log"
        _, log_output, _ = run_ssh_command(log_cmd)
        print(f"Server logs:\n{log_output}")
        return None
    
    print_success("VuenCode server is running")
    
    # Step 11: Start ngrok tunnel
    print_status("Starting ngrok tunnel...")
    ngrok_cmd = """
    cd /root/VuenCode-2 &&
    nohup ngrok http 8000 --basic-auth "competition:vuencode2024" > logs/ngrok.log 2>&1 &
    echo $! > logs/ngrok.pid
    """
    run_ssh_command(ngrok_cmd)
    
    # Step 12: Wait for tunnel to establish
    print_status("Waiting for ngrok tunnel to establish...")
    time.sleep(10)
    
    # Step 13: Get public URL
    print_status("Retrieving public endpoint...")
    get_url_cmd = """
    curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnels = data.get('tunnels', [])
    if tunnels:
        print(tunnels[0]['public_url'])
    else:
        print('NO_TUNNEL')
except:
    print('ERROR')
"
    """
    
    returncode, public_url, stderr = run_ssh_command(get_url_cmd)
    
    if returncode != 0 or not public_url.strip() or public_url.strip() in ['NO_TUNNEL', 'ERROR']:
        print_error("Failed to get ngrok tunnel URL")
        # Get ngrok logs
        log_cmd = "cd /root/VuenCode-2 && tail -20 logs/ngrok.log"
        _, log_output, _ = run_ssh_command(log_cmd)
        print(f"Ngrok logs:\n{log_output}")
        return None
    
    public_url = public_url.strip()
    print_success(f"Public endpoint established: {public_url}")
    
    return public_url

def test_public_endpoint(public_url):
    """Test the public endpoint from local machine"""
    print_status("Testing public endpoint from local machine...")
    
    try:
        import requests
        
        # Test health endpoint
        health_url = f"{public_url}/health"
        response = requests.get(
            health_url,
            auth=("competition", "vuencode2024"),
            timeout=30
        )
        
        if response.status_code == 200:
            health_data = response.json()
            print_success("Health endpoint test passed!")
            print(f"Response: {json.dumps(health_data, indent=2)}")
            return True
        else:
            print_error(f"Health endpoint test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Failed to test endpoint: {e}")
        return False

def main():
    """Main deployment function"""
    
    # Check if SSH key exists
    if not os.path.exists(SSH_KEY):
        print_error(f"SSH key not found: {SSH_KEY}")
        return
    
    # Deploy to remote server
    public_url = deploy_vuencode_remote()
    
    if not public_url:
        print_error("Deployment failed")
        return
    
    # Test the endpoint
    if test_public_endpoint(public_url):
        print("\n" + "="*70)
        print("🎉 VuenCode Phase 2 - Remote Deployment Complete!")
        print("="*70)
        print(f"\n📍 PUBLIC ENDPOINTS:")
        print(f"   Health Check: {public_url}/health")
        print(f"   Video Analysis: {public_url}/infer")
        print(f"\n🔐 AUTHENTICATION:")
        print(f"   Username: competition")
        print(f"   Password: vuencode2024")
        print(f"\n🧪 TEST COMMANDS:")
        print(f"   curl -u competition:vuencode2024 {public_url}/health")
        print(f"   curl -u competition:vuencode2024 -X POST {public_url}/infer \\")
        print(f"        -H 'Content-Type: application/json' \\")
        print(f"        -d '{{\"video_url\":\"your_test_video_url\"}}'")
        print(f"\n🎯 COMPETITION SUBMISSION URL:")
        print(f"   {public_url}")
        print(f"\n📊 REMOTE MONITORING:")
        print(f"   SSH: ssh -i {SSH_KEY} -p {REMOTE_PORT} root@{REMOTE_HOST}")
        print(f"   Logs: tail -f /root/VuenCode-2/logs/vuencode.log")
        print(f"   GPU: watch -n 1 nvidia-smi")
        print("="*70)
        
        # Save endpoint details locally
        with open("competition_endpoint.txt", "w") as f:
            f.write(f"VuenCode Phase 2 Competition Endpoint\n")
            f.write(f"=====================================\n")
            f.write(f"URL: {public_url}\n")
            f.write(f"Username: competition\n")
            f.write(f"Password: vuencode2024\n")
            f.write(f"Health: {public_url}/health\n")
            f.write(f"Infer: {public_url}/infer\n")
            f.write(f"Deployed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print_success("Endpoint details saved to competition_endpoint.txt")
        print("🚀 Ready for competition submission!")
        
    else:
        print_error("Endpoint test failed")

if __name__ == "__main__":
    main()