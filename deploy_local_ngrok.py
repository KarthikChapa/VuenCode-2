#!/usr/bin/env python3
"""
VuenCode Phase 2 - Local Ngrok Deployment Script
Deploy VuenCode locally with ngrok tunnel for competition submission
"""

import subprocess
import sys
import time
import requests
import json
import os
from pathlib import Path

def print_status(message):
    print(f"[INFO] {message}")

def print_error(message):
    print(f"[ERROR] {message}")

def print_success(message):
    print(f"[SUCCESS] ✅ {message}")

def check_ngrok_installed():
    """Check if ngrok is installed"""
    try:
        subprocess.run(["ngrok", "version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ngrok_windows():
    """Install ngrok on Windows"""
    print_status("Installing ngrok for Windows...")
    
    # Download ngrok
    import urllib.request
    import zipfile
    
    ngrok_url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
    ngrok_zip = "ngrok.zip"
    
    print_status("Downloading ngrok...")
    urllib.request.urlretrieve(ngrok_url, ngrok_zip)
    
    print_status("Extracting ngrok...")
    with zipfile.ZipFile(ngrok_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Clean up
    os.remove(ngrok_zip)
    print_success("Ngrok installed successfully")

def setup_ngrok_auth():
    """Set up ngrok authentication"""
    token = "318E2nqW697Tr4Leg6cJgLETIXD_oFmZ4zyvNdVwhs8c1JxL"
    
    try:
        result = subprocess.run(
            ["ngrok", "config", "add-authtoken", token],
            capture_output=True,
            text=True,
            check=True
        )
        print_success("Ngrok authentication configured")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to configure ngrok auth: {e}")
        return False

def start_vuencode_server():
    """Start VuenCode server"""
    print_status("Starting VuenCode Phase 2 server...")
    
    # Start server in background
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    print_status("Waiting for server to start...")
    time.sleep(10)
    
    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print_success("VuenCode server is running")
            return process
        else:
            print_error("Server started but health check failed")
            return None
    except requests.RequestException:
        print_error("Failed to connect to server")
        return None

def start_ngrok_tunnel():
    """Start ngrok tunnel"""
    print_status("Starting ngrok tunnel...")
    
    # Start ngrok in background
    process = subprocess.Popen([
        "ngrok", "http", "8000",
        "--basic-auth", "competition:vuencode2024"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for tunnel to establish
    time.sleep(5)
    
    # Get tunnel URL
    try:
        response = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=5)
        if response.status_code == 200:
            tunnels = response.json()["tunnels"]
            if tunnels:
                public_url = tunnels[0]["public_url"]
                print_success(f"Ngrok tunnel established: {public_url}")
                return process, public_url
    except requests.RequestException:
        pass
    
    print_error("Failed to get ngrok tunnel URL")
    return process, None

def test_public_endpoint(public_url):
    """Test the public endpoint"""
    print_status("Testing public endpoint...")
    
    # Test health endpoint
    health_url = f"{public_url}/health"
    try:
        response = requests.get(
            health_url,
            auth=("competition", "vuencode2024"),
            timeout=10
        )
        
        if response.status_code == 200:
            health_data = response.json()
            print_success("Health endpoint test passed")
            print(f"Response: {json.dumps(health_data, indent=2)}")
            return True
        else:
            print_error(f"Health endpoint test failed: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print_error(f"Failed to test health endpoint: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 VuenCode Phase 2 - Local Ngrok Deployment")
    print("=" * 50)
    
    # Check if ngrok is installed
    if not check_ngrok_installed():
        print_status("Ngrok not found, installing...")
        install_ngrok_windows()
    else:
        print_success("Ngrok is already installed")
    
    # Set up ngrok authentication
    if not setup_ngrok_auth():
        print_error("Failed to set up ngrok authentication")
        return
    
    # Start VuenCode server
    server_process = start_vuencode_server()
    if not server_process:
        print_error("Failed to start VuenCode server")
        return
    
    try:
        # Start ngrok tunnel
        ngrok_process, public_url = start_ngrok_tunnel()
        
        if not public_url:
            print_error("Failed to establish ngrok tunnel")
            return
        
        # Test public endpoint
        if test_public_endpoint(public_url):
            print("\n" + "=" * 70)
            print("🎉 VuenCode Phase 2 - Competition Deployment Complete!")
            print("=" * 70)
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
            print(f"\n🎯 COMPETITION SUBMISSION:")
            print(f"   Submit this URL: {public_url}")
            print(f"\n📊 MONITORING:")
            print(f"   Ngrok Dashboard: http://localhost:4040")
            print(f"   Local Server: http://localhost:8000")
            print("\n💡 Press Ctrl+C to stop all services")
            print("=" * 70)
            
            # Save URL to file
            with open("competition_endpoint.txt", "w") as f:
                f.write(f"VuenCode Phase 2 Competition Endpoint\n")
                f.write(f"=====================================\n")
                f.write(f"URL: {public_url}\n")
                f.write(f"Username: competition\n")
                f.write(f"Password: vuencode2024\n")
                f.write(f"Health: {public_url}/health\n")
                f.write(f"Infer: {public_url}/infer\n")
            
            print_success("Endpoint details saved to competition_endpoint.txt")
            
            # Keep services running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\nShutting down services...")
                
        else:
            print_error("Endpoint test failed")
            
    finally:
        # Clean up processes
        try:
            server_process.terminate()
            ngrok_process.terminate()
        except:
            pass
        
        print_success("Services stopped")

if __name__ == "__main__":
    main()