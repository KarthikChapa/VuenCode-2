#!/usr/bin/env python3
"""
Test script for the competition endpoint.
Tests the multipart/form-data endpoint that accepts video file and prompt.
"""

import asyncio
import aiohttp
import os
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

async def test_competition_endpoint():
    """Test the competition endpoint with multipart/form-data."""
    
    # Test video file path
    video_path = "../Cheetah cub learns how to hunt and kill a scrub hare.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Test video not found: {video_path}")
        return False
    
    # Test prompt
    prompt = "What action is happening in this clip?"
    
    # API endpoint
    url = "http://127.0.0.1:8000/infer"
    
    print(f"🧪 Testing competition endpoint...")
    print(f"📹 Video: {video_path}")
    print(f"❓ Prompt: {prompt}")
    print(f"🌐 Endpoint: {url}")
    print()
    
    try:
        async with aiohttp.ClientSession() as session:
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('video', 
                          open(video_path, 'rb'),
                          filename=os.path.basename(video_path),
                          content_type='video/mp4')
            data.add_field('prompt', prompt)
            
            # Make request
            async with session.post(url, data=data) as response:
                print(f"📊 Response Status: {response.status}")
                print(f"📋 Response Headers: {dict(response.headers)}")
                
                # Get response content
                content = await response.text()
                print(f"📝 Response Content:")
                print(f"{'='*50}")
                print(content)
                print(f"{'='*50}")
                
                # Check if response is successful
                if response.status == 200:
                    print("✅ Test PASSED - Endpoint working correctly!")
                    return True
                else:
                    print(f"❌ Test FAILED - Status code: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Test FAILED - Error: {e}")
        return False

async def test_health_endpoint():
    """Test the health endpoint."""
    
    url = "http://127.0.0.1:8000/health"
    
    print(f"🏥 Testing health endpoint: {url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                print(f"📊 Health Status: {response.status}")
                
                if response.status == 200:
                    health_data = await response.json()
                    print(f"📋 Health Data: {health_data}")
                    print("✅ Health endpoint working!")
                    return True
                else:
                    print(f"❌ Health endpoint failed: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Health test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 VuenCode Competition Endpoint Test")
    print("=" * 50)
    
    # Test health endpoint first
    health_ok = await test_health_endpoint()
    print()
    
    if not health_ok:
        print("❌ Health check failed. Make sure the server is running.")
        print("💡 Start the server with: python -m api.main")
        return
    
    # Test competition endpoint
    competition_ok = await test_competition_endpoint()
    
    print()
    print("=" * 50)
    if competition_ok:
        print("🎉 All tests PASSED! Competition endpoint is ready.")
        print("📋 Competition Format:")
        print("   - Endpoint: POST /infer")
        print("   - Content-Type: multipart/form-data")
        print("   - Fields: video (file), prompt (string)")
        print("   - Response: plain text")
    else:
        print("💥 Tests FAILED! Please check the implementation.")

if __name__ == "__main__":
    asyncio.run(main())
