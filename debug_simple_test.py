#!/usr/bin/env python3
"""
Simple debug test for the inference endpoint
"""

import requests
import json
from pathlib import Path

def test_simple_inference():
    """Test with a simple JSON request (no file upload)"""
    print("🔍 Testing simple inference request...")
    
    # Test with just text and a simple query
    try:
        response = requests.post(
            "http://localhost:8000/infer",
            json={
                "query": "What is this?",
                "video_data": None,
                "video_url": None,
                "category": "general_understanding"
            },
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Simple inference test PASSED")
            return True
        else:
            print("❌ Simple inference test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Simple inference test ERROR: {e}")
        return False

def test_health_detailed():
    """Get detailed health information"""
    print("\n🔍 Getting detailed health info...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Status: {health_data.get('status')}")
            print(f"✅ Capabilities: {health_data.get('capabilities', {})}")
            print(f"✅ System Resources: {health_data.get('system_resources', {})}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    print("🚀 VuenCode Simple Debug Test")
    print("=" * 40)
    
    # Test health first
    health_ok = test_health_detailed()
    
    if health_ok:
        # Test simple inference
        inference_ok = test_simple_inference()
        
        if inference_ok:
            print("\n✅ BASIC FUNCTIONALITY WORKING")
        else:
            print("\n❌ INFERENCE ENDPOINT HAS ISSUES")
    else:
        print("\n❌ HEALTH ENDPOINT HAS ISSUES")

if __name__ == "__main__":
    main()