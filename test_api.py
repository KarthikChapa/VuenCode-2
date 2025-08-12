#!/usr/bin/env python3
"""
Simple test script to verify VuenCode API functionality
"""

import asyncio
import base64
import json
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.schemas import InferenceRequest, QueryCategory
from VuenCode.models.gemini_processor import get_gemini_processor
from VuenCode.models.video_processor import get_enhanced_video_processor
from VuenCode.utils import get_config

async def test_video_processing():
    """Test video processing functionality"""
    print("=== VuenCode API Test ===")
    
    # Initialize components
    config = get_config()
    print(f"Deployment mode: {config.deployment_mode}")
    print(f"Gemini API key: {'Set' if config.gemini_api_key else 'Not set'}")
    
    # Test video file path
    video_path = Path("../VID20220913223713.mp4")
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print(f"✅ Video file found: {video_path}")
    
    # Initialize processors
    try:
        gemini_processor = get_gemini_processor()
        video_processor = get_enhanced_video_processor()
        print("✅ Processors initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize processors: {e}")
        return False
    
    # Test video processing
    try:
        # Read video file
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        # Convert to base64
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        
        # Create test request
        request = InferenceRequest(
            video_data=video_base64,
            query="What is happening in this video?",
            category=QueryCategory.GENERAL_UNDERSTANDING
        )
        
        print("✅ Test request created successfully")
        print(f"Query: {request.query}")
        print(f"Category: {request.category}")
        print(f"Video size: {len(video_data)} bytes")
        
        # Test video processing
        print("\n🔄 Testing video processing...")
        result = await video_processor.analyze_video_enhanced(
            video_data, 
            request.query, 
            request.category
        )
        
        if result and 'content' in result:
            print("✅ Video processing successful!")
            print(f"Response: {result['content'][:200]}...")
            return True
        else:
            print("❌ Video processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_video_processing())
    if success:
        print("\n🎉 All tests passed! The API is ready for deployment.")
    else:
        print("\n❌ Tests failed. Please check the configuration and dependencies.")
        sys.exit(1)
