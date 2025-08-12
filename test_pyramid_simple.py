#!/usr/bin/env python3
"""
Simplified test script to test Pyramid Context system without VLM dependencies.
"""

import asyncio
import base64
import json
import sys
import os
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.schemas import InferenceRequest, QueryCategory
from VuenCode.models.video_processor import get_enhanced_video_processor
from VuenCode.utils import get_config

async def test_standard_approach(video_data: bytes, query: str):
    """Test standard enhanced video processing."""
    print("\n🔍 Testing Standard Enhanced Video Processing...")
    
    processor = get_enhanced_video_processor()
    start_time = time.time()
    
    result = await processor.analyze_video_enhanced(
        video_data, 
        query, 
        QueryCategory.GENERAL_UNDERSTANDING
    )
    
    processing_time = time.time() - start_time
    
    return {
        "approach": "Standard Enhanced",
        "processing_time": processing_time,
        "result": result,
        "success": "content" in result
    }

async def test_pyramid_context_approach(video_data: bytes, query: str):
    """Test Pyramid Context system."""
    print("\n🏗️ Testing Pyramid Context System...")
    
    # Enable Pyramid Context in config
    config = get_config()
    config.enable_pyramid_context = True
    
    processor = get_enhanced_video_processor(config)
    start_time = time.time()
    
    result = await processor.analyze_video_pyramid(
        video_data, 
        query, 
        QueryCategory.GENERAL_UNDERSTANDING,
        use_pyramid=True
    )
    
    processing_time = time.time() - start_time
    
    return {
        "approach": "Pyramid Context",
        "processing_time": processing_time,
        "result": result,
        "success": "content" in result
    }

async def test_temporal_queries(video_data: bytes):
    """Test temporal queries with Pyramid Context."""
    print("\n⏰ Testing Temporal Queries with Pyramid Context...")
    
    config = get_config()
    config.enable_pyramid_context = True
    
    processor = get_enhanced_video_processor(config)
    
    temporal_queries = [
        "What happened at 0:15?",
        "What did the person say at 0:30?",
        "What was happening around 0:45?",
        "Describe the scene at 1:00"
    ]
    
    results = []
    
    for query in temporal_queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        
        result = await processor.analyze_video_pyramid(
            video_data, 
            query, 
            QueryCategory.TEMPORAL_REASONING,
            use_pyramid=True
        )
        
        processing_time = time.time() - start_time
        
        results.append({
            "query": query,
            "processing_time": processing_time,
            "result": result,
            "success": "content" in result
        })
        
        if "content" in result:
            print(f"Response: {result['content'][:200]}...")
        else:
            print("Failed to generate response")
    
    return results

async def compare_approaches():
    """Compare standard vs Pyramid Context approaches."""
    print("=== VuenCode Pyramid Context Test ===")
    
    # Initialize components
    config = get_config()
    print(f"Deployment mode: {config.deployment_mode}")
    print(f"Gemini API key: {'Set' if config.gemini_api_key else 'Not set'}")
    
    # Test video file path
    video_path = Path("../Cheetah cub learns how to hunt and kill a scrub hare.mp4")
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    print(f"✅ Video file found: {video_path}")
    
    # Read video file
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    print(f"Video size: {len(video_data)} bytes")
    
    # Test queries
    test_queries = [
        "What is happening in this video?",
        "Describe the main actions and events",
        "What objects and people can you see?"
    ]
    
    all_results = []
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing Query: {query}")
        print(f"{'='*60}")
        
        query_results = []
        
        # Test Standard Approach
        try:
            result = await test_standard_approach(video_data, query)
            query_results.append(result)
        except Exception as e:
            print(f"❌ Standard approach failed: {e}")
            query_results.append({
                "approach": "Standard Enhanced",
                "processing_time": 0,
                "result": {"error": str(e)},
                "success": False
            })
        
        # Test Pyramid Context Approach
        try:
            result = await test_pyramid_context_approach(video_data, query)
            query_results.append(result)
        except Exception as e:
            print(f"❌ Pyramid Context approach failed: {e}")
            query_results.append({
                "approach": "Pyramid Context",
                "processing_time": 0,
                "result": {"error": str(e)},
                "success": False
            })
        
        all_results.append({
            "query": query,
            "results": query_results
        })
    
    # Test temporal queries
    print(f"\n{'='*60}")
    print("Testing Temporal Queries")
    print(f"{'='*60}")
    
    try:
        temporal_results = await test_temporal_queries(video_data)
        all_results.append({
            "query": "Temporal Queries",
            "results": temporal_results
        })
    except Exception as e:
        print(f"❌ Temporal queries failed: {e}")
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for query_result in all_results:
        print(f"\nQuery: {query_result['query']}")
        print("-" * 40)
        
        for result in query_result['results']:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['approach']}: {result['processing_time']:.2f}s")
            
            if result['success'] and 'content' in result['result']:
                print(f"   Response: {result['result']['content'][:100]}...")
    
    # Save detailed results
    with open("pyramid_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to pyramid_test_results.json")
    return True

if __name__ == "__main__":
    success = asyncio.run(compare_approaches())
    if success:
        print("\n🎉 All tests completed! Check the results above.")
    else:
        print("\n❌ Tests failed. Please check the configuration and dependencies.")
        sys.exit(1)
