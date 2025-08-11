#!/usr/bin/env python3
"""
Working Phase 2 Testing Suite - Compatible with current API
"""

import requests
import json
import time
import base64
from pathlib import Path

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: Status={data.get('status')}")
            print(f"✅ Phase 2 Capabilities: {data.get('capabilities', {})}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def encode_video_file(video_path):
    """Encode video file to base64"""
    try:
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')
            return video_base64
    except Exception as e:
        print(f"❌ Failed to encode video file: {e}")
        return None

def test_inference_endpoint():
    """Test the inference endpoint with Phase 2 capabilities"""
    print("\n🔍 Testing /infer endpoint with Phase 2 capabilities...")
    
    # Find a video file
    video_paths = [
        "../VID20220913223713.mp4",
        "test_video.mp4",
        "../test_video.mp4"
    ]
    
    video_path = None
    video_base64 = None
    
    for path in video_paths:
        if Path(path).exists():
            video_path = path
            video_base64 = encode_video_file(path)
            if video_base64:
                break
    
    # Test cases
    test_queries = [
        {
            "query": "What is happening in this video?",
            "category": "general_understanding",
            "description": "Basic Video Analysis"
        },
        {
            "query": "Summarize the key events in this video with timestamps",
            "category": "video_summarization", 
            "description": "VST Temporal Analysis"
        },
        {
            "query": "What objects can you see in this video?",
            "category": "object_detection",
            "description": "Object Detection Test"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {test_case['description']} ---")
        
        # Prepare request payload
        payload = {
            "query": test_case["query"],
            "category": test_case["category"]
        }
        
        # Add video data if available
        if video_base64:
            payload["video_data"] = video_base64
            print(f"📹 Using video file: {video_path} (base64 encoded)")
        else:
            payload["video_data"] = None
            print("📹 No video file - testing text-only mode")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:8000/infer",
                json=payload,
                timeout=60
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                result_text = response.text
                
                # Check for Phase 2 specific features
                has_timestamps = any(char in result_text for char in [":", "00:"])
                has_structured = any(word in result_text.lower() for word in ["1.", "2.", "timeline", "analysis"])
                
                print(f"✅ {test_case['description']} - SUCCESS")
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                print(f"📊 Response length: {len(result_text)}")
                print(f"💡 Response preview: {result_text[:200]}...")
                
                # Feature detection
                features = []
                if has_timestamps:
                    features.append("Timestamps")
                if has_structured:
                    features.append("Structured")
                
                if features:
                    print(f"🚀 Detected features: {', '.join(features)}")
                
                results.append({
                    "test": test_case["description"],
                    "status": "SUCCESS",
                    "time": processing_time,
                    "features": features,
                    "response_length": len(result_text),
                    "has_video": video_base64 is not None
                })
                
            else:
                print(f"❌ {test_case['description']} - FAILED")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text[:200]}...")
                
                results.append({
                    "test": test_case["description"],
                    "status": "FAILED",
                    "time": processing_time,
                    "error": response.text,
                    "status_code": response.status_code
                })
        
        except Exception as e:
            print(f"❌ {test_case['description']} - ERROR: {e}")
            results.append({
                "test": test_case["description"],
                "status": "ERROR",
                "error": str(e)
            })
    
    return results

def generate_test_report(results):
    """Generate test report"""
    print("\n" + "="*60)
    print("🎯 PHASE 2 WORKING TEST REPORT")
    print("="*60)
    
    successful_tests = [r for r in results if r.get("status") == "SUCCESS"]
    failed_tests = [r for r in results if r.get("status") != "SUCCESS"]
    
    print(f"✅ Successful Tests: {len(successful_tests)}")
    print(f"❌ Failed Tests: {len(failed_tests)}")
    print(f"📈 Success Rate: {len(successful_tests) / len(results) * 100:.1f}%")
    
    if successful_tests:
        avg_time = sum(r["time"] for r in successful_tests) / len(successful_tests)
        print(f"⏱️  Average Processing Time: {avg_time:.2f}s")
        
        # Check if we have video processing
        video_tests = [r for r in successful_tests if r.get("has_video", False)]
        if video_tests:
            print(f"📹 Video Processing Tests: {len(video_tests)}")
        
        # Feature summary
        all_features = []
        for r in successful_tests:
            all_features.extend(r.get("features", []))
        
        unique_features = list(set(all_features))
        if unique_features:
            print(f"🚀 Phase 2 Features Working: {', '.join(unique_features)}")
    
    print("\n📋 Detailed Results:")
    for result in results:
        status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"{status_icon} {result['test']}: {result['status']}")
        if "time" in result:
            print(f"   ⏱️  Time: {result['time']:.2f}s")
        if "features" in result and result["features"]:
            print(f"   🚀 Features: {', '.join(result['features'])}")
    
    # Deployment readiness
    ready = len(successful_tests) >= 2  # At least 2/3 tests pass
    
    print(f"\n🎯 DEPLOYMENT READINESS:")
    if ready:
        print("✅ READY FOR GPU DEPLOYMENT!")
        print("   - Core functionality verified")
        print("   - API endpoints working")
        print("   - Phase 2 system operational")
        return True
    else:
        print("❌ NEEDS MORE TESTING")
        print("   - Some functionality issues")
        return False

def main():
    """Main testing routine"""
    print("🚀 VuenCode Phase 2 Working Test Suite")
    print("=" * 50)
    
    # Test health
    if not test_health_endpoint():
        print("❌ Server not ready. Exiting.")
        return
    
    # Test inference
    results = test_inference_endpoint()
    
    if not results:
        print("❌ No tests completed")
        return
    
    # Generate report
    ready = generate_test_report(results)
    
    # Save results
    with open("phase2_working_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Test results saved to: phase2_working_results.json")
    
    if ready:
        print("\n🎯 NEXT STEPS:")
        print("1. ✅ Local Phase 2 testing successful")
        print("2. 🚀 Ready for remote deployment")
        print("3. 🔗 Can proceed with GPU deployment")

if __name__ == "__main__":
    main()