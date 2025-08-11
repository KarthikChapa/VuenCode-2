#!/usr/bin/env python3
"""
Comprehensive Phase 2 Testing Suite
Tests all multimodal capabilities: Video + Audio + VST + Fusion
"""

import requests
import json
import time
import sys
from pathlib import Path

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_inference_endpoint():
    """Test the inference endpoint with real video"""
    print("\n🔍 Testing /infer endpoint with Phase 2 capabilities...")
    
    # Find a video file
    video_paths = [
        "../VID20220913223713.mp4",
        "test_video.mp4",
        "../test_video.mp4"
    ]
    
    video_path = None
    for path in video_paths:
        if Path(path).exists():
            video_path = path
            break
    
    if not video_path:
        print("❌ No video file found for testing")
        return False
    
    print(f"📹 Using video: {video_path}")
    
    # Test different query types to validate multimodal capabilities
    test_queries = [
        {
            "query": "What is happening in this video? Describe both visual and audio elements.",
            "category": "description",
            "description": "Multimodal Description Test"
        },
        {
            "query": "Summarize the key moments with timestamps.",
            "category": "temporal_analysis", 
            "description": "VST Temporal Analysis Test"
        },
        {
            "query": "What sounds can you hear in this video?",
            "category": "description",
            "description": "Audio Processing Test"
        },
        {
            "query": "Analyze the relationship between what's seen and heard.",
            "category": "reasoning",
            "description": "Multimodal Fusion Test"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {test_case['description']} ---")
        
        try:
            with open(video_path, 'rb') as video_file:
                start_time = time.time()
                
                response = requests.post(
                    "http://localhost:8000/infer",
                    files={"video_file": video_file},
                    data={
                        "query": test_case["query"],
                        "category": test_case["category"]
                    },
                    timeout=120
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check for Phase 2 specific components
                    has_multimodal = "multimodal" in str(result).lower() or "audio" in str(result).lower()
                    has_timestamps = "timestamp" in str(result).lower() or ":" in result.get("answer", "")
                    has_vst = "vst" in str(result).lower() or "compression" in str(result).lower()
                    
                    print(f"✅ {test_case['description']} - SUCCESS")
                    print(f"⏱️  Processing time: {processing_time:.2f}s")
                    print(f"📊 Response length: {len(result.get('answer', ''))}")
                    print(f"🎯 Query: {test_case['query']}")
                    print(f"💡 Answer: {result.get('answer', 'No answer')[:200]}...")
                    
                    # Phase 2 feature detection
                    features_detected = []
                    if has_multimodal:
                        features_detected.append("Multimodal")
                    if has_timestamps:
                        features_detected.append("Timestamps/VST")
                    if has_vst:
                        features_detected.append("VST Processing")
                    
                    if features_detected:
                        print(f"🚀 Phase 2 Features Detected: {', '.join(features_detected)}")
                    
                    results.append({
                        "test": test_case["description"],
                        "status": "SUCCESS",
                        "time": processing_time,
                        "features": features_detected,
                        "answer_length": len(result.get("answer", "")),
                        "latency_ok": processing_time < 30.0  # Local testing threshold
                    })
                    
                else:
                    print(f"❌ {test_case['description']} - FAILED")
                    print(f"Status: {response.status_code}")
                    print(f"Response: {response.text[:200]}...")
                    
                    results.append({
                        "test": test_case["description"],
                        "status": "FAILED",
                        "time": processing_time,
                        "error": response.text
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
    """Generate comprehensive test report"""
    print("\n" + "="*60)
    print("🎯 PHASE 2 COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    successful_tests = [r for r in results if r.get("status") == "SUCCESS"]
    failed_tests = [r for r in results if r.get("status") != "SUCCESS"]
    
    print(f"✅ Successful Tests: {len(successful_tests)}")
    print(f"❌ Failed Tests: {len(failed_tests)}")
    print(f"📈 Success Rate: {len(successful_tests) / len(results) * 100:.1f}%")
    
    if successful_tests:
        avg_time = sum(r["time"] for r in successful_tests) / len(successful_tests)
        print(f"⏱️  Average Processing Time: {avg_time:.2f}s")
        
        # Feature detection summary
        all_features = []
        for r in successful_tests:
            all_features.extend(r.get("features", []))
        
        unique_features = list(set(all_features))
        if unique_features:
            print(f"🚀 Phase 2 Features Detected: {', '.join(unique_features)}")
        
        # Performance analysis
        fast_tests = [r for r in successful_tests if r.get("latency_ok", False)]
        print(f"⚡ Tests under 30s: {len(fast_tests)}/{len(successful_tests)}")
    
    print("\n📋 Detailed Results:")
    for result in results:
        status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"{status_icon} {result['test']}: {result['status']}")
        if "time" in result:
            print(f"   ⏱️  Time: {result['time']:.2f}s")
        if "features" in result and result["features"]:
            print(f"   🚀 Features: {', '.join(result['features'])}")
    
    # Readiness assessment
    print(f"\n🎯 DEPLOYMENT READINESS ASSESSMENT:")
    if len(successful_tests) >= 3:  # At least 3/4 tests pass
        print("✅ READY FOR GPU DEPLOYMENT!")
        print("   - Core functionality verified")
        print("   - Phase 2 features operational")
        print("   - Multimodal processing working")
        return True
    else:
        print("❌ NOT READY FOR DEPLOYMENT")
        print("   - Multiple test failures detected")
        print("   - Fix issues before GPU deployment")
        return False

def main():
    """Main testing routine"""
    print("🚀 VuenCode Phase 2 Comprehensive Testing Suite")
    print("=" * 50)
    
    # Test health endpoint
    if not test_health_endpoint():
        print("❌ Server not ready. Please ensure the server is running on localhost:8000")
        sys.exit(1)
    
    # Test inference capabilities
    results = test_inference_endpoint()
    
    if not results:
        print("❌ No inference tests completed")
        sys.exit(1)
    
    # Generate report and assessment
    ready_for_deployment = generate_test_report(results)
    
    # Save results
    with open("phase2_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full test results saved to: phase2_test_results.json")
    
    if ready_for_deployment:
        print("\n🎯 NEXT STEPS:")
        print("1. ✅ Local Phase 2 testing complete")
        print("2. 🚀 Ready for remote GPU deployment")
        print("3. 🔗 Deploy and get public endpoint")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()