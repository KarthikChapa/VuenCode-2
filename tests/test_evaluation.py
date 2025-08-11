"""
Competition evaluation test suite for VuenCode system.
Tests system performance against competition benchmarks and requirements.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any, Optional
import httpx
import json
import base64
from pathlib import Path

from ..api.schemas import QueryCategory
from ..utils import get_config


class TestCompetitionEvaluation:
    """
    Competition evaluation test suite.
    Tests system against realistic evaluation scenarios.
    """
    
    @pytest.fixture
    def config(self):
        """Get system configuration."""
        return get_config()
    
    @pytest.fixture
    def evaluation_queries(self):
        """
        Realistic evaluation queries covering all 16 competition categories.
        Based on typical competition evaluation patterns.
        """
        return [
            # Video Summarization
            {
                "category": QueryCategory.VIDEO_SUMMARIZATION,
                "query": "Provide a comprehensive summary of this video, highlighting the main events and key visual elements.",
                "expected_keywords": ["summary", "events", "main", "activities"]
            },
            
            # Temporal Reasoning
            {
                "category": QueryCategory.TEMPORAL_REASONING,
                "query": "What sequence of events occurs in this video? Describe the temporal order.",
                "expected_keywords": ["sequence", "order", "first", "then", "after"]
            },
            
            # Spatial Reasoning
            {
                "category": QueryCategory.SPATIAL_REASONING,
                "query": "Describe the spatial relationships between objects in different parts of the video.",
                "expected_keywords": ["spatial", "location", "position", "left", "right"]
            },
            
            # Object Detection
            {
                "category": QueryCategory.OBJECT_DETECTION,
                "query": "Identify and list all the distinct objects visible throughout this video.",
                "expected_keywords": ["objects", "visible", "identify", "list"]
            },
            
            # Action Recognition
            {
                "category": QueryCategory.ACTION_RECOGNITION,
                "query": "What specific actions and activities are being performed in this video?",
                "expected_keywords": ["actions", "activities", "performing", "movement"]
            },
            
            # Scene Understanding
            {
                "category": QueryCategory.SCENE_UNDERSTANDING,
                "query": "Describe the setting, environment, and overall context of this video scene.",
                "expected_keywords": ["setting", "environment", "scene", "context"]
            },
            
            # Counting/Quantification
            {
                "category": QueryCategory.COUNTING_QUANTIFICATION,
                "query": "How many distinct objects or people can you count in this video?",
                "expected_keywords": ["how many", "count", "number", "distinct"]
            },
            
            # Causal Reasoning
            {
                "category": QueryCategory.CAUSAL_REASONING,
                "query": "Explain the cause-and-effect relationships demonstrated in this video.",
                "expected_keywords": ["cause", "effect", "because", "leads to", "result"]
            },
            
            # Comparative Analysis
            {
                "category": QueryCategory.COMPARATIVE_ANALYSIS,
                "query": "Compare and contrast different elements or time periods within this video.",
                "expected_keywords": ["compare", "contrast", "different", "similar", "versus"]
            },
            
            # Visual Question Answering
            {
                "category": QueryCategory.VISUAL_QUESTION_ANSWERING,
                "query": "What is the primary focus of this video and how does it change over time?",
                "expected_keywords": ["what", "primary", "focus", "change", "time"]
            }
        ]
    
    @pytest.fixture
    def sample_video_base64(self):
        """Create sample video for evaluation testing."""
        # This would typically use a standardized test video
        # For now, create a simple colored video like in test_api.py
        import tempfile
        import cv2
        import numpy as np
        
        width, height, fps = 640, 480, 10
        duration = 5  # 5 seconds for more comprehensive testing
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            # Create more complex test video with different scenes
            scenes = [
                (255, 0, 0, "Red Scene - Objects Moving"),
                (0, 255, 0, "Green Scene - People Walking"), 
                (0, 0, 255, "Blue Scene - Items Interacting"),
                (255, 255, 0, "Yellow Scene - Activities"),
                (255, 0, 255, "Magenta Scene - Final")
            ]
            
            frames_per_scene = fps * duration // len(scenes)
            
            for scene_idx, (r, g, b, text) in enumerate(scenes):
                for frame_idx in range(frames_per_scene):
                    frame = np.full((height, width, 3), (b, g, r), dtype=np.uint8)  # BGR format
                    
                    # Add text overlay
                    cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Frame {scene_idx * frames_per_scene + frame_idx}", 
                              (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Add some geometric shapes for object detection
                    cv2.circle(frame, (200 + frame_idx * 2, 200), 30, (255, 255, 255), -1)
                    cv2.rectangle(frame, (300, 250), (400, 350), (0, 0, 0), 2)
                    
                    out.write(frame)
            
            out.release()
            
            # Read and encode video
            with open(temp_path, 'rb') as f:
                video_data = f.read()
            
            return base64.b64encode(video_data).decode()
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_category_comprehensive_evaluation(self, evaluation_queries, sample_video_base64):
        """
        Comprehensive evaluation across all competition categories.
        Tests both performance and quality metrics.
        """
        results = []
        
        async with httpx.AsyncClient(
            app=None,  # Will be set dynamically
            base_url="http://localhost:8000",
            timeout=30.0
        ) as client:
            
            # Import app dynamically to avoid circular imports
            from ..api.main import app
            client._app = app
            
            for test_case in evaluation_queries:
                print(f"\n--- Testing Category: {test_case['category'].value} ---")
                
                request_data = {
                    "video_data": sample_video_base64,
                    "query": test_case["query"],
                    "category": test_case["category"].value,
                    "quality": "balanced",
                    "request_id": f"eval_{test_case['category'].value}"
                }
                
                # Measure performance
                start_time = time.time()
                
                try:
                    response = await client.post("/infer", json=request_data)
                    end_time = time.time()
                    
                    # Collect metrics
                    result = {
                        "category": test_case["category"].value,
                        "query": test_case["query"],
                        "status_code": response.status_code,
                        "response_time_ms": (end_time - start_time) * 1000,
                        "server_process_time_ms": float(response.headers.get("X-Process-Time-MS", 0)),
                        "response_length": len(response.text) if response.status_code == 200 else 0,
                        "success": response.status_code == 200
                    }
                    
                    if response.status_code == 200:
                        content = response.text
                        result["content"] = content
                        
                        # Quality assessment
                        result["quality_score"] = self._assess_response_quality(
                            content, test_case["expected_keywords"]
                        )
                        
                        print(f"✓ Success: {result['response_time_ms']:.1f}ms")
                        print(f"  Quality: {result['quality_score']:.2f}")
                        print(f"  Response: {content[:100]}...")
                        
                    else:
                        print(f"✗ Failed: HTTP {response.status_code}")
                        result["error"] = response.text[:200]
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"✗ Exception: {e}")
                    results.append({
                        "category": test_case["category"].value,
                        "query": test_case["query"],
                        "status_code": 500,
                        "success": False,
                        "error": str(e),
                        "response_time_ms": (time.time() - start_time) * 1000
                    })
                
                # Small delay between requests
                await asyncio.sleep(0.1)
        
        # Analyze results
        await self._analyze_evaluation_results(results)
        
        return results
    
    def _assess_response_quality(self, content: str, expected_keywords: List[str]) -> float:
        """
        Assess response quality based on content analysis.
        Returns score between 0.0 and 1.0.
        """
        content_lower = content.lower()
        
        # Keyword matching score
        keyword_matches = sum(1 for keyword in expected_keywords if keyword in content_lower)
        keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0.5
        
        # Length assessment (reasonable response length)
        length_score = min(1.0, len(content) / 200.0)  # Normalize to ~200 chars
        if len(content) < 20:
            length_score = 0.1  # Too short
        elif len(content) > 1000:
            length_score = 0.8  # Very long but not necessarily bad
        
        # Coherence assessment (basic)
        coherence_score = 0.8  # Default - would need NLP for proper assessment
        
        # Combined quality score
        quality_score = (keyword_score * 0.4 + length_score * 0.3 + coherence_score * 0.3)
        
        return max(0.0, min(1.0, quality_score))
    
    async def _analyze_evaluation_results(self, results: List[Dict[str, Any]]):
        """Analyze and report evaluation results."""
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        print(f"\n=== EVALUATION RESULTS SUMMARY ===")
        print(f"Total tests: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"Success rate: {len(successful_results) / len(results) * 100:.1f}%")
        
        if successful_results:
            # Performance analysis
            response_times = [r["response_time_ms"] for r in successful_results]
            server_times = [r["server_process_time_ms"] for r in successful_results if r["server_process_time_ms"] > 0]
            
            print(f"\n--- Performance Metrics ---")
            print(f"Response time - Avg: {statistics.mean(response_times):.1f}ms")
            print(f"Response time - P95: {statistics.quantiles(response_times, n=20)[18]:.1f}ms")  # 95th percentile
            print(f"Response time - Max: {max(response_times):.1f}ms")
            
            if server_times:
                print(f"Server processing - Avg: {statistics.mean(server_times):.1f}ms")
                print(f"Server processing - P95: {statistics.quantiles(server_times, n=20)[18]:.1f}ms")
            
            # Quality analysis
            quality_scores = [r["quality_score"] for r in successful_results if "quality_score" in r]
            if quality_scores:
                print(f"\n--- Quality Metrics ---")
                print(f"Quality score - Avg: {statistics.mean(quality_scores):.3f}")
                print(f"Quality score - Min: {min(quality_scores):.3f}")
                print(f"Quality score - Max: {max(quality_scores):.3f}")
            
            # Category performance breakdown
            print(f"\n--- Category Performance ---")
            categories = {}
            for result in successful_results:
                cat = result["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result["response_time_ms"])
            
            for category, times in categories.items():
                avg_time = statistics.mean(times)
                print(f"{category}: {avg_time:.1f}ms avg")
        
        if failed_results:
            print(f"\n--- Failed Tests ---")
            for result in failed_results:
                print(f"{result['category']}: {result.get('error', 'Unknown error')}")
        
        # Competition compliance check
        config = get_config()
        target_latency = config.target_latency_ms
        
        if successful_results:
            compliant_results = [r for r in successful_results if r["response_time_ms"] <= target_latency]
            compliance_rate = len(compliant_results) / len(successful_results) * 100
            
            print(f"\n--- Competition Compliance ---")
            print(f"Target latency: {target_latency}ms")
            print(f"Compliance rate: {compliance_rate:.1f}%")
            
            if compliance_rate < 90:
                print(f"WARNING: Compliance rate below 90%")
            else:
                print(f"GOOD: High compliance rate")
    
    @pytest.mark.asyncio  
    async def test_stress_evaluation(self, sample_video_base64):
        """
        Stress test evaluation with multiple concurrent requests.
        Tests system stability under competition load conditions.
        """
        print(f"\n=== STRESS EVALUATION ===")
        
        # Configuration for stress test
        num_concurrent = 8
        num_rounds = 3
        
        async with httpx.AsyncClient(
            base_url="http://localhost:8000",
            timeout=60.0
        ) as client:
            
            from ..api.main import app
            client._app = app
            
            all_results = []
            
            for round_num in range(num_rounds):
                print(f"\nRound {round_num + 1}/{num_rounds} - {num_concurrent} concurrent requests")
                
                # Create batch of requests
                requests = []
                for i in range(num_concurrent):
                    request_data = {
                        "video_data": sample_video_base64,
                        "query": f"Analyze this video focusing on key elements (request {i+1})",
                        "category": "general_understanding",
                        "request_id": f"stress_{round_num}_{i}"
                    }
                    requests.append(client.post("/infer", json=request_data))
                
                # Execute batch
                start_time = time.time()
                responses = await asyncio.gather(*requests, return_exceptions=True)
                end_time = time.time()
                
                batch_time = (end_time - start_time) * 1000
                
                # Analyze batch results
                successful = 0
                failed = 0
                response_times = []
                
                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        failed += 1
                        print(f"  Request {i+1}: Exception - {response}")
                    else:
                        if response.status_code == 200:
                            successful += 1
                            process_time = float(response.headers.get("X-Process-Time-MS", 0))
                            response_times.append(process_time)
                        else:
                            failed += 1
                            print(f"  Request {i+1}: HTTP {response.status_code}")
                
                # Round summary
                print(f"  Batch time: {batch_time:.1f}ms")
                print(f"  Success: {successful}/{num_concurrent}")
                print(f"  Failed: {failed}/{num_concurrent}")
                
                if response_times:
                    print(f"  Avg response: {statistics.mean(response_times):.1f}ms")
                    print(f"  Max response: {max(response_times):.1f}ms")
                
                all_results.extend([
                    {
                        "round": round_num,
                        "successful": successful,
                        "failed": failed,
                        "batch_time_ms": batch_time,
                        "avg_response_time": statistics.mean(response_times) if response_times else 0
                    }
                ])
                
                # Brief pause between rounds
                await asyncio.sleep(1)
            
            # Overall stress test analysis
            total_requests = num_concurrent * num_rounds
            total_successful = sum(r["successful"] for r in all_results)
            overall_success_rate = total_successful / total_requests * 100
            
            print(f"\n--- Stress Test Summary ---")
            print(f"Total requests: {total_requests}")
            print(f"Overall success rate: {overall_success_rate:.1f}%")
            
            # Performance stability check
            avg_times = [r["avg_response_time"] for r in all_results if r["avg_response_time"] > 0]
            if len(avg_times) > 1:
                time_variance = statistics.variance(avg_times)
                print(f"Performance variance: {time_variance:.1f}ms²")
                
                if time_variance < 10000:  # 100ms standard deviation
                    print("STABLE: Low performance variance")
                else:
                    print("WARNING: High performance variance")
            
            # Assert minimum performance requirements
            assert overall_success_rate >= 85, f"Success rate {overall_success_rate:.1f}% below minimum 85%"
            
            if avg_times:
                max_avg_time = max(avg_times)
                config = get_config()
                assert max_avg_time <= config.target_latency_ms * 1.5, f"Max avg time {max_avg_time:.1f}ms too high"


@pytest.mark.asyncio
async def test_public_url_evaluation():
    """
    Test evaluation against public URL (for ngrok/production testing).
    This test is skipped unless EVALUATION_URL environment variable is set.
    """
    import os
    evaluation_url = os.getenv("EVALUATION_URL")
    
    if not evaluation_url:
        pytest.skip("EVALUATION_URL not set, skipping public URL evaluation")
    
    print(f"\n=== PUBLIC URL EVALUATION ===")
    print(f"Testing URL: {evaluation_url}")
    
    # Simple health check
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{evaluation_url}/health")
            assert response.status_code == 200
            
            health_data = response.json()
            print(f"Health check: {health_data['status']} in {health_data['deployment_mode']} mode")
            
            # Test inference endpoint
            test_video_b64 = "test_base64_data"  # Would use real test data
            
            request_data = {
                "video_data": test_video_b64,
                "query": "Test query for public evaluation",
                "category": "general_understanding"
            }
            
            start_time = time.time()
            response = await client.post(f"{evaluation_url}/infer", json=request_data)
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = (end_time - start_time) * 1000
                print(f"Public endpoint working: {response_time:.1f}ms")
            else:
                print(f"Public endpoint failed: HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}")
            
        except Exception as e:
            print(f"Public URL evaluation failed: {e}")
            raise


if __name__ == "__main__":
    # Run evaluation tests
    pytest.main([
        __file__,
        "-v",
        "-s",  # Don't capture output so we can see progress
        "--tb=short"
    ])