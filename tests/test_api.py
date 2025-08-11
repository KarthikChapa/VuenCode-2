"""
Comprehensive API tests for VuenCode competition system.
Tests both local development and production deployment scenarios.
"""

import pytest
import asyncio
import time
import json
import base64
from typing import Dict, Any
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image
import io

from fastapi.testclient import TestClient
import httpx

# Import our application
from ..api.main import app
from ..api.schemas import QueryCategory
from ..utils import get_config, get_performance_tracker, get_fallback_handler


class TestVuenCodeAPI:
    """Test suite for VuenCode API endpoints."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client for FastAPI application."""
        with TestClient(app) as client:
            yield client
    
    @pytest.fixture(scope="class") 
    def sample_video_data(self):
        """Create sample video data for testing."""
        # Create a simple test video (solid color frames)
        width, height, fps = 640, 480, 10
        duration = 3  # 3 seconds
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            # Generate frames with different colors
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR format
            frames_per_color = fps * duration // len(colors)
            
            for color_idx, color in enumerate(colors):
                for frame_idx in range(frames_per_color):
                    # Create solid color frame
                    frame = np.full((height, width, 3), color, dtype=np.uint8)
                    
                    # Add some text for variation
                    cv2.putText(frame, f"Frame {color_idx * frames_per_color + frame_idx}", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    out.write(frame)
            
            out.release()
            
            # Read video data
            with open(temp_path, 'rb') as f:
                video_data = f.read()
            
            return {
                "data": video_data,
                "base64": base64.b64encode(video_data).decode(),
                "path": temp_path
            }
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        assert "status" in data
        assert "deployment_mode" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Verify performance data structure
        if "performance_summary" in data:
            perf = data["performance_summary"]
            assert isinstance(perf, dict)
        
        print(f"Health check passed: {data['status']} in {data['deployment_mode']} mode")
    
    def test_infer_endpoint_with_base64(self, client, sample_video_data):
        """Test inference endpoint with base64 video data."""
        request_data = {
            "video_data": sample_video_data["base64"],
            "query": "What colors are shown in this video?",
            "category": "object_detection",
            "quality": "fast"
        }
        
        start_time = time.time()
        response = client.post("/infer", json=request_data)
        end_time = time.time()
        
        # Check response
        assert response.status_code == 200
        
        # Should return plain text (competition requirement)
        content = response.text
        assert isinstance(content, str)
        assert len(content) > 10  # Should be meaningful response
        
        # Check performance headers
        assert "X-Process-Time-MS" in response.headers
        process_time = float(response.headers["X-Process-Time-MS"])
        
        # Should be fast in local mode (stub processing)
        assert process_time < 1000, f"Processing took {process_time}ms, expected <1000ms"
        
        print(f"Inference completed in {process_time:.1f}ms")
        print(f"Response: {content[:100]}...")
    
    @pytest.mark.asyncio
    async def test_infer_endpoint_concurrent(self, sample_video_data):
        """Test concurrent requests to infer endpoint."""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Prepare multiple requests
            requests = []
            queries = [
                "Summarize this video",
                "What objects are visible?", 
                "Describe the main activities",
                "Count the number of different colors",
                "What happens in the video?"
            ]
            
            for i, query in enumerate(queries):
                request_data = {
                    "video_data": sample_video_data["base64"],
                    "query": query,
                    "category": "general_understanding",
                    "request_id": f"test_concurrent_{i}"
                }
                requests.append(
                    client.post("/infer", json=request_data)
                )
            
            # Execute requests concurrently
            start_time = time.time()
            responses = await asyncio.gather(*requests)
            end_time = time.time()
            
            # Verify all responses
            assert len(responses) == len(queries)
            
            for i, response in enumerate(responses):
                assert response.status_code == 200
                content = response.text
                assert len(content) > 10
                
                process_time = float(response.headers.get("X-Process-Time-MS", 0))
                assert process_time < 2000  # Allow more time for concurrent processing
                
                print(f"Request {i}: {process_time:.1f}ms - {content[:50]}...")
            
            total_time = (end_time - start_time) * 1000
            print(f"Concurrent processing: {len(queries)} requests in {total_time:.1f}ms")
    
    def test_infer_endpoint_validation(self, client):
        """Test input validation on infer endpoint."""
        # Test missing video source
        response = client.post("/infer", json={
            "query": "Test query"
        })
        assert response.status_code == 422  # Validation error
        
        # Test empty query
        response = client.post("/infer", json={
            "video_data": "fake_base64_data",
            "query": ""
        })
        assert response.status_code == 422
        
        # Test both video sources provided
        response = client.post("/infer", json={
            "video_url": "https://example.com/video.mp4",
            "video_data": "fake_base64_data", 
            "query": "Test query"
        })
        assert response.status_code == 422
        
        print("Input validation tests passed")
    
    def test_error_handling(self, client):
        """Test error handling for various failure scenarios."""
        # Test invalid base64 video data
        response = client.post("/infer", json={
            "video_data": "invalid_base64_data_that_will_fail",
            "query": "What is in this video?"
        })
        
        # Should return 500 but with plain text error message
        assert response.status_code == 500
        error_content = response.text
        assert isinstance(error_content, str)
        assert "technical difficulties" in error_content.lower() or "error" in error_content.lower()
        
        print(f"Error handling test passed: {error_content[:50]}...")
    
    @pytest.mark.parametrize("category,query", [
        ("video_summarization", "Summarize this video in detail"),
        ("action_recognition", "What actions are happening in this video?"),
        ("object_detection", "What objects can you see in this video?"),
        ("temporal_reasoning", "What happens first in this video sequence?"),
        ("scene_understanding", "Describe the setting and environment"),
    ])
    def test_category_specific_processing(self, client, sample_video_data, category, query):
        """Test category-specific processing optimization."""
        request_data = {
            "video_data": sample_video_data["base64"],
            "query": query,
            "category": category,
            "quality": "balanced"
        }
        
        response = client.post("/infer", json=request_data)
        
        assert response.status_code == 200
        content = response.text
        assert len(content) > 10
        
        # Check processing time is reasonable for category
        process_time = float(response.headers.get("X-Process-Time-MS", 0))
        
        # Different categories might have different performance expectations
        if category in ["object_detection", "video_summarization"]:
            assert process_time < 800, f"{category} took {process_time}ms (expected <800ms)"
        else:
            assert process_time < 1200, f"{category} took {process_time}ms (expected <1200ms)"
        
        print(f"{category}: {process_time:.1f}ms - {content[:50]}...")
    
    def test_metrics_endpoint_local_mode(self, client):
        """Test metrics endpoint (only available in local mode)."""
        config = get_config()
        
        if config.is_local_mode:
            response = client.get("/metrics")
            assert response.status_code == 200
            
            data = response.json()
            assert "competition_stats" in data
            assert isinstance(data["competition_stats"], dict)
            
            print("Metrics endpoint accessible in local mode")
        else:
            # In production mode, metrics should not be accessible
            response = client.get("/metrics")
            assert response.status_code == 404
            
            print("Metrics endpoint correctly blocked in production mode")
    
    def test_admin_reset_metrics(self, client):
        """Test admin endpoint for metrics reset."""
        config = get_config()
        
        if config.is_local_mode:
            response = client.post("/admin/reset-metrics")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "metrics_reset"
            assert "timestamp" in data
            
            print("Metrics reset successful")
        else:
            response = client.post("/admin/reset-metrics")
            assert response.status_code == 404
            
            print("Admin endpoint correctly blocked in production mode")


class TestPerformanceRequirements:
    """Test performance requirements for competition compliance."""
    
    @pytest.fixture
    def performance_tracker(self):
        """Get performance tracker for testing."""
        return get_performance_tracker()
    
    @pytest.fixture
    def fallback_handler(self):
        """Get fallback handler for testing.""" 
        return get_fallback_handler()
    
    def test_latency_targets(self, performance_tracker):
        """Test that system meets latency targets."""
        config = get_config()
        target_latency = config.target_latency_ms
        
        # Get current performance metrics
        summary = performance_tracker.get_performance_summary()
        
        if summary["competition_stats"]["total_requests"] > 0:
            p95_latency = summary["competition_stats"]["p95_latency_ms"]
            avg_latency = summary["competition_stats"]["average_latency_ms"]
            
            # Check latency compliance
            assert avg_latency < target_latency, f"Average latency {avg_latency:.1f}ms exceeds target {target_latency}ms"
            
            # P95 should be within reasonable bound of target
            assert p95_latency < target_latency * 1.5, f"P95 latency {p95_latency:.1f}ms too high"
            
            print(f"Latency compliance: avg={avg_latency:.1f}ms, p95={p95_latency:.1f}ms (target={target_latency}ms)")
        else:
            print("No performance data available yet")
    
    def test_fallback_system_health(self, fallback_handler):
        """Test fallback system health and availability."""
        health_status = fallback_handler.get_tier_health()
        
        # Check overall system health
        assert health_status["overall"]["status"] in ["healthy", "degraded"]
        
        # At least one tier should be available
        available_tiers = [
            tier for tier in ["primary", "secondary", "emergency"] 
            if health_status[tier]["available"]
        ]
        assert len(available_tiers) >= 1, "No fallback tiers available"
        
        print(f"Fallback system: {health_status['overall']['status']}")
        print(f"Available tiers: {available_tiers}")
    
    @pytest.mark.asyncio
    async def test_throughput_capacity(self, sample_video_data):
        """Test system throughput under load."""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            
            # Number of concurrent requests to test
            num_requests = 5
            
            # Create identical requests for consistent testing
            request_data = {
                "video_data": sample_video_data["base64"],
                "query": "Quick analysis of this video",
                "quality": "fast"
            }
            
            # Measure throughput
            start_time = time.time()
            
            requests = [
                client.post("/infer", json={**request_data, "request_id": f"throughput_test_{i}"})
                for i in range(num_requests)
            ]
            
            responses = await asyncio.gather(*requests)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify all responses successful
            successful_responses = [r for r in responses if r.status_code == 200]
            assert len(successful_responses) == num_requests
            
            # Calculate throughput
            throughput = num_requests / total_time
            
            # Should handle at least 2 RPS in local mode, more in production
            config = get_config()
            min_throughput = 1.5 if config.is_local_mode else 5.0
            
            assert throughput >= min_throughput, f"Throughput {throughput:.2f} RPS below minimum {min_throughput} RPS"
            
            print(f"Throughput test: {throughput:.2f} RPS ({num_requests} requests in {total_time:.2f}s)")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before running tests."""
    print("\n=== VuenCode Test Suite Starting ===")
    
    # Reset metrics for clean testing
    tracker = get_performance_tracker()
    fallback = get_fallback_handler()
    
    tracker.reset_metrics()
    fallback.reset_error_counts()
    
    yield
    
    print("\n=== VuenCode Test Suite Completed ===")
    
    # Print final test metrics
    summary = tracker.get_performance_summary()
    if summary["competition_stats"]["total_requests"] > 0:
        print(f"Test Summary:")
        print(f"  Total requests: {summary['competition_stats']['total_requests']}")
        print(f"  Average latency: {summary['competition_stats']['average_latency_ms']:.1f}ms")
        print(f"  Success rate: {summary['competition_stats']['successful_requests'] / summary['competition_stats']['total_requests'] * 100:.1f}%")


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=../api",
        "--cov=../models", 
        "--cov=../utils",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_html"
    ])