"""
VuenCode FastAPI Application - Competition-Optimized Video Understanding System
Designed for sub-500ms latency with maximum reliability and performance.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

# Import our modules
from .schemas import (
    InferenceRequest, InferenceResponse, HealthResponse, ErrorResponse,
    QueryCategory, detect_query_category
)
from ..utils import (
    get_config, get_performance_tracker, get_fallback_handler,
    DeploymentMode, track_performance
)
from ..models.preprocessing import get_video_preprocessor
from ..models.gemini_processor import get_gemini_processor
from ..models.video_processor import get_enhanced_video_processor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for resource initialization and cleanup."""
    # Startup
    logger.info("=== VuenCode System Starting ===")
    
    # Initialize system components
    config = get_config()
    tracker = get_performance_tracker()
    fallback_handler = get_fallback_handler()
    
    # Initialize processors based on deployment mode
    video_processor = get_video_preprocessor()
    gemini_processor = get_gemini_processor()
    enhanced_video_processor = get_enhanced_video_processor()
    
    logger.info(f"System initialized in {config.deployment_mode} mode")
    logger.info(f"Target latency: {config.target_latency_ms}ms")
    logger.info(f"GPU acceleration: {config.use_gpu_acceleration}")
    
    # Phase 2 capabilities
    processing_summary = enhanced_video_processor.get_processing_summary()
    logger.info(f"Phase 2 capabilities: VST={processing_summary['capabilities']['vst_compression']}, "
               f"Multimodal={processing_summary['capabilities']['multimodal_fusion']}, "
               f"Audio={processing_summary['capabilities']['audio_processing']}")
    
    # Store in app state for endpoint access
    app.state.config = config
    app.state.tracker = tracker
    app.state.fallback_handler = fallback_handler
    app.state.video_processor = video_processor  # Keep for backward compatibility
    app.state.enhanced_video_processor = enhanced_video_processor  # Phase 2 processor
    app.state.gemini_processor = gemini_processor
    
    yield
    
    # Shutdown
    logger.info("=== VuenCode System Shutting Down ===")
    tracker.stop_monitoring()
    
    # Export final performance metrics
    try:
        tracker.export_metrics("performance_final.json")
        logger.info("Final performance metrics exported")
    except Exception as e:
        logger.error(f"Failed to export final metrics: {e}")


# Create FastAPI application with competition optimization
app = FastAPI(
    title="VuenCode - Video Understanding Competition System",
    description="High-performance video analysis with sub-500ms latency",
    version="1.0.0",
    docs_url="/docs" if get_config().deployment_mode == DeploymentMode.LOCAL else None,
    redoc_url="/redoc" if get_config().deployment_mode == DeploymentMode.LOCAL else None,
    lifespan=lifespan
)

# Add middleware for performance optimization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom middleware for request tracking
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Track request performance and add response headers."""
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    # Add request ID to state
    request.state.request_id = request_id
    request.state.start_time = start_time
    
    try:
        response = await call_next(request)
        
        # Add performance headers
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
        response.headers["X-Deployment-Mode"] = app.state.config.deployment_mode.value
        
        return response
        
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Competition-grade health check endpoint.
    Returns comprehensive system status for evaluation platform monitoring.
    """
    try:
        config = app.state.config
        tracker = app.state.tracker
        fallback_handler = app.state.fallback_handler
        
        # Get performance summary
        perf_summary = tracker.get_performance_summary()
        system_health = perf_summary.get("system_health", {})
        target_compliance = perf_summary.get("target_compliance", {})
        
        # Determine overall health status
        status = "healthy"
        if system_health.get("status") == "stressed":
            status = "degraded"
        elif not target_compliance.get("latency_compliance", True):
            status = "degraded"
        elif target_compliance.get("error_rate", 0) > 0.05:  # >5% error rate
            status = "unhealthy"
        
        # Get fallback system status
        fallback_status = fallback_handler.get_tier_health()
        
        # Get Phase 2 capabilities from enhanced video processor
        enhanced_processor = app.state.enhanced_video_processor
        processing_summary = enhanced_processor.get_processing_summary()
        phase2_capabilities = processing_summary['capabilities']
        
        return HealthResponse(
            status=status,
            deployment_mode=config.deployment_mode.value,
            performance_summary={
                "avg_latency_ms": perf_summary["competition_stats"]["average_latency_ms"],
                "p95_latency_ms": perf_summary["competition_stats"]["p95_latency_ms"],
                "p99_latency_ms": perf_summary["competition_stats"]["p99_latency_ms"],
                "throughput_rps": perf_summary["competition_stats"]["throughput_rps"],
                "success_rate": (perf_summary["competition_stats"]["successful_requests"] /
                               max(1, perf_summary["competition_stats"]["total_requests"])),
                "target_compliance": target_compliance.get("latency_compliance", True)
            },
            system_resources=system_health,
            fallback_status={
                "overall_status": fallback_status["overall"]["status"],
                "primary_available": fallback_status["primary"]["available"],
                "secondary_available": fallback_status["secondary"]["available"],
                "emergency_cache_size": fallback_status["overall"]["cache_size"],
                "emergency_mode": fallback_status["overall"]["emergency_mode"]
            },
            capabilities=phase2_capabilities  # Add Phase 2 capabilities
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            deployment_mode=app.state.config.deployment_mode.value,
            performance_summary={
                "error": "health_check_failed",
                "message": str(e)
            }
        )


@app.post("/infer", response_class=PlainTextResponse)
async def infer_video(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Competition-optimized video inference endpoint.
    Returns PLAIN TEXT only as required by evaluation platform.
    """
    # Generate request ID if not provided
    request_id = request.request_id or f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Get system components
        config = app.state.config
        tracker = app.state.tracker
        fallback_handler = app.state.fallback_handler
        video_processor = app.state.video_processor
        enhanced_video_processor = app.state.enhanced_video_processor
        gemini_processor = app.state.gemini_processor
        
        # Detect query category if not provided
        category = request.category or detect_query_category(request.query)
        
        # Track request performance
        async with tracker.track_request(request_id, category.value) as metrics:
            metrics.frame_count = request.max_frames or config.max_frames_per_video
            
            # Process with fallback handling using Phase 2 enhanced processor
            async def primary_processor(video_data, query):
                """Primary Phase 2 enhanced multimodal processing."""
                stage_start = time.time()
                
                # Use enhanced video processor with multimodal capabilities
                try:
                    result = await enhanced_video_processor.analyze_video_enhanced(
                        video_data=video_data,
                        query=query,
                        category=category,
                        max_frames=metrics.frame_count,
                        quality=request.quality or "balanced",
                        use_pro_model=False  # Start with Flash for speed
                    )
                    
                    # Track processing stages from enhanced result
                    if "performance_breakdown" in result:
                        breakdown = result["performance_breakdown"]
                        tracker.track_processing_stage(request_id, "video_processing",
                                                     breakdown.get("video_processing_ms", 0))
                        tracker.track_processing_stage(request_id, "audio_processing",
                                                     breakdown.get("audio_processing_ms", 0))
                        tracker.track_processing_stage(request_id, "vst_compression",
                                                     breakdown.get("vst_compression_ms", 0))
                        tracker.track_processing_stage(request_id, "multimodal_fusion",
                                                     breakdown.get("multimodal_fusion_ms", 0))
                        tracker.track_processing_stage(request_id, "gemini_inference",
                                                     breakdown.get("gemini_inference_ms", 0))
                    
                    metrics.model_used = result.get("model_used", "gemini-2.5-flash-enhanced")
                    
                    # Add Phase 2 metadata to metrics for monitoring
                    if "multimodal_analysis" in result:
                        multimodal = result["multimodal_analysis"]
                        metrics.multimodal_confidence = multimodal.get("fusion_confidence", 1.0)
                        metrics.audio_duration = multimodal.get("audio_duration", 0.0)
                        metrics.vst_compression_ratio = multimodal.get("vst_compression_ratio", 1.0)
                    
                    return result["content"]
                    
                except Exception as e:
                    logger.warning(f"Enhanced processing failed for {request_id}, falling back to basic: {e}")
                    # Fallback to basic processing within primary processor
                    frames = await video_processor.extract_frames(
                        video_data,
                        max_frames=metrics.frame_count,
                        quality=request.quality or "balanced"
                    )
                    
                    basic_result = await gemini_processor.analyze_video(
                        frames,
                        query,
                        category=category,
                        use_pro_model=False
                    )
                    
                    metrics.model_used = basic_result.get("model_used", "gemini-2.5-flash-basic")
                    return basic_result["content"]
            
            async def secondary_processor(video_data, query):
                """Secondary CPU fallback processing."""
                # Use CPU-only processing with reduced quality
                frames = await video_processor.extract_frames(
                    video_data,
                    max_frames=min(16, metrics.frame_count),  # Reduced frame count
                    quality="fast",
                    use_gpu=False
                )
                
                # Try with Flash model but with simpler prompt
                simplified_query = f"Briefly describe: {query}"
                result = await gemini_processor.analyze_video(
                    frames,
                    simplified_query,
                    category=category,
                    use_pro_model=False
                )
                
                metrics.model_used = result.get("model_used", "gemini-2.5-flash-cpu")
                return result["content"]
            
            # Prepare video data
            video_data = None
            video_hash = "unknown"
            
            if request.video_url:
                # Download video from URL
                stage_start = time.time()
                video_data = await video_processor.download_video(str(request.video_url))
                video_hash = video_processor.generate_video_hash(video_data)
                stage_time = (time.time() - stage_start) * 1000
                tracker.track_processing_stage(request_id, "video_download", stage_time)
                
            elif request.video_data:
                # Decode base64 video data
                import base64
                video_data = base64.b64decode(request.video_data)
                video_hash = video_processor.generate_video_hash(video_data)
            
            else:
                raise HTTPException(status_code=400, detail="No video source provided")
            
            # Process with intelligent fallback
            fallback_response = await fallback_handler.process_with_fallback(
                video_data=video_data,
                query=request.query,
                video_hash=video_hash,
                primary_processor=primary_processor,
                secondary_processor=secondary_processor,
                category=category.value
            )
            
            # Update metrics from fallback response
            metrics.model_used = fallback_response.metadata.get("model_used", "unknown")
            metrics.cache_hit = fallback_response.cached
            
            # Add background task to update cache if high-quality response
            if fallback_response.confidence >= 0.8 and not fallback_response.cached:
                background_tasks.add_task(
                    fallback_handler.cache.store,
                    video_hash, request.query, fallback_response
                )
            
            # Log successful processing
            logger.info(
                f"Request {request_id} processed successfully: "
                f"latency={fallback_response.latency_ms:.1f}ms, "
                f"tier={fallback_response.tier_used.value}, "
                f"category={category.value}"
            )
            
            # Return plain text content as required by competition
            return fallback_response.content
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id} failed with error: {e}")
        
        # Return error as plain text (competition requirement)
        error_message = (
            "I apologize, but I'm currently unable to process this video. "
            "Please try again in a moment. If the issue persists, "
            "please ensure your video is in a supported format (MP4, AVI, MOV)."
        )
        
        return error_message


@app.get("/metrics")
async def get_metrics():
    """
    Performance metrics endpoint for monitoring and debugging.
    Only available in local development mode.
    """
    config = app.state.config
    
    if config.deployment_mode == DeploymentMode.LOCAL:
        tracker = app.state.tracker
        return tracker.get_performance_summary()
    else:
        raise HTTPException(status_code=404, detail="Metrics endpoint not available in production")


@app.post("/admin/reset-metrics")
async def reset_metrics():
    """Reset performance metrics (development only)."""
    config = app.state.config
    
    if config.deployment_mode == DeploymentMode.LOCAL:
        tracker = app.state.tracker
        fallback_handler = app.state.fallback_handler
        
        tracker.reset_metrics()
        fallback_handler.reset_error_counts()
        
        return {"status": "metrics_reset", "timestamp": time.time()}
    else:
        raise HTTPException(status_code=404, detail="Admin endpoints not available in production")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions and return appropriate error responses."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    # For /infer endpoint, return plain text error (competition requirement)
    if request.url.path == "/infer":
        return PlainTextResponse(
            content=f"Error processing request: {exc.detail}",
            status_code=exc.status_code
        )
    
    # For other endpoints, return JSON error
    return ErrorResponse(
        error=f"http_{exc.status_code}",
        message=exc.detail,
        request_id=request_id
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Unhandled exception for request {request_id}: {exc}", exc_info=True)
    
    # For /infer endpoint, return plain text error (competition requirement)
    if request.url.path == "/infer":
        return PlainTextResponse(
            content="I apologize, but I'm experiencing technical difficulties. Please try again.",
            status_code=500
        )
    
    # For other endpoints, return JSON error
    return ErrorResponse(
        error="internal_server_error",
        message="An unexpected error occurred. Please try again.",
        request_id=request_id
    )


# Development server runner
def run_development_server():
    """Run development server with optimal settings."""
    config = get_config()
    
    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.deployment_mode == DeploymentMode.LOCAL,
        log_level="info",
        access_log=True,
        loop="asyncio"
    )


if __name__ == "__main__":
    run_development_server()