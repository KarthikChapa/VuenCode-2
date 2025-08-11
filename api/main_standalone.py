"""
VuenCode Phase 2 - Main FastAPI Application (Standalone Version)
Competition-ready multimodal video understanding API with Phase 2 enhancements
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now import our modules
from utils.config import get_config
from utils.fallback import FallbackHandler
from utils.metrics import get_performance_tracker
from models.video_processor import VideoProcessor
from models.gemini_processor import GeminiProcessor
from models.multimodal_fusion import MultimodalFusion
from models.vst_processor import VSTProcessor
from models.audio_processor import AudioProcessor

# Configuration
config = get_config()
logger = logging.getLogger(__name__)

# Global components
video_processor = None
gemini_processor = None
multimodal_fusion = None
vst_processor = None
audio_processor = None
fallback_handler = None
performance_tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown handling"""
    global video_processor, gemini_processor, multimodal_fusion
    global vst_processor, audio_processor, fallback_handler, performance_tracker
    
    try:
        logger.info("Initializing VuenCode Phase 2 components...")
        
        # Initialize performance tracker
        performance_tracker = get_performance_tracker()
        
        # Initialize fallback handler
        fallback_handler = FallbackHandler(config)
        
        # Initialize processors
        video_processor = VideoProcessor(config)
        gemini_processor = GeminiProcessor(config)
        
        # Initialize Phase 2 components
        multimodal_fusion = MultimodalFusion(config)
        vst_processor = VSTProcessor(config)
        audio_processor = AudioProcessor(config)
        
        logger.info("VuenCode Phase 2 initialization complete")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize VuenCode: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Shutting down VuenCode Phase 2...")

# Create FastAPI app
app = FastAPI(
    title="VuenCode Phase 2",
    description="Competition-grade multimodal video understanding API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class VideoRequest(BaseModel):
    video_url: str = Field(..., description="URL of the video to analyze")
    prompt: Optional[str] = Field(None, description="Optional custom prompt")

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    phase2_enabled: bool
    components: Dict[str, bool]
    performance: Dict[str, Any]

class VideoResponse(BaseModel):
    video_url: str
    analysis: str
    confidence: float
    processing_time: float
    phase2_features: Dict[str, Any]
    metadata: Dict[str, Any]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with component status"""
    try:
        components_status = {
            "video_processor": video_processor is not None,
            "gemini_processor": gemini_processor is not None,
            "multimodal_fusion": multimodal_fusion is not None,
            "vst_processor": vst_processor is not None,
            "audio_processor": audio_processor is not None,
            "fallback_handler": fallback_handler is not None
        }
        
        performance_stats = {}
        if performance_tracker:
            performance_stats = performance_tracker.get_performance_summary()
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            environment=config.get("environment", "unknown"),
            phase2_enabled=True,
            components=components_status,
            performance=performance_stats
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer", response_model=VideoResponse)
async def analyze_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Enhanced video analysis with Phase 2 multimodal capabilities"""
    
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    logger.info(f"[{request_id}] Starting Phase 2 video analysis: {request.video_url}")
    
    try:
        # Start performance tracking (handled by context manager)
        
        # Phase 1: Video preprocessing and frame extraction
        logger.info(f"[{request_id}] Phase 1: Video preprocessing...")
        frames = await video_processor.extract_frames_async(request.video_url)
        
        if not frames:
            raise HTTPException(
                status_code=422,
                detail="No frames could be extracted from the video"
            )
        
        # Phase 2: Enhanced multimodal processing
        logger.info(f"[{request_id}] Phase 2: Multimodal processing...")
        
        # VST Processing for long videos
        if len(frames) > 10:  # Use VST for longer videos
            logger.info(f"[{request_id}] Applying VST compression...")
            vst_tokens = await vst_processor.compress_video_async(frames)
            processing_frames = vst_tokens
        else:
            processing_frames = frames
        
        # Audio processing
        logger.info(f"[{request_id}] Processing audio...")
        audio_features = await audio_processor.process_video_audio_async(request.video_url)
        
        # Multimodal fusion
        logger.info(f"[{request_id}] Performing multimodal fusion...")
        fused_features = await multimodal_fusion.fuse_modalities_async(
            video_features=processing_frames,
            audio_features=audio_features,
            text_context=request.prompt
        )
        
        # Generate analysis using Gemini
        logger.info(f"[{request_id}] Generating analysis with Gemini...")
        analysis_result = await gemini_processor.analyze_video_async(
            frames=processing_frames,
            prompt=request.prompt,
            multimodal_context=fused_features
        )
        
        processing_time = time.time() - start_time
        
        # Prepare Phase 2 features metadata
        phase2_features = {
            "vst_compression_applied": len(frames) > 10,
            "original_frames": len(frames),
            "processed_tokens": len(processing_frames) if isinstance(processing_frames, list) else 1,
            "audio_processing": audio_features is not None,
            "multimodal_fusion": fused_features is not None,
            "compression_ratio": len(frames) / len(processing_frames) if isinstance(processing_frames, list) and len(processing_frames) > 0 else 1.0
        }
        
        # Metadata
        metadata = {
            "request_id": request_id,
            "processing_time": processing_time,
            "model_version": "phase2-multimodal",
            "environment": config.get("environment", "production")
        }
        
        # Performance tracking handled by context manager
        
        logger.info(f"[{request_id}] Analysis completed in {processing_time:.2f}s")
        
        return VideoResponse(
            video_url=request.video_url,
            analysis=analysis_result["analysis"],
            confidence=analysis_result["confidence"],
            processing_time=processing_time,
            phase2_features=phase2_features,
            metadata=metadata
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"[{request_id}] Error during analysis: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Performance tracking handled by context manager
        
        # Try fallback if available
        if fallback_handler:
            logger.info(f"[{request_id}] Attempting fallback processing...")
            try:
                fallback_result = await fallback_handler.process_with_fallback(request.video_url, request.prompt)
                return VideoResponse(
                    video_url=request.video_url,
                    analysis=fallback_result["analysis"],
                    confidence=0.7,  # Lower confidence for fallback
                    processing_time=processing_time,
                    phase2_features={"fallback_used": True},
                    metadata={"request_id": request_id, "fallback": True}
                )
            except Exception as fallback_error:
                logger.error(f"[{request_id}] Fallback also failed: {fallback_error}")
        
        # Return error
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    # For standalone execution
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )