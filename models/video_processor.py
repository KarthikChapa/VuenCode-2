"""
Enhanced Video Processor for VuenCode Phase 2.
Integrates multimodal fusion, VST compression, audio processing, and TensorRT optimization
for competition-grade video understanding with sub-400ms latency.
"""

import asyncio
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import aiohttp
import base64

from .preprocessing import VideoFrame, get_video_preprocessor
from .gemini_processor import GeminiProcessor, get_gemini_processor, QueryCategory
from .audio_processor import AudioProcessor, get_audio_processor
from .multimodal_fusion import MultimodalFusionBus, get_fusion_bus
from .vst_processor import VSTProcessor, get_vst_processor, VSTCompressionLevel
from .pyramid_context import get_pyramid_context_processor

# Set up logger
logger = logging.getLogger(__name__)

# Try absolute imports first, then fall back to direct or relative imports
try:
    # Absolute imports for standalone deployment compatibility
    from VuenCode.utils.config import get_config
    from VuenCode.utils.metrics import track_performance
    logger.info("Using absolute imports from VuenCode package")
except ImportError:
    try:
        # Try direct imports (assuming the package is in PYTHONPATH)
        from utils.config import get_config
        from utils.metrics import track_performance
        logger.info("Using direct imports")
    except ImportError:
        try:
            # Fall back to relative imports for local development
            from ..utils.config import get_config
            from ..utils.metrics import track_performance
            logger.info("Using relative imports")
        except ImportError:
            logger.error("Could not import required modules. Using fallbacks.")
            # Define fallback functions if imports fail
            def get_config(key, default=None):
                return default
                
            def track_performance(func):
                return func


logger = logging.getLogger(__name__)


class EnhancedVideoProcessor:
    """
    Phase 2 Enhanced Video Processor with multimodal capabilities.
    Combines video preprocessing, audio processing, VST compression, and multimodal fusion.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize component processors
        self.video_preprocessor = get_video_preprocessor()
        self.audio_processor = get_audio_processor()
        self.fusion_bus = get_fusion_bus()
        self.vst_processor = get_vst_processor()
        self.gemini_processor = get_gemini_processor()
        
        # Initialize Pyramid Context processor (optional)
        self.pyramid_processor = None
        self.enable_pyramid_context = getattr(self.config, 'enable_pyramid_context', False)
        if self.enable_pyramid_context:
            self.pyramid_processor = get_pyramid_context_processor(config)
            self.logger.info("Pyramid Context processor enabled")
        
        # Performance configuration
        self.enable_vst_compression = getattr(self.config, 'enable_vst_compression', True)
        self.enable_multimodal_fusion = getattr(self.config, 'enable_multimodal_fusion', True)
        self.enable_audio_processing = getattr(self.config, 'enable_audio_processing', True)
        self.max_video_duration = 7200  # 120 minutes max
        
        self.logger.info(f"EnhancedVideoProcessor initialized - VST:{self.enable_vst_compression}, "
                        f"Multimodal:{self.enable_multimodal_fusion}, Audio:{self.enable_audio_processing}")
    
    @track_performance("video_analysis")
    async def analyze_video_enhanced(
        self,
        video_data: bytes,
        query: str,
        category: QueryCategory,
        max_frames: Optional[int] = None,
        quality: str = "balanced",
        use_pro_model: bool = False
    ) -> Dict[str, Any]:
        """
        Enhanced video analysis with Phase 2 multimodal capabilities.
        
        Args:
            video_data: Raw video bytes
            query: Natural language query
            category: Query category for optimization
            max_frames: Maximum frames to process
            quality: Processing quality level
            use_pro_model: Whether to use Gemini Pro model
            
        Returns:
            Enhanced analysis result with multimodal insights
        """
        start_time = time.time()
        
        try:
            # Step 1: Parallel video and audio processing
            video_task = asyncio.create_task(
                self._process_video_stream(video_data, max_frames, quality)
            )
            
            audio_task = asyncio.create_task(
                self._process_audio_stream(video_data) if self.enable_audio_processing else self._no_audio_fallback()
            )
            
            # Wait for both streams
            video_result, audio_result = await asyncio.gather(video_task, audio_task)
            frames = video_result["frames"]
            video_duration = video_result["duration"]
            audio_features = audio_result["features"]
            audio_segments = audio_result["segments"]
            
            self.logger.info(f"Parallel processing completed: {len(frames)} frames, "
                           f"{audio_features.duration:.1f}s audio in {time.time() - start_time:.3f}s")
            
            # Step 2: VST Compression for long videos
            if self.enable_vst_compression and video_duration > 300:  # >5 minutes
                vst_start = time.time()
                vst_compression = await self.vst_processor.compress_to_vst(
                    frames, category
                )
                
                # Use VST representative frames for processing
                frames = self.vst_processor.get_vst_frames_for_gemini(vst_compression)
                vst_time = time.time() - vst_start
                
                self.logger.info(f"VST compression: {vst_compression.original_frame_count} -> "
                                f"{len(frames)} frames ({vst_compression.compression_ratio:.1f}x) "
                                f"in {vst_time:.3f}s")
            else:
                vst_compression = None
            
            # Step 3: Multimodal Fusion
            multimodal_features = None
            if self.enable_multimodal_fusion:
                fusion_start = time.time()
                multimodal_features = await self.fusion_bus.fuse_modalities(
                    frames, audio_features, query, category
                )
                fusion_time = time.time() - fusion_start
                
                self.logger.info(f"Multimodal fusion completed in {fusion_time:.3f}s "
                                f"(confidence: {multimodal_features.fusion_confidence:.3f})")
            
            # Step 4: Enhanced Gemini Processing
            gemini_start = time.time()
            
            # Enhance prompt with multimodal context
            enhanced_prompt = query
            if multimodal_features:
                enhanced_prompt = self.fusion_bus.enhance_gemini_prompt(
                    query, multimodal_features, audio_segments
                )
            
            # Add VST context if available
            if vst_compression:
                enhanced_prompt = self.vst_processor.enhance_prompt_with_vst_context(
                    enhanced_prompt, vst_compression
                )
            
            # Process with Gemini
            gemini_result = await self.gemini_processor.analyze_video(
                frames,
                enhanced_prompt,
                category=category,
                use_pro_model=use_pro_model
            )
            gemini_time = time.time() - gemini_start
            
            # Step 5: Compile enhanced result
            total_time = time.time() - start_time
            
            enhanced_result = {
                "content": gemini_result["content"],
                "model_used": gemini_result["model_used"],
                "confidence": gemini_result.get("confidence", 0.85),
                "processing_time_ms": total_time * 1000,
                
                # Phase 2 enhancements
                "multimodal_analysis": {
                    "video_frames": len(frames),
                    "audio_duration": audio_features.duration,
                    "audio_transcription": [seg.text for seg in audio_segments if seg.text.strip()],
                    "fusion_confidence": multimodal_features.fusion_confidence if multimodal_features else None,
                    "vst_compression_ratio": vst_compression.compression_ratio if vst_compression else 1.0
                },
                
                # Performance metrics
                "performance_breakdown": {
                    "video_processing_ms": video_result["processing_time"] * 1000,
                    "audio_processing_ms": audio_result["processing_time"] * 1000,
                    "vst_compression_ms": vst_time * 1000 if vst_compression else 0,
                    "multimodal_fusion_ms": fusion_time * 1000 if multimodal_features else 0,
                    "gemini_inference_ms": gemini_time * 1000,
                    "total_latency_ms": total_time * 1000
                },
                
                # Quality indicators
                "quality_metrics": {
                    "frames_processed": len(frames),
                    "audio_quality": audio_features.confidence,
                    "multimodal_confidence": multimodal_features.fusion_confidence if multimodal_features else 1.0,
                    "compression_efficiency": vst_compression.compression_ratio if vst_compression else 1.0
                }
            }
            
            self.logger.info(f"Enhanced video analysis completed: {total_time * 1000:.1f}ms total, "
                           f"multimodal_confidence={enhanced_result['quality_metrics']['multimodal_confidence']:.3f}")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Enhanced video analysis failed: {e}")
            
            # Fallback to basic processing
            try:
                frames = await self.video_preprocessor.extract_frames(
                    video_data, max_frames or 20, quality
                )
                
                basic_result = await self.gemini_processor.analyze_video(
                    frames, query, category=category, use_pro_model=False
                )
                
                # Add fallback indicators
                basic_result.update({
                    "fallback_mode": True,
                    "fallback_reason": str(e),
                    "multimodal_analysis": {
                        "video_frames": len(frames),
                        "audio_duration": 0,
                        "audio_transcription": [],
                        "fusion_confidence": None,
                        "vst_compression_ratio": 1.0
                    }
                })
                
                return basic_result
                
            except Exception as fallback_error:
                self.logger.error(f"Fallback processing also failed: {fallback_error}")
                raise
    
    @track_performance("pyramid_video_analysis")
    async def analyze_video_pyramid(
        self,
        video_data: bytes,
        query: str,
        category: QueryCategory,
        use_pyramid: bool = True
    ) -> Dict[str, Any]:
        """
        Enhanced video analysis with optional Pyramid Context system.
        
        Args:
            video_data: Raw video bytes
            query: Natural language query
            category: Query category for optimization
            use_pyramid: Whether to use Pyramid Context system
            
        Returns:
            Dictionary with analysis results
        """
        
        if use_pyramid and self.pyramid_processor:
            self.logger.info("Using Pyramid Context system for video analysis")
            return await self.pyramid_processor.process_video_pyramid(video_data, query)
        else:
            self.logger.info("Using standard enhanced video analysis")
            return await self.analyze_video_enhanced(video_data, query, category)
    
    async def _process_video_stream(
        self, 
        video_data: bytes, 
        max_frames: Optional[int], 
        quality: str
    ) -> Dict[str, Any]:
        """Process video stream and extract frames."""
        start_time = time.time()
        
        try:
            # Extract frames using existing preprocessor
            frames = await self.video_preprocessor.extract_frames(
                video_data, 
                max_frames or self.config.max_frames_per_video,
                quality
            )
            
            # Estimate video duration from frame timestamps
            if frames:
                duration = frames[-1].timestamp - frames[0].timestamp
            else:
                duration = 0.0
            
            processing_time = time.time() - start_time
            
            return {
                "frames": frames,
                "duration": duration,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Video stream processing failed: {e}")
            return {
                "frames": [],
                "duration": 0.0,
                "processing_time": time.time() - start_time
            }
    
    async def _process_audio_stream(self, video_data: bytes) -> Dict[str, Any]:
        """Process audio stream and extract features."""
        start_time = time.time()
        
        try:
            # Extract audio features using audio processor
            audio_features = await self.audio_processor.extract_audio_from_video(video_data)
            
            # Get audio segments with transcription
            audio_segments = []
            if audio_features.duration > 0:
                audio_segments = await self.audio_processor.transcribe_speech(audio_features)
            
            processing_time = time.time() - start_time
            
            return {
                "features": audio_features,
                "segments": audio_segments,
                "processing_time": processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Audio stream processing failed: {e}")
            return await self._no_audio_fallback()
    
    async def _no_audio_fallback(self) -> Dict[str, Any]:
        """Fallback when audio processing is disabled or fails."""
        from .audio_processor import AudioFeatures
        import torch
        
        # Create empty audio features
        empty_features = AudioFeatures(
            waveform=torch.zeros(1000),
            sample_rate=16000,
            duration=0.0,
            embeddings=torch.zeros(768),
            confidence=0.0
        )
        
        return {
            "features": empty_features,
            "segments": [],
            "processing_time": 0.0
        }
    
    # Backward compatibility methods for existing API
    async def extract_frames(
        self, 
        video_data: bytes, 
        max_frames: int = 32, 
        quality: str = "balanced",
        use_gpu: bool = True
    ) -> List[VideoFrame]:
        """Extract frames using video preprocessor (backward compatibility)."""
        return await self.video_preprocessor.extract_frames(
            video_data, max_frames, quality, use_gpu
        )
    
    async def download_video(self, video_url: str) -> bytes:
        """Download video from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        raise Exception(f"Failed to download video: HTTP {response.status}")
        except Exception as e:
            self.logger.error(f"Video download failed: {e}")
            raise
    
    def generate_video_hash(self, video_data: bytes) -> str:
        """Generate hash for video data for caching."""
        return hashlib.sha256(video_data).hexdigest()[:16]
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing capabilities and configuration."""
        return {
            "phase": "2.0",
            "capabilities": {
                "vst_compression": self.enable_vst_compression,
                "multimodal_fusion": self.enable_multimodal_fusion,
                "audio_processing": self.enable_audio_processing,
                "max_video_duration": self.max_video_duration
            },
            "components": {
                "video_preprocessor": type(self.video_preprocessor).__name__,
                "audio_processor": type(self.audio_processor).__name__,
                "fusion_bus": type(self.fusion_bus).__name__,
                "vst_processor": type(self.vst_processor).__name__,
                "gemini_processor": type(self.gemini_processor).__name__
            },
            "performance_targets": {
                "target_latency_ms": getattr(self.config, 'target_latency_ms', 400),
                "max_frames": getattr(self.config, 'max_frames_per_video', 32)
            }
        }


# Global enhanced video processor instance
_enhanced_video_processor: Optional[EnhancedVideoProcessor] = None


def get_enhanced_video_processor() -> EnhancedVideoProcessor:
    """Get or create the global enhanced video processor instance."""
    global _enhanced_video_processor
    if _enhanced_video_processor is None:
        _enhanced_video_processor = EnhancedVideoProcessor()
    return _enhanced_video_processor


def initialize_enhanced_video_processor(config=None) -> EnhancedVideoProcessor:
    """Initialize the global enhanced video processor with custom config."""
    
    
# For compatibility with main_standalone.py
# Alias for backward compatibility with main_standalone.py
VideoProcessor = EnhancedVideoProcessor

def get_video_processor() -> EnhancedVideoProcessor:
    """Alias for get_enhanced_video_processor for backward compatibility."""
    return get_enhanced_video_processor()


# Backward compatibility: Export enhanced processor as default
get_video_processor = get_enhanced_video_processor
initialize_video_processor = initialize_enhanced_video_processor