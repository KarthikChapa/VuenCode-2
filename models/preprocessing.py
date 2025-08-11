"""
Advanced video preprocessing module for VuenCode competition system.
Supports both CPU (Phase 1) and GPU-accelerated (Phase 2) processing.
Implements hybrid frame extraction strategies for maximum performance.
"""

import asyncio
import cv2
import numpy as np
import hashlib
import logging
import time
from typing import List, Optional, Union, Tuple, Any, Dict
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import aiofiles
from PIL import Image
import io
import base64

from ..utils import get_config, track_performance


logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    """Represents a single video frame with metadata."""
    image: Image.Image
    timestamp: float  # Frame timestamp in seconds
    frame_index: int  # Original frame number
    confidence: float = 1.0  # Frame quality/relevance score
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_base64(self) -> str:
        """Convert frame to base64 string for API transmission."""
        buffer = io.BytesIO()
        self.image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def resize(self, max_size: Tuple[int, int] = (1024, 1024)) -> 'VideoFrame':
        """Resize frame while maintaining aspect ratio."""
        image = self.image.copy()
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        return VideoFrame(
            image=image,
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            confidence=self.confidence,
            metadata=self.metadata.copy()
        )


class SceneDetector:
    """
    Scene change detection for intelligent frame selection.
    Removes redundant frames to optimize processing.
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
    
    def detect_scene_changes(self, frames: List[np.ndarray]) -> List[int]:
        """
        Detect scene changes in video frames using histogram comparison.
        
        Args:
            frames: List of frame arrays (BGR format)
            
        Returns:
            List of frame indices where scene changes occur
        """
        if len(frames) <= 1:
            return list(range(len(frames)))
        
        scene_frames = [0]  # Always include first frame
        prev_hist = self._compute_histogram(frames[0])
        
        for i, frame in enumerate(frames[1:], 1):
            current_hist = self._compute_histogram(frame)
            
            # Calculate histogram correlation
            correlation = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
            
            # If correlation is below threshold, we have a scene change
            if correlation < (1.0 - self.threshold):
                scene_frames.append(i)
                prev_hist = current_hist
        
        # Always include last frame
        if scene_frames[-1] != len(frames) - 1:
            scene_frames.append(len(frames) - 1)
        
        self.logger.debug(f"Scene detection: {len(frames)} -> {len(scene_frames)} frames")
        return scene_frames
    
    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for frame comparison."""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for each channel
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return hist


class VideoPreprocessor:
    """
    Competition-optimized video preprocessing system.
    Supports multiple extraction strategies and GPU acceleration.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components based on configuration
        self.scene_detector = SceneDetector(threshold=self.config.scene_detection_threshold)
        
        # GPU acceleration setup
        self.use_gpu = self.config.use_gpu_acceleration
        if self.use_gpu:
            try:
                import torch
                import torchvision.transforms as transforms
                
                self.device = torch.device(f'cuda:{self.config.gpu_device_id}' if torch.cuda.is_available() else 'cpu')
                self.gpu_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                self.logger.info(f"GPU acceleration enabled on device: {self.device}")
                
            except ImportError:
                self.logger.warning("GPU acceleration requested but PyTorch not available, falling back to CPU")
                self.use_gpu = False
        
        # Frame extraction strategy
        self.extraction_strategy = self._select_extraction_strategy()
        
        self.logger.info(f"VideoPreprocessor initialized: strategy={self.extraction_strategy}, gpu={self.use_gpu}")
    
    def _select_extraction_strategy(self) -> str:
        """Select optimal frame extraction strategy based on deployment mode."""
        if self.config.is_local_mode:
            return "uniform_sampling"  # Simple and fast for development
        elif self.config.is_competition_mode:
            return "hybrid_intelligent"  # Maximum quality for competition
        else:
            return "scene_aware"  # Balanced approach for production
    
    async def download_video(self, url: str, timeout: int = 30) -> bytes:
        """
        Download video from URL with optimized settings.
        
        Args:
            url: Video URL
            timeout: Download timeout in seconds
            
        Returns:
            Video data as bytes
        """
        try:
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download video: HTTP {response.status}")
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '')
                    if not any(video_type in content_type.lower() for video_type in ['video/', 'application/octet-stream']):
                        self.logger.warning(f"Unexpected content type: {content_type}")
                    
                    # Download video data
                    video_data = await response.read()
                    
                    self.logger.info(f"Downloaded video: {len(video_data)} bytes from {url}")
                    return video_data
                    
        except Exception as e:
            self.logger.error(f"Failed to download video from {url}: {e}")
            raise
    
    def generate_video_hash(self, video_data: bytes) -> str:
        """Generate unique hash for video content."""
        return hashlib.md5(video_data).hexdigest()
    
    @track_performance("video_preprocessing")
    async def extract_frames(
        self,
        video_data: bytes,
        max_frames: Optional[int] = None,
        quality: str = "auto",
        use_gpu: Optional[bool] = None
    ) -> List[VideoFrame]:
        """
        Extract frames from video using optimal strategy.
        
        Args:
            video_data: Video data as bytes
            max_frames: Maximum number of frames to extract
            quality: Quality mode ('fast', 'balanced', 'high', 'auto')
            use_gpu: Override GPU usage setting
            
        Returns:
            List of extracted VideoFrame objects
        """
        if use_gpu is None:
            use_gpu = self.use_gpu
        
        max_frames = max_frames or self.config.max_frames_per_video
        
        try:
            # Save video data to temporary file for OpenCV
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(video_data)
                temp_path = temp_file.name
            
            try:
                # Extract frames using selected strategy
                if self.extraction_strategy == "uniform_sampling":
                    frames = await self._extract_uniform_sampling(temp_path, max_frames, quality)
                elif self.extraction_strategy == "scene_aware":
                    frames = await self._extract_scene_aware(temp_path, max_frames, quality)
                elif self.extraction_strategy == "hybrid_intelligent":
                    frames = await self._extract_hybrid_intelligent(temp_path, max_frames, quality)
                else:
                    # Fallback to uniform sampling
                    frames = await self._extract_uniform_sampling(temp_path, max_frames, quality)
                
                self.logger.info(f"Extracted {len(frames)} frames using {self.extraction_strategy} strategy")
                return frames
                
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            raise
    
    async def _extract_uniform_sampling(self, video_path: str, max_frames: int, quality: str) -> List[VideoFrame]:
        """Extract frames using uniform temporal sampling."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25  # Default FPS if not available
            duration = total_frames / fps
            
            # Calculate sampling interval
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames / max_frames
                frame_indices = [int(i * step) for i in range(max_frames)]
            
            frames = []
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Resize based on quality setting
                    pil_image = self._resize_for_quality(pil_image, quality)
                    
                    timestamp = frame_idx / fps
                    video_frame = VideoFrame(
                        image=pil_image,
                        timestamp=timestamp,
                        frame_index=frame_idx,
                        confidence=1.0,
                        metadata={"extraction_method": "uniform_sampling"}
                    )
                    
                    frames.append(video_frame)
            
            return frames
            
        finally:
            cap.release()
    
    async def _extract_scene_aware(self, video_path: str, max_frames: int, quality: str) -> List[VideoFrame]:
        """Extract frames using scene change detection."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        
        try:
            # Read all frames for scene detection (or sample if too many)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # For very long videos, sample frames for scene detection
            detection_frames = min(total_frames, 1000)  # Max 1000 frames for detection
            detection_step = max(1, total_frames // detection_frames)
            
            raw_frames = []
            frame_indices = []
            
            for i in range(0, total_frames, detection_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    raw_frames.append(frame)
                    frame_indices.append(i)
            
            # Detect scene changes
            scene_indices = self.scene_detector.detect_scene_changes(raw_frames)
            
            # Map back to original frame indices
            selected_frame_indices = [frame_indices[i] for i in scene_indices]
            
            # If we have too many scene frames, sample uniformly
            if len(selected_frame_indices) > max_frames:
                step = len(selected_frame_indices) / max_frames
                selected_frame_indices = [selected_frame_indices[int(i * step)] for i in range(max_frames)]
            
            # Extract the selected frames
            frames = []
            for frame_idx in selected_frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    pil_image = self._resize_for_quality(pil_image, quality)
                    
                    timestamp = frame_idx / fps
                    video_frame = VideoFrame(
                        image=pil_image,
                        timestamp=timestamp,
                        frame_index=frame_idx,
                        confidence=1.0,
                        metadata={"extraction_method": "scene_aware"}
                    )
                    
                    frames.append(video_frame)
            
            return frames
            
        finally:
            cap.release()
    
    async def _extract_hybrid_intelligent(self, video_path: str, max_frames: int, quality: str) -> List[VideoFrame]:
        """
        Competition-grade hybrid extraction combining multiple strategies.
        Uses scene detection + uniform sampling + quality scoring.
        """
        # For now, use scene-aware extraction as the intelligent method
        # In Phase 2, this will be enhanced with dynamic selection and VST compression
        frames = await self._extract_scene_aware(video_path, max_frames, quality)
        
        # Add quality scoring (placeholder for advanced algorithms)
        for frame in frames:
            frame.metadata["extraction_method"] = "hybrid_intelligent"
            frame.confidence = self._calculate_frame_quality(frame.image)
        
        # Sort by quality and keep the best frames
        frames.sort(key=lambda f: f.confidence, reverse=True)
        
        return frames[:max_frames]
    
    def _resize_for_quality(self, image: Image.Image, quality: str) -> Image.Image:
        """Resize image based on quality setting."""
        size_configs = {
            "fast": (512, 512),
            "balanced": (768, 768),
            "high": (1024, 1024),
            "auto": (768, 768) if self.config.is_local_mode else (1024, 1024)
        }
        
        target_size = size_configs.get(quality, (768, 768))
        
        # Resize while maintaining aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        return image
    
    def _calculate_frame_quality(self, image: Image.Image) -> float:
        """
        Calculate frame quality score (0.0 - 1.0).
        Placeholder for advanced quality metrics.
        """
        # Simple quality metrics for now
        # In Phase 2, this will use advanced computer vision techniques
        
        # Convert to numpy for analysis
        np_image = np.array(image)
        
        # Calculate variance (higher variance = more detail = higher quality)
        variance = np.var(np_image)
        normalized_variance = min(1.0, variance / 10000.0)
        
        # Calculate brightness (avoid over/under-exposed frames)
        brightness = np.mean(np_image) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Prefer medium brightness
        
        # Combine metrics
        quality_score = (normalized_variance * 0.7 + brightness_score * 0.3)
        
        return max(0.1, min(1.0, quality_score))  # Clamp between 0.1 and 1.0


# Global preprocessor instance
_video_preprocessor: Optional[VideoPreprocessor] = None


def get_video_preprocessor() -> VideoPreprocessor:
    """Get or create the global video preprocessor instance."""
    global _video_preprocessor
    if _video_preprocessor is None:
        _video_preprocessor = VideoPreprocessor()
    return _video_preprocessor


def initialize_video_preprocessor(config=None) -> VideoPreprocessor:
    """Initialize the global video preprocessor with custom config."""
    global _video_preprocessor
    _video_preprocessor = VideoPreprocessor(config)
    return _video_preprocessor