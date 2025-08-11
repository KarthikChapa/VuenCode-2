"""
Visual Summarization Tokens (VST) Processor for VuenCode Phase 2.
Implements intelligent video compression for handling long-form content (up to 120 minutes)
while maintaining semantic richness for accurate video understanding.
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn.functional as F
from PIL import Image

from .preprocessing import VideoFrame
from .gemini_processor import QueryCategory

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


class VSTCompressionLevel(Enum):
    """VST compression levels for different video lengths."""
    MINIMAL = "minimal"      # <5 minutes: Light compression
    MODERATE = "moderate"    # 5-30 minutes: Moderate compression
    AGGRESSIVE = "aggressive"  # 30-60 minutes: Heavy compression
    EXTREME = "extreme"      # 60-120 minutes: Maximum compression


@dataclass
class VSTToken:
    """Visual Summarization Token representing a compressed video segment."""
    token_id: str
    start_time: float
    end_time: float
    frame_count: int
    representative_frame: VideoFrame
    semantic_features: torch.Tensor
    importance_score: float
    compression_ratio: float
    
    # Content analysis
    scene_type: str = "unknown"
    motion_level: str = "low"
    visual_complexity: float = 0.0
    
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        return f"VST[{self.token_id}]: {self.start_time:.1f}s-{self.end_time:.1f}s ({self.frame_count}frames, score={self.importance_score:.3f})"


@dataclass
class VSTCompression:
    """Result of VST compression process."""
    vst_tokens: List[VSTToken]
    original_frame_count: int
    compressed_frame_count: int
    compression_ratio: float
    processing_time: float
    compression_level: VSTCompressionLevel
    
    # Quality metrics
    semantic_preservation: float = 1.0
    temporal_coverage: float = 1.0
    information_density: float = 1.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get compression summary for monitoring."""
        return {
            "vst_tokens": len(self.vst_tokens),
            "original_frames": self.original_frame_count,
            "compressed_frames": self.compressed_frame_count,
            "compression_ratio": f"{self.compression_ratio:.2f}x",
            "processing_time": f"{self.processing_time:.3f}s",
            "compression_level": self.compression_level.value,
            "quality_metrics": {
                "semantic_preservation": self.semantic_preservation,
                "temporal_coverage": self.temporal_coverage,
                "information_density": self.information_density
            },
            "efficiency": f"{self.compressed_frame_count / max(1, self.processing_time):.1f} tokens/sec"
        }


class VSTProcessor:
    """
    Advanced Visual Summarization Token processor for intelligent video compression.
    Implements scene-aware compression with semantic preservation.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # VST parameters
        self.max_tokens_per_minute = {
            VSTCompressionLevel.MINIMAL: 20,      # 20 tokens/minute
            VSTCompressionLevel.MODERATE: 15,     # 15 tokens/minute  
            VSTCompressionLevel.AGGRESSIVE: 10,   # 10 tokens/minute
            VSTCompressionLevel.EXTREME: 6       # 6 tokens/minute
        }
        
        # Scene detection parameters
        self.scene_change_threshold = 0.3
        self.motion_threshold = 0.2
        self.visual_complexity_threshold = 0.4
        
        # Device for tensor operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"VSTProcessor initialized on device: {self.device}")
    
    def determine_compression_level(self, video_duration: float) -> VSTCompressionLevel:
        """
        Determine optimal compression level based on video duration.
        
        Args:
            video_duration: Video duration in seconds
            
        Returns:
            Appropriate compression level
        """
        if video_duration < 300:  # < 5 minutes
            return VSTCompressionLevel.MINIMAL
        elif video_duration < 1800:  # < 30 minutes
            return VSTCompressionLevel.MODERATE
        elif video_duration < 3600:  # < 60 minutes
            return VSTCompressionLevel.AGGRESSIVE
        else:  # >= 60 minutes
            return VSTCompressionLevel.EXTREME
    
    @track_performance("scene_detection")
    async def detect_scene_changes(self, frames: List[VideoFrame]) -> List[int]:
        """
        Detect scene changes in video frames using visual similarity.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of frame indices where scene changes occur
        """
        if len(frames) < 2:
            return [0]
        
        scene_boundaries = [0]  # Always include first frame
        
        try:
            prev_features = None
            
            for i, frame in enumerate(frames):
                # Extract simple visual features
                img_array = np.array(frame.image)
                if len(img_array.shape) == 3:
                    # Convert to tensor and compute features
                    img_tensor = torch.from_numpy(img_array).float()
                    
                    # Color histogram features
                    r_hist = torch.histc(img_tensor[:,:,0], bins=8, min=0, max=255)
                    g_hist = torch.histc(img_tensor[:,:,1], bins=8, min=0, max=255)  
                    b_hist = torch.histc(img_tensor[:,:,2], bins=8, min=0, max=255)
                    
                    # Spatial features
                    gray = torch.mean(img_tensor, dim=2)
                    brightness = torch.mean(gray)
                    contrast = torch.std(gray)
                    
                    # Edge features
                    grad_x = torch.abs(torch.diff(gray, dim=1))
                    grad_y = torch.abs(torch.diff(gray, dim=0))
                    edge_density = torch.mean(grad_x) + torch.mean(grad_y)
                    
                    # Combine features
                    features = torch.cat([
                        F.normalize(r_hist, p=1, dim=0),
                        F.normalize(g_hist, p=1, dim=0),
                        F.normalize(b_hist, p=1, dim=0),
                        torch.tensor([brightness / 255.0, contrast / 255.0, edge_density / 255.0])
                    ])
                    
                    if prev_features is not None:
                        # Compute cosine similarity
                        similarity = F.cosine_similarity(features, prev_features, dim=0)
                        
                        # Scene change if similarity below threshold
                        if similarity < (1.0 - self.scene_change_threshold):
                            scene_boundaries.append(i)
                    
                    prev_features = features
            
            # Always include last frame
            if scene_boundaries[-1] != len(frames) - 1:
                scene_boundaries.append(len(frames) - 1)
            
            self.logger.debug(f"Scene detection: {len(scene_boundaries)} scenes in {len(frames)} frames")
            return scene_boundaries
            
        except Exception as e:
            self.logger.error(f"Scene detection failed: {e}")
            # Fallback: uniform sampling
            num_scenes = max(2, len(frames) // 30)
            return [i * len(frames) // num_scenes for i in range(num_scenes)]
    
    @track_performance("frame_importance")
    async def calculate_frame_importance(
        self, 
        frames: List[VideoFrame],
        query_category: QueryCategory
    ) -> List[float]:
        """
        Calculate importance scores for frames based on content and query category.
        
        Args:
            frames: List of video frames
            query_category: Query category for context-aware scoring
            
        Returns:
            List of importance scores (0.0 to 1.0)
        """
        if not frames:
            return []
        
        importance_scores = []
        
        try:
            for frame in frames:
                score = 0.0
                img_array = np.array(frame.image)
                
                if len(img_array.shape) == 3:
                    img_tensor = torch.from_numpy(img_array).float()
                    h, w, c = img_tensor.shape
                    
                    # Visual complexity score
                    gray = torch.mean(img_tensor, dim=2)
                    
                    # Edge density (indicates detail)
                    grad_x = torch.abs(torch.diff(gray, dim=1))
                    grad_y = torch.abs(torch.diff(gray, dim=0))
                    edge_score = (torch.mean(grad_x) + torch.mean(grad_y)) / 255.0
                    
                    # Color diversity
                    r_var = torch.var(img_tensor[:,:,0]) / (255.0 * 255.0)
                    g_var = torch.var(img_tensor[:,:,1]) / (255.0 * 255.0)
                    b_var = torch.var(img_tensor[:,:,2]) / (255.0 * 255.0)
                    color_diversity = (r_var + g_var + b_var) / 3.0
                    
                    # Contrast
                    contrast = torch.std(gray) / 255.0
                    
                    # Base importance from visual features
                    visual_importance = (edge_score * 0.4 + color_diversity * 0.3 + contrast * 0.3)
                    score += min(visual_importance.item(), 1.0) * 0.6
                    
                    # Category-specific boosting
                    if query_category == QueryCategory.ACTION_RECOGNITION:
                        # Boost frames with high motion (edge density)
                        score += min(edge_score.item(), 1.0) * 0.3
                    elif query_category == QueryCategory.OBJECT_DETECTION:
                        # Boost frames with distinct objects (high contrast)
                        score += min(contrast.item(), 1.0) * 0.3
                    elif query_category == QueryCategory.SCENE_UNDERSTANDING:
                        # Boost visually diverse frames
                        score += min(color_diversity.item(), 1.0) * 0.3
                    else:
                        # Default: balanced importance
                        score += 0.2
                    
                    # Temporal position bonus (beginning and end are often important)
                    frame_idx = frames.index(frame)
                    total_frames = len(frames)
                    if frame_idx < total_frames * 0.1 or frame_idx > total_frames * 0.9:
                        score += 0.1
                    
                else:
                    # Fallback score for invalid frames
                    score = 0.1
                
                importance_scores.append(min(score, 1.0))
            
            # Normalize scores to ensure proper distribution
            if importance_scores:
                max_score = max(importance_scores)
                if max_score > 0:
                    importance_scores = [s / max_score for s in importance_scores]
            
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Frame importance calculation failed: {e}")
            # Fallback: uniform importance
            return [0.5] * len(frames)
    
    @track_performance("vst_compression")
    async def compress_to_vst(
        self,
        frames: List[VideoFrame],
        query_category: QueryCategory,
        target_compression_level: Optional[VSTCompressionLevel] = None
    ) -> VSTCompression:
        """
        Compress video frames to Visual Summarization Tokens.
        
        Args:
            frames: Input video frames
            query_category: Query category for context-aware compression
            target_compression_level: Override automatic compression level
            
        Returns:
            VSTCompression result with tokens and metrics
        """
        start_time = time.time()
        
        if not frames:
            return VSTCompression(
                vst_tokens=[],
                original_frame_count=0,
                compressed_frame_count=0,
                compression_ratio=1.0,
                processing_time=0.0,
                compression_level=VSTCompressionLevel.MINIMAL
            )
        
        try:
            # Determine video duration and compression level
            video_duration = frames[-1].timestamp - frames[0].timestamp if len(frames) > 1 else 0
            compression_level = target_compression_level or self.determine_compression_level(video_duration)
            
            # Calculate target number of VST tokens
            target_tokens = max(1, int(video_duration / 60.0 * self.max_tokens_per_minute[compression_level]))
            
            self.logger.info(f"VST compression: {len(frames)} frames -> {target_tokens} tokens "
                           f"(level: {compression_level.value}, duration: {video_duration:.1f}s)")
            
            # Detect scene boundaries
            scene_boundaries = await self.detect_scene_changes(frames)
            
            # Calculate frame importance scores
            importance_scores = await self.calculate_frame_importance(frames, query_category)
            
            # Create segments based on scene boundaries
            segments = []
            for i in range(len(scene_boundaries) - 1):
                start_idx = scene_boundaries[i]
                end_idx = scene_boundaries[i + 1]
                segment_frames = frames[start_idx:end_idx + 1]
                segment_scores = importance_scores[start_idx:end_idx + 1]
                
                if segment_frames:
                    segments.append({
                        'frames': segment_frames,
                        'scores': segment_scores,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'avg_importance': sum(segment_scores) / len(segment_scores)
                    })
            
            # Sort segments by importance and select top segments
            segments.sort(key=lambda x: x['avg_importance'], reverse=True)
            
            # Distribute tokens across segments based on importance
            vst_tokens = []
            
            for i, segment in enumerate(segments):
                if len(vst_tokens) >= target_tokens:
                    break
                
                segment_frames = segment['frames']
                segment_scores = segment['scores']
                
                # Select most important frame from segment as representative
                max_score_idx = segment_scores.index(max(segment_scores))
                representative_frame = segment_frames[max_score_idx]
                
                # Create semantic features for the segment
                semantic_features = await self.extract_segment_features(segment_frames)
                
                # Analyze segment content
                scene_analysis = await self.analyze_segment_content(segment_frames)
                
                # Create VST token
                vst_token = VSTToken(
                    token_id=f"vst_{i:04d}_{representative_frame.timestamp:.1f}",
                    start_time=segment_frames[0].timestamp,
                    end_time=segment_frames[-1].timestamp,
                    frame_count=len(segment_frames),
                    representative_frame=representative_frame,
                    semantic_features=semantic_features,
                    importance_score=segment['avg_importance'],
                    compression_ratio=len(frames) / target_tokens,
                    scene_type=scene_analysis['scene_type'],
                    motion_level=scene_analysis['motion_level'],
                    visual_complexity=scene_analysis['visual_complexity']
                )
                
                vst_tokens.append(vst_token)
            
            # Sort tokens by timestamp for temporal ordering
            vst_tokens.sort(key=lambda x: x.start_time)
            
            # Calculate compression metrics
            processing_time = time.time() - start_time
            compression_ratio = len(frames) / max(1, len(vst_tokens))
            
            # Quality assessment
            temporal_coverage = self.calculate_temporal_coverage(vst_tokens, video_duration)
            semantic_preservation = self.estimate_semantic_preservation(vst_tokens, frames)
            information_density = len(vst_tokens) / max(1, video_duration / 60.0)
            
            vst_compression = VSTCompression(
                vst_tokens=vst_tokens,
                original_frame_count=len(frames),
                compressed_frame_count=len(vst_tokens),
                compression_ratio=compression_ratio,
                processing_time=processing_time,
                compression_level=compression_level,
                semantic_preservation=semantic_preservation,
                temporal_coverage=temporal_coverage,
                information_density=information_density
            )
            
            self.logger.info(f"VST compression completed: {len(frames)} -> {len(vst_tokens)} tokens "
                           f"({compression_ratio:.1f}x compression) in {processing_time:.3f}s")
            
            return vst_compression
            
        except Exception as e:
            self.logger.error(f"VST compression failed: {e}")
            # Fallback: create minimal tokens from uniform sampling
            fallback_tokens = []
            num_fallback = min(target_tokens or 5, len(frames))
            
            for i in range(num_fallback):
                frame_idx = i * len(frames) // num_fallback
                frame = frames[frame_idx]
                
                fallback_token = VSTToken(
                    token_id=f"fallback_{i:04d}",
                    start_time=frame.timestamp,
                    end_time=frame.timestamp,
                    frame_count=1,
                    representative_frame=frame,
                    semantic_features=torch.zeros(64).to(self.device),
                    importance_score=0.5,
                    compression_ratio=len(frames) / num_fallback
                )
                fallback_tokens.append(fallback_token)
            
            return VSTCompression(
                vst_tokens=fallback_tokens,
                original_frame_count=len(frames),
                compressed_frame_count=len(fallback_tokens),
                compression_ratio=len(frames) / max(1, len(fallback_tokens)),
                processing_time=time.time() - start_time,
                compression_level=compression_level,
                semantic_preservation=0.7,
                temporal_coverage=0.8,
                information_density=1.0
            )
    
    async def extract_segment_features(self, segment_frames: List[VideoFrame]) -> torch.Tensor:
        """Extract semantic features for a video segment."""
        try:
            if not segment_frames:
                return torch.zeros(64).to(self.device)
            
            # Simple feature extraction for segment
            features = []
            
            for frame in segment_frames:
                img_array = np.array(frame.image)
                if len(img_array.shape) == 3:
                    img_tensor = torch.from_numpy(img_array).float()
                    
                    # Basic visual statistics
                    mean_rgb = torch.mean(img_tensor, dim=(0, 1))
                    std_rgb = torch.std(img_tensor, dim=(0, 1))
                    features.extend(mean_rgb.tolist())
                    features.extend(std_rgb.tolist())
            
            # Aggregate segment features
            if features:
                # Pad or truncate to 64 dimensions
                if len(features) > 64:
                    features = features[:64]
                elif len(features) < 64:
                    features.extend([0.0] * (64 - len(features)))
                
                return torch.tensor(features, dtype=torch.float32).to(self.device)
            else:
                return torch.zeros(64).to(self.device)
                
        except Exception:
            return torch.zeros(64).to(self.device)
    
    async def analyze_segment_content(self, segment_frames: List[VideoFrame]) -> Dict[str, Any]:
        """Analyze content characteristics of a video segment."""
        try:
            if not segment_frames:
                return {
                    'scene_type': 'unknown',
                    'motion_level': 'low',
                    'visual_complexity': 0.0
                }
            
            # Motion analysis (frame-to-frame differences)
            motion_scores = []
            for i in range(1, len(segment_frames)):
                prev_img = np.array(segment_frames[i-1].image)
                curr_img = np.array(segment_frames[i].image)
                
                if prev_img.shape == curr_img.shape:
                    diff = np.mean(np.abs(curr_img.astype(float) - prev_img.astype(float)))
                    motion_scores.append(diff / 255.0)
            
            avg_motion = sum(motion_scores) / max(1, len(motion_scores))
            
            # Motion level classification
            if avg_motion > 0.3:
                motion_level = 'high'
            elif avg_motion > 0.1:
                motion_level = 'medium'
            else:
                motion_level = 'low'
            
            # Visual complexity (edge density of representative frame)
            mid_frame = segment_frames[len(segment_frames) // 2]
            img_array = np.array(mid_frame.image)
            
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                visual_complexity = (np.mean(grad_x) + np.mean(grad_y)) / 255.0
            else:
                visual_complexity = 0.0
            
            # Scene type classification (simple heuristic)
            if visual_complexity > 0.4:
                scene_type = 'complex'
            elif avg_motion > 0.2:
                scene_type = 'action'
            else:
                scene_type = 'static'
            
            return {
                'scene_type': scene_type,
                'motion_level': motion_level,
                'visual_complexity': min(visual_complexity, 1.0)
            }
            
        except Exception:
            return {
                'scene_type': 'unknown',
                'motion_level': 'low', 
                'visual_complexity': 0.0
            }
    
    def calculate_temporal_coverage(self, vst_tokens: List[VSTToken], total_duration: float) -> float:
        """Calculate temporal coverage of VST tokens."""
        if not vst_tokens or total_duration <= 0:
            return 0.0
        
        covered_duration = sum(token.duration() for token in vst_tokens)
        return min(covered_duration / total_duration, 1.0)
    
    def estimate_semantic_preservation(self, vst_tokens: List[VSTToken], original_frames: List[VideoFrame]) -> float:
        """Estimate how well VST tokens preserve semantic information."""
        if not vst_tokens or not original_frames:
            return 0.0
        
        # Heuristic based on compression ratio and importance scores
        avg_importance = sum(token.importance_score for token in vst_tokens) / len(vst_tokens)
        compression_ratio = len(original_frames) / len(vst_tokens)
        
        # Penalize extreme compression, reward high importance preservation
        preservation = avg_importance * (1.0 - min(compression_ratio / 100.0, 0.5))
        return max(min(preservation, 1.0), 0.3)  # Clamp between 0.3 and 1.0
    
    def get_vst_frames_for_gemini(self, vst_compression: VSTCompression) -> List[VideoFrame]:
        """Extract representative frames from VST tokens for Gemini processing."""
        return [token.representative_frame for token in vst_compression.vst_tokens]
    
    def enhance_prompt_with_vst_context(self, base_prompt: str, vst_compression: VSTCompression) -> str:
        """Enhance prompt with VST compression context."""
        if not vst_compression.vst_tokens:
            return base_prompt
        
        vst_context = "\n\nVIDEO STRUCTURE (Visual Summarization Tokens):\n"
        
        for i, token in enumerate(vst_compression.vst_tokens, 1):
            time_range = f"{token.start_time:.1f}s-{token.end_time:.1f}s"
            vst_context += f"Token {i} [{time_range}]: {token.scene_type} scene with {token.motion_level} motion "
            vst_context += f"(complexity: {token.visual_complexity:.2f}, importance: {token.importance_score:.2f})\n"
        
        vst_context += f"\nCompression: {vst_compression.original_frame_count} frames -> "
        vst_context += f"{len(vst_compression.vst_tokens)} tokens ({vst_compression.compression_ratio:.1f}x)\n"
        vst_context += "Focus on key moments represented by these visual tokens.\n"
        
        return base_prompt + vst_context


# Global VST processor instance
_vst_processor: Optional[VSTProcessor] = None


def get_vst_processor() -> VSTProcessor:
    """Get or create the global VST processor instance."""
    global _vst_processor
    if _vst_processor is None:
        _vst_processor = VSTProcessor()
    return _vst_processor


def initialize_vst_processor(config=None) -> VSTProcessor:
    """Initialize the global VST processor with custom config."""
    global _vst_processor
    _vst_processor = VSTProcessor(config)
    return _vst_processor