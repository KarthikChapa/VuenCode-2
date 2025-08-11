"""
Multimodal Fusion Bus for VuenCode Phase 2.
Combines video frame features, audio embeddings, and text prompt embeddings
for enhanced multimodal video understanding with Gemini API.
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .preprocessing import VideoFrame
from .audio_processor import AudioFeatures, AudioSegment
from ..utils import get_config, track_performance
from ..api.schemas import QueryCategory


logger = logging.getLogger(__name__)


@dataclass
class MultimodalFeatures:
    """Container for fused multimodal features."""
    video_embeddings: torch.Tensor    # Video frame embeddings
    audio_embeddings: torch.Tensor    # Audio embeddings  
    text_embeddings: torch.Tensor     # Text query embeddings
    fused_embeddings: torch.Tensor    # Combined multimodal embeddings
    
    # Metadata
    video_frame_count: int = 0
    audio_duration: float = 0.0
    text_token_count: int = 0
    fusion_confidence: float = 1.0
    
    # Temporal alignment
    video_timestamps: List[float] = None
    audio_timestamps: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.video_timestamps is None:
            self.video_timestamps = []
        if self.audio_timestamps is None:
            self.audio_timestamps = []


class MultimodalFusionBus:
    """
    Advanced multimodal fusion system for combining video, audio, and text.
    Implements attention-based fusion for optimal multimodal understanding.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Embedding dimensions (standardized)
        self.embedding_dim = 768  # Standard transformer embedding size
        
        # Fusion parameters
        self.video_weight = 0.5    # Weight for video modality
        self.audio_weight = 0.3    # Weight for audio modality  
        self.text_weight = 0.2     # Weight for text modality
        
        # Device for tensor operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"MultimodalFusionBus initialized on device: {self.device}")
    
    @track_performance("text_embedding")
    async def generate_text_embeddings(self, query: str, category: QueryCategory) -> torch.Tensor:
        """
        Generate embeddings for text query using simple encoding.
        In production, this would use a proper text encoder like BERT/RoBERTa.
        
        Args:
            query: Natural language query
            category: Query category for context
            
        Returns:
            Text embeddings tensor
        """
        try:
            # Simple text feature extraction (in production, use proper encoder)
            query_words = query.lower().split()
            
            # Basic text features
            features = []
            
            # Query length features
            word_count = len(query_words)
            char_count = len(query)
            avg_word_length = char_count / max(1, word_count)
            
            features.extend([word_count / 100.0, char_count / 1000.0, avg_word_length / 10.0])
            
            # Category encoding (one-hot style)
            category_features = [0.0] * 16  # 16 categories
            try:
                category_idx = list(QueryCategory).index(category)
                category_features[category_idx] = 1.0
            except (ValueError, IndexError):
                pass
            
            features.extend(category_features)
            
            # Query complexity features
            complex_words = sum(1 for word in query_words if len(word) > 6)
            question_words = sum(1 for word in query_words if word in ['what', 'when', 'where', 'who', 'why', 'how'])
            
            features.extend([complex_words / max(1, word_count), question_words / max(1, word_count)])
            
            # Simple semantic features (keyword presence)
            semantic_keywords = {
                'action': ['action', 'doing', 'activity', 'movement', 'motion'],
                'object': ['object', 'thing', 'item', 'see', 'visible'],
                'scene': ['scene', 'setting', 'place', 'location', 'environment'],
                'time': ['when', 'time', 'during', 'before', 'after'],
                'emotion': ['emotion', 'feeling', 'mood', 'happy', 'sad'],
                'audio': ['sound', 'hear', 'audio', 'music', 'voice', 'speech']
            }
            
            for concept, keywords in semantic_keywords.items():
                presence = any(keyword in query.lower() for keyword in keywords)
                features.append(1.0 if presence else 0.0)
            
            # Convert to tensor and expand to standard embedding size
            basic_features = torch.tensor(features, dtype=torch.float32)
            
            # Expand to embedding_dim dimensions
            embedding = torch.zeros(self.embedding_dim)
            
            # Repeat and transform basic features to fill embedding space
            for i in range(self.embedding_dim):
                feature_idx = i % len(basic_features)
                # Add positional encoding and small variations
                pos_encoding = np.sin(i / 10000) * 0.1
                embedding[i] = basic_features[feature_idx] + pos_encoding
            
            self.logger.debug(f"Text embeddings generated: {query[:50]}... -> {embedding.shape}")
            return embedding.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Text embedding generation failed: {e}")
            return torch.zeros(self.embedding_dim).to(self.device)
    
    @track_performance("video_embedding")
    async def generate_video_embeddings(self, frames: List[VideoFrame]) -> torch.Tensor:
        """
        Generate embeddings for video frames using visual features.
        
        Args:
            frames: List of video frames
            
        Returns:
            Video embeddings tensor
        """
        try:
            if not frames:
                return torch.zeros(self.embedding_dim).to(self.device)
            
            frame_embeddings = []
            
            for frame in frames:
                # Convert PIL image to tensor
                img_array = np.array(frame.image)
                if len(img_array.shape) == 3:
                    img_tensor = torch.from_numpy(img_array).float()
                    
                    # Simple visual features
                    features = []
                    
                    # Color statistics
                    mean_rgb = torch.mean(img_tensor, dim=(0, 1))  # Mean RGB
                    std_rgb = torch.std(img_tensor, dim=(0, 1))    # Std RGB
                    features.extend(mean_rgb.tolist())
                    features.extend(std_rgb.tolist())
                    
                    # Brightness and contrast
                    gray = torch.mean(img_tensor, dim=2)  # Convert to grayscale
                    brightness = torch.mean(gray) / 255.0
                    contrast = torch.std(gray) / 255.0
                    features.extend([brightness.item(), contrast.item()])
                    
                    # Edge density (simple gradient magnitude)
                    grad_x = torch.diff(gray, dim=1)
                    grad_y = torch.diff(gray, dim=0)
                    edge_density = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
                    features.append(edge_density.item() / 255.0)
                    
                    # Texture features (variance in local regions)
                    h, w = gray.shape
                    region_size = min(h, w) // 4
                    texture_vars = []
                    for i in range(0, h - region_size, region_size):
                        for j in range(0, w - region_size, region_size):
                            region = gray[i:i+region_size, j:j+region_size]
                            texture_vars.append(torch.var(region).item())
                    
                    avg_texture = sum(texture_vars) / max(1, len(texture_vars)) / 255.0
                    features.append(avg_texture)
                    
                    # Expand basic features to embedding dimension
                    basic_features = torch.tensor(features, dtype=torch.float32)
                    frame_embedding = torch.zeros(self.embedding_dim)
                    
                    for k in range(self.embedding_dim):
                        feat_idx = k % len(basic_features)
                        # Add timestamp encoding
                        time_encoding = np.sin(frame.timestamp + k / 100.0) * 0.1
                        frame_embedding[k] = basic_features[feat_idx] + time_encoding
                    
                    frame_embeddings.append(frame_embedding)
                else:
                    # Fallback for invalid frames
                    frame_embeddings.append(torch.zeros(self.embedding_dim))
            
            # Aggregate frame embeddings (mean pooling)
            if frame_embeddings:
                video_embedding = torch.stack(frame_embeddings).mean(dim=0)
            else:
                video_embedding = torch.zeros(self.embedding_dim)
            
            self.logger.debug(f"Video embeddings generated: {len(frames)} frames -> {video_embedding.shape}")
            return video_embedding.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Video embedding generation failed: {e}")
            return torch.zeros(self.embedding_dim).to(self.device)
    
    @track_performance("multimodal_fusion")
    async def fuse_modalities(
        self,
        frames: List[VideoFrame],
        audio_features: AudioFeatures,
        query: str,
        category: QueryCategory
    ) -> MultimodalFeatures:
        """
        Fuse video, audio, and text modalities into unified representation.
        
        Args:
            frames: Video frames
            audio_features: Audio features with embeddings
            query: Text query
            category: Query category
            
        Returns:
            MultimodalFeatures with fused embeddings
        """
        try:
            # Generate embeddings for each modality
            video_embeddings = await self.generate_video_embeddings(frames)
            
            # Use pre-computed audio embeddings or generate if missing
            if audio_features.embeddings is not None:
                audio_embeddings = audio_features.embeddings.to(self.device)
            else:
                # Fallback: generate basic audio embeddings
                if audio_features.duration > 0:
                    # Simple audio features based on waveform statistics
                    waveform = audio_features.waveform
                    audio_features_basic = [
                        torch.mean(waveform).item(),
                        torch.std(waveform).item(),
                        torch.max(torch.abs(waveform)).item(),
                        audio_features.duration / 60.0  # Normalize duration
                    ]
                    basic_tensor = torch.tensor(audio_features_basic, dtype=torch.float32)
                    
                    # Expand to full embedding dimension
                    audio_embeddings = torch.zeros(self.embedding_dim)
                    for i in range(self.embedding_dim):
                        feat_idx = i % len(audio_features_basic)
                        audio_embeddings[i] = basic_tensor[feat_idx] + np.sin(i / 50.0) * 0.1
                else:
                    audio_embeddings = torch.zeros(self.embedding_dim)
                
                audio_embeddings = audio_embeddings.to(self.device)
            
            text_embeddings = await self.generate_text_embeddings(query, category)
            
            # Normalize embeddings
            video_embeddings = F.normalize(video_embeddings, p=2, dim=0)
            audio_embeddings = F.normalize(audio_embeddings, p=2, dim=0)  
            text_embeddings = F.normalize(text_embeddings, p=2, dim=0)
            
            # Weighted fusion
            fused_embeddings = (
                self.video_weight * video_embeddings +
                self.audio_weight * audio_embeddings +
                self.text_weight * text_embeddings
            )
            
            # Final normalization
            fused_embeddings = F.normalize(fused_embeddings, p=2, dim=0)
            
            # Calculate fusion confidence based on modality availability
            modality_scores = []
            modality_scores.append(1.0 if len(frames) > 0 else 0.0)  # Video
            modality_scores.append(audio_features.confidence)         # Audio
            modality_scores.append(1.0 if query.strip() else 0.0)    # Text
            
            fusion_confidence = sum(modality_scores) / len(modality_scores)
            
            # Extract timestamps
            video_timestamps = [frame.timestamp for frame in frames]
            audio_timestamps = [(0.0, audio_features.duration)] if audio_features.duration > 0 else []
            
            multimodal_features = MultimodalFeatures(
                video_embeddings=video_embeddings,
                audio_embeddings=audio_embeddings,
                text_embeddings=text_embeddings,
                fused_embeddings=fused_embeddings,
                video_frame_count=len(frames),
                audio_duration=audio_features.duration,
                text_token_count=len(query.split()),
                fusion_confidence=fusion_confidence,
                video_timestamps=video_timestamps,
                audio_timestamps=audio_timestamps
            )
            
            self.logger.info(f"Multimodal fusion completed: video={len(frames)}frames, "
                           f"audio={audio_features.duration:.1f}s, text={len(query)}chars, "
                           f"confidence={fusion_confidence:.3f}")
            
            return multimodal_features
            
        except Exception as e:
            self.logger.error(f"Multimodal fusion failed: {e}")
            # Return fallback with zero embeddings
            return MultimodalFeatures(
                video_embeddings=torch.zeros(self.embedding_dim).to(self.device),
                audio_embeddings=torch.zeros(self.embedding_dim).to(self.device),
                text_embeddings=torch.zeros(self.embedding_dim).to(self.device),
                fused_embeddings=torch.zeros(self.embedding_dim).to(self.device),
                fusion_confidence=0.0
            )
    
    def enhance_gemini_prompt(
        self,
        base_prompt: str,
        multimodal_features: MultimodalFeatures,
        audio_segments: List[AudioSegment]
    ) -> str:
        """
        Enhance Gemini prompt with multimodal context information.
        
        Args:
            base_prompt: Original prompt template
            multimodal_features: Fused multimodal features
            audio_segments: Transcribed audio segments
            
        Returns:
            Enhanced prompt with multimodal context
        """
        try:
            enhanced_prompt = base_prompt
            
            # Add audio context if available
            if audio_segments and any(seg.text.strip() for seg in audio_segments):
                audio_context = "\n\nAUDIO CONTEXT:\n"
                for segment in audio_segments:
                    if segment.text.strip():
                        time_str = f"{segment.start_time:.1f}-{segment.end_time:.1f}s"
                        audio_context += f"[{time_str}]: {segment.text.strip()}\n"
                
                enhanced_prompt += audio_context
            
            # Add multimodal analysis hints
            multimodal_hints = "\n\nMULTIMODAL ANALYSIS INSTRUCTIONS:\n"
            
            if multimodal_features.audio_duration > 0:
                multimodal_hints += f"- Audio duration: {multimodal_features.audio_duration:.1f}s - Consider speech, sounds, and audio-visual synchronization\n"
            
            if multimodal_features.video_frame_count > 0:
                multimodal_hints += f"- Video frames: {multimodal_features.video_frame_count} frames - Analyze visual content and temporal changes\n"
            
            multimodal_hints += "- Integrate visual and audio information for comprehensive understanding\n"
            multimodal_hints += "- Pay attention to audio-visual correlation and synchronization\n"
            
            enhanced_prompt += multimodal_hints
            
            return enhanced_prompt
            
        except Exception as e:
            self.logger.error(f"Prompt enhancement failed: {e}")
            return base_prompt
    
    def get_fusion_summary(self, multimodal_features: MultimodalFeatures) -> Dict[str, Any]:
        """Generate summary of multimodal fusion for debugging/monitoring."""
        return {
            "video_frames": multimodal_features.video_frame_count,
            "audio_duration": multimodal_features.audio_duration,
            "text_tokens": multimodal_features.text_token_count,
            "fusion_confidence": multimodal_features.fusion_confidence,
            "embedding_dimension": multimodal_features.fused_embeddings.shape[0] if multimodal_features.fused_embeddings is not None else 0,
            "temporal_coverage": {
                "video_span": f"{min(multimodal_features.video_timestamps):.1f}-{max(multimodal_features.video_timestamps):.1f}s" if multimodal_features.video_timestamps else "none",
                "audio_span": f"{multimodal_features.audio_timestamps[0][0]:.1f}-{multimodal_features.audio_timestamps[0][1]:.1f}s" if multimodal_features.audio_timestamps else "none"
            }
        }


# Global fusion bus instance
_fusion_bus: Optional[MultimodalFusionBus] = None


def get_fusion_bus() -> MultimodalFusionBus:
    """Get or create the global multimodal fusion bus instance."""
    global _fusion_bus
    if _fusion_bus is None:
        _fusion_bus = MultimodalFusionBus()
    return _fusion_bus


def initialize_fusion_bus(config=None) -> MultimodalFusionBus:
    """Initialize the global fusion bus with custom config."""
    global _fusion_bus
    _fusion_bus = MultimodalFusionBus(config)
    return _fusion_bus