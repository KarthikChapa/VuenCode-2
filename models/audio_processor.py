"""
Advanced audio processing module for VuenCode Phase 2.
Implements Whisper-based speech recognition and audio feature extraction
for true multimodal video understanding.
"""

import asyncio
import logging
import tempfile
import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
import torchaudio
import whisper

from ..utils import get_config, track_performance


logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Represents extracted audio features and metadata."""
    waveform: torch.Tensor  # Raw audio waveform
    sample_rate: int        # Audio sample rate
    duration: float         # Audio duration in seconds
    embeddings: Optional[torch.Tensor] = None  # Audio embeddings
    transcript: Optional[str] = None           # Speech transcription
    confidence: float = 1.0                   # Processing confidence
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AudioSegment:
    """Represents a time-aligned audio segment."""
    start_time: float      # Start time in seconds
    end_time: float        # End time in seconds
    text: str             # Transcribed text
    confidence: float     # Transcription confidence
    embeddings: torch.Tensor  # Segment embeddings
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class AudioProcessor:
    """
    Competition-grade audio processing system with Whisper integration.
    Supports speech recognition, audio feature extraction, and embedding generation.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Whisper model
        self.whisper_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Audio processing parameters
        self.target_sample_rate = 16000  # Whisper's preferred sample rate
        self.chunk_duration = 30.0       # 30-second chunks for long audio
        
        # Initialize model lazily
        self._model_loaded = False
        
        self.logger.info(f"AudioProcessor initialized on device: {self.device}")
    
    def _load_whisper_model(self):
        """Load Whisper model on first use."""
        if not self._model_loaded:
            try:
                # Use appropriate Whisper model based on config
                model_size = "base" if self.config.is_local_mode else "small"
                self.whisper_model = whisper.load_model(model_size, device=self.device)
                self._model_loaded = True
                self.logger.info(f"Whisper {model_size} model loaded on {self.device}")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    @track_performance("audio_extraction")
    async def extract_audio_from_video(self, video_data: bytes) -> AudioFeatures:
        """
        Extract audio track from video data.
        
        Args:
            video_data: Video data as bytes
            
        Returns:
            AudioFeatures object with extracted audio
        """
        try:
            # Save video to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(video_data)
                temp_path = temp_file.name
            
            try:
                # Load audio using torchaudio
                waveform, sample_rate = torchaudio.load(temp_path, format="mp4")
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample to target sample rate if needed
                if sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=self.target_sample_rate
                    )
                    waveform = resampler(waveform)
                    sample_rate = self.target_sample_rate
                
                duration = waveform.shape[1] / sample_rate
                
                audio_features = AudioFeatures(
                    waveform=waveform.squeeze(0),  # Remove channel dimension
                    sample_rate=sample_rate,
                    duration=duration,
                    metadata={
                        "extraction_method": "torchaudio",
                        "original_format": "mp4",
                        "channels": "mono"
                    }
                )
                
                self.logger.info(f"Audio extracted: {duration:.2f}s at {sample_rate}Hz")
                return audio_features
                
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)
                
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            # Return empty audio features as fallback
            return AudioFeatures(
                waveform=torch.zeros(1),
                sample_rate=self.target_sample_rate,
                duration=0.0,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    @track_performance("speech_recognition")
    async def transcribe_speech(self, audio_features: AudioFeatures) -> List[AudioSegment]:
        """
        Transcribe speech from audio using Whisper.
        
        Args:
            audio_features: Audio features to transcribe
            
        Returns:
            List of time-aligned transcription segments
        """
        try:
            # Load Whisper model if not already loaded
            self._load_whisper_model()
            
            if audio_features.duration == 0.0 or audio_features.confidence == 0.0:
                return []
            
            # Convert tensor to numpy for Whisper
            audio_np = audio_features.waveform.cpu().numpy()
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(
                audio_np,
                task="transcribe",
                language=None,  # Auto-detect language
                word_timestamps=True,
                condition_on_previous_text=False
            )
            
            # Process segments
            segments = []
            for segment in result.get("segments", []):
                # Generate embeddings for this segment (simplified approach)
                start_sample = int(segment["start"] * audio_features.sample_rate)
                end_sample = int(segment["end"] * audio_features.sample_rate)
                
                segment_audio = audio_features.waveform[start_sample:end_sample]
                
                # Simple embedding: mean of audio features (in practice, use proper encoder)
                embeddings = torch.mean(segment_audio.unsqueeze(0), dim=1)
                
                audio_segment = AudioSegment(
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text=segment["text"].strip(),
                    confidence=segment.get("confidence", 0.9),
                    embeddings=embeddings
                )
                segments.append(audio_segment)
            
            # Store transcript in audio features
            audio_features.transcript = result.get("text", "").strip()
            
            self.logger.info(f"Speech transcribed: {len(segments)} segments, {len(audio_features.transcript)} chars")
            return segments
            
        except Exception as e:
            self.logger.error(f"Speech transcription failed: {e}")
            return []
    
    @track_performance("audio_embeddings")
    async def generate_audio_embeddings(self, audio_features: AudioFeatures) -> torch.Tensor:
        """
        Generate audio embeddings for multimodal fusion.
        
        Args:
            audio_features: Audio features to encode
            
        Returns:
            Audio embeddings tensor
        """
        try:
            if audio_features.duration == 0.0 or audio_features.confidence == 0.0:
                # Return zero embeddings for silent/failed audio
                return torch.zeros(768)  # Standard embedding dimension
            
            # Simple audio feature extraction (in production, use proper audio encoder)
            waveform = audio_features.waveform
            
            # Extract basic audio features
            features = []
            
            # RMS energy
            rms = torch.sqrt(torch.mean(waveform ** 2))
            features.append(rms)
            
            # Zero crossing rate
            zero_crossings = torch.sum(torch.diff(torch.sign(waveform)) != 0).float()
            zcr = zero_crossings / len(waveform)
            features.append(zcr)
            
            # Spectral features using FFT
            fft = torch.fft.fft(waveform)
            magnitude = torch.abs(fft)[:len(fft)//2]  # Keep only positive frequencies
            
            # Spectral centroid
            freqs = torch.linspace(0, audio_features.sample_rate/2, len(magnitude))
            spectral_centroid = torch.sum(freqs * magnitude) / torch.sum(magnitude)
            features.append(spectral_centroid / audio_features.sample_rate)  # Normalize
            
            # Spectral rolloff
            cumsum = torch.cumsum(magnitude, dim=0)
            rolloff_point = 0.85 * cumsum[-1]
            rolloff_idx = torch.where(cumsum >= rolloff_point)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            features.append(spectral_rolloff / audio_features.sample_rate)  # Normalize
            
            # Convert to tensor and expand to standard embedding size
            basic_features = torch.stack(features)
            
            # Expand to 768 dimensions (repeat and add noise for diversity)
            embedding_size = 768
            expanded = torch.zeros(embedding_size)
            
            for i in range(embedding_size):
                feature_idx = i % len(basic_features)
                noise = torch.randn(1) * 0.01  # Small noise for diversity
                expanded[i] = basic_features[feature_idx] + noise
            
            audio_features.embeddings = expanded
            
            self.logger.debug(f"Audio embeddings generated: {expanded.shape}")
            return expanded
            
        except Exception as e:
            self.logger.error(f"Audio embedding generation failed: {e}")
            # Return zero embeddings as fallback
            return torch.zeros(768)
    
    @track_performance("audio_processing_full")
    async def process_audio_full(self, video_data: bytes) -> Tuple[AudioFeatures, List[AudioSegment]]:
        """
        Complete audio processing pipeline: extraction + transcription + embeddings.
        
        Args:
            video_data: Video data containing audio track
            
        Returns:
            Tuple of (AudioFeatures with embeddings, transcription segments)
        """
        # Extract audio
        audio_features = await self.extract_audio_from_video(video_data)
        
        # Generate embeddings
        await self.generate_audio_embeddings(audio_features)
        
        # Transcribe speech
        segments = await self.transcribe_speech(audio_features)
        
        return audio_features, segments
    
    def get_audio_summary(self, audio_features: AudioFeatures, segments: List[AudioSegment]) -> str:
        """Generate a summary of audio content for logging/debugging."""
        if not segments:
            return "No speech detected"
        
        total_speech = sum(seg.duration for seg in segments)
        speech_ratio = total_speech / audio_features.duration if audio_features.duration > 0 else 0
        
        return f"{len(segments)} speech segments, {total_speech:.1f}s/{audio_features.duration:.1f}s speech ({speech_ratio:.1%})"


# Global processor instance
_audio_processor: Optional[AudioProcessor] = None


def get_audio_processor() -> AudioProcessor:
    """Get or create the global audio processor instance."""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor


def initialize_audio_processor(config=None) -> AudioProcessor:
    """Initialize the global audio processor with custom config."""
    global _audio_processor
    _audio_processor = AudioProcessor(config)
    return _audio_processor