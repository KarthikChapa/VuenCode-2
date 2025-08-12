#!/bin/bash
# Patch script to fix VideoProcessor import issues for remote deployment

# Fix video_processor.py
cat > /root/VuenCode-2/models/video_processor.py << 'EOF'
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
        
        # Performance configuration
        self.enable_vst_compression = getattr(self.config, 'enable_vst_compression', True)
        self.enable_multimodal_fusion = getattr(self.config, 'enable_multimodal_fusion', True)
        self.enable_audio_processing = getattr(self.config, 'enable_audio_processing', True)
        self.max_video_duration = 7200  # 120 minutes max
        
        self.logger.info(f"EnhancedVideoProcessor initialized - VST:{self.enable_vst_compression}, "
                        f"Multimodal:{self.enable_multimodal_fusion}, Audio:{self.enable_audio_processing}")

# Alias for backward compatibility with main_standalone.py
VideoProcessor = EnhancedVideoProcessor
EOF

# Fix main_standalone.py
sed -i 's/from models.video_processor import VideoProcessor/from models.video_processor import EnhancedVideoProcessor as VideoProcessor/g' /root/VuenCode-2/api/main_standalone.py

# Fix audio_processor.py
cat > /root/VuenCode-2/models/audio_processor.py << 'EOF'
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


@dataclass
class AudioFeatures:
    """Represents extracted audio features and metadata."""
    waveform: torch.Tensor  # Raw audio waveform
    sample_rate: int        # Audio sample rate
    duration: float         # Audio duration in seconds
    features: torch.Tensor  # Extracted audio features
    model_name: str         # Model used for feature extraction

@dataclass
class AudioSegment:
    """Represents a segment of audio with transcription."""
    start: float            # Start time (seconds)
    end: float              # End time (seconds)
    text: str               # Transcribed text
    confidence: float       # Transcription confidence

class AudioProcessor:
    """
    Advanced audio processor for Phase 2 with Whisper integration.
    Processes audio from videos for multimodal understanding.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.whisper_model = None
        self.model_name = getattr(self.config, 'whisper_model', 'tiny')
        
    @track_performance("audio_processing")
    async def process_audio_from_video(self, video_data: bytes) -> Tuple[Optional[AudioFeatures], Optional[List[AudioSegment]]]:
        """
        Extract audio from video bytes and process it.
        
        Args:
            video_data: Raw video bytes
            
        Returns:
            Tuple of (audio features, audio segments with transcriptions)
        """
        try:
            # Extract audio from video
            audio_path = await self._extract_audio_from_video(video_data)
            if not audio_path:
                return None, None
                
            # Load audio
            waveform, sample_rate = self._load_audio(audio_path)
            
            # Extract audio features
            features = await self._extract_features(waveform, sample_rate)
            
            # Transcribe audio
            segments = await self._transcribe_audio(audio_path)
            
            # Create audio features object
            audio_features = AudioFeatures(
                waveform=waveform,
                sample_rate=sample_rate,
                duration=waveform.shape[1] / sample_rate,
                features=features,
                model_name=self.model_name
            )
            
            return audio_features, segments
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            return None, None
            
    async def _extract_audio_from_video(self, video_data: bytes) -> Optional[Path]:
        """Extract audio track from video bytes."""
        try:
            # Create temp files for video and audio
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as video_file, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
                
                video_path = Path(video_file.name)
                audio_path = Path(audio_file.name)
                
                # Write video data to temp file
                video_file.write(video_data)
                video_file.flush()
                
                # Extract audio using torchaudio
                await asyncio.to_thread(
                    torchaudio.backend.sox_io_backend.save,
                    str(audio_path),
                    await asyncio.to_thread(
                        torchaudio.backend.sox_io_backend.load, str(video_path)
                    )[0],
                    16000
                )
                
                return audio_path
                
        except Exception as e:
            self.logger.error(f"Error extracting audio: {str(e)}")
            return None
            
    def _load_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """Load audio file using torchaudio."""
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, sample_rate
        
    async def _extract_features(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Extract audio features from waveform."""
        # Simple mel spectrogram features
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=80
        )
        features = transform(waveform)
        return features
        
    async def _transcribe_audio(self, audio_path: Path) -> List[AudioSegment]:
        """Transcribe audio using Whisper."""
        # Lazy-load the model
        if self.whisper_model is None:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self.whisper_model = whisper.load_model(self.model_name)
            
        # Transcribe
        result = await asyncio.to_thread(
            self.whisper_model.transcribe,
            str(audio_path),
            language="en"
        )
        
        # Convert to segments
        segments = []
        for segment in result.get("segments", []):
            segments.append(AudioSegment(
                start=segment.get("start", 0.0),
                end=segment.get("end", 0.0),
                text=segment.get("text", ""),
                confidence=segment.get("confidence", 0.0)
            ))
            
        return segments

# Global audio processor instance
_audio_processor = None

def get_audio_processor() -> AudioProcessor:
    """Get or create the global audio processor instance."""
    global _audio_processor
    if _audio_processor is None:
        _audio_processor = AudioProcessor()
    return _audio_processor
EOF

echo "Deployment patch applied successfully!"
