"""
Advanced configuration management for VuenCode competition system.
Supports both local development and GPU-accelerated production deployment.
"""

import os
from enum import Enum
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class DeploymentMode(str, Enum):
    """Deployment mode enumeration for different environments."""
    LOCAL = "local"
    GPU = "gpu"
    COMPETITION = "competition"


class PerformanceProfile(str, Enum):
    """Performance optimization profiles."""
    DEVELOPMENT = "development"  # Fast iteration, minimal optimization
    TESTING = "testing"          # Balanced for testing
    PRODUCTION = "production"    # Maximum optimization
    COMPETITION = "competition"  # Competition-winning configuration


class VuenCodeConfig(BaseSettings):
    """
    Centralized configuration management with environment-specific optimization.
    
    Automatically detects deployment mode and applies optimal settings for:
    - Local development (CPU-based, fast iteration)
    - GPU production (maximum performance)
    - Competition deployment (sub-500ms target)
    """
    
    # === Core Deployment Settings ===
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.LOCAL,
        description="Deployment mode: local, gpu, or competition"
    )
    
    performance_profile: PerformanceProfile = Field(
        default=PerformanceProfile.DEVELOPMENT,
        description="Performance optimization profile"
    )
    
    # === API Configuration ===
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # === Model Configuration ===
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    gemini_flash_model: str = Field(default="gemini-2.5-flash", description="Fast Gemini model")
    gemini_pro_model: str = Field(default="gemini-2.5-pro", description="Complex Gemini model")
    
    # Model selection thresholds
    complexity_threshold: float = Field(default=0.6, description="Threshold for Flash vs Pro selection")
    max_retries: int = Field(default=3, description="Maximum API retry attempts")
    request_timeout: int = Field(default=30, description="API request timeout (seconds)")
    
    # === Performance Targets ===
    target_latency_ms: int = Field(default=500, description="Target end-to-end latency (ms)")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent processing")
    
    # === Video Processing Configuration ===
    # Frame extraction settings
    max_frames_per_video: int = Field(default=32, description="Maximum frames to extract")
    default_fps_sample: float = Field(default=1.0, description="Default sampling FPS")
    scene_detection_threshold: float = Field(default=0.3, description="Scene change detection threshold")
    
    # GPU acceleration settings
    use_gpu_acceleration: bool = Field(default=False, description="Enable GPU acceleration")
    gpu_device_id: int = Field(default=0, description="GPU device ID")
    tensorrt_optimization: bool = Field(default=False, description="Enable TensorRT optimization")
    
    # === Caching Configuration ===
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for distributed caching")
    
    # === Deployment & Monitoring ===
    ngrok_token: Optional[str] = Field(default=None, description="Ngrok authentication token")
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # === Fallback Configuration ===
    enable_fallback: bool = Field(default=True, description="Enable three-tier fallback")
    fallback_response_cache: bool = Field(default=True, description="Cache fallback responses")
    emergency_mode_threshold: float = Field(default=0.95, description="Error rate threshold for emergency mode")
    
    # === Phase 2 Enhanced Features ===
    enable_vst_compression: bool = Field(default=True, description="Enable Visual Summarization Tokens")
    enable_multimodal_fusion: bool = Field(default=True, description="Enable multimodal video+audio+text fusion")
    enable_audio_processing: bool = Field(default=True, description="Enable audio processing with Whisper")
    whisper_model_size: str = Field(default="base", description="Whisper model size (tiny/base/small/medium/large)")
    vst_max_tokens_per_minute: int = Field(default=15, description="Maximum VST tokens per minute of video")
    multimodal_embedding_dim: int = Field(default=768, description="Multimodal embedding dimension")
    
    class Config:
        env_file = ".env"
        env_prefix = "VUENCODE_"
        case_sensitive = False
    
    @property
    def is_local_mode(self) -> bool:
        """Check if running in local development mode."""
        return self.deployment_mode == DeploymentMode.LOCAL
    
    @property
    def is_gpu_mode(self) -> bool:
        """Check if running in GPU-accelerated mode."""
        return self.deployment_mode in [DeploymentMode.GPU, DeploymentMode.COMPETITION]
    
    @property
    def is_competition_mode(self) -> bool:
        """Check if running in competition optimization mode."""
        return self.deployment_mode == DeploymentMode.COMPETITION
    
    def get_optimized_settings(self) -> Dict[str, Any]:
        """
        Get optimized settings based on deployment mode and performance profile.
        
        Returns:
            Dict containing optimized configuration parameters
        """
        settings = {}
        
        if self.is_local_mode:
            # Local development optimizations
            settings.update({
                "max_frames_per_video": 8,  # Reduced for fast iteration
                "default_fps_sample": 0.5,  # Lower sampling rate
                "use_gpu_acceleration": False,
                "tensorrt_optimization": False,
                "target_latency_ms": 200,  # Relaxed target for development
            })
        
        elif self.is_competition_mode:
            # Competition-winning optimizations
            settings.update({
                "max_frames_per_video": 64,  # Higher quality
                "default_fps_sample": 2.0,   # Optimal sampling rate
                "use_gpu_acceleration": True,
                "tensorrt_optimization": True,
                "target_latency_ms": 400,    # Aggressive target (20% buffer)
                "max_concurrent_requests": 20,  # Higher throughput
                "complexity_threshold": 0.7,    # More Flash usage
            })
        
        elif self.is_gpu_mode:
            # GPU production optimizations
            settings.update({
                "max_frames_per_video": 48,
                "default_fps_sample": 1.5,
                "use_gpu_acceleration": True,
                "tensorrt_optimization": True,
                "target_latency_ms": 600,    # Conservative target
                "max_concurrent_requests": 15,
            })
        
        return settings
    
    def apply_optimizations(self) -> None:
        """Apply optimized settings based on current mode."""
        optimized = self.get_optimized_settings()
        
        for key, value in optimized.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def load_for_environment(cls, env_file: Optional[str] = None) -> "VuenCodeConfig":
        """
        Load configuration optimized for the current environment.
        
        Args:
            env_file: Optional path to environment file
            
        Returns:
            Optimized VuenCodeConfig instance
        """
        # If no env_file specified, try to auto-detect the appropriate one
        if env_file is None:
            # Default to local.env for development
            local_env = Path("VuenCode/configs/local.env")
            if local_env.exists():
                env_file = str(local_env)
        
        if env_file and Path(env_file).exists():
            config = cls(_env_file=env_file)
        else:
            config = cls()
        
        # Auto-detect deployment mode if not explicitly set
        if os.getenv("CUDA_VISIBLE_DEVICES") is not None or os.getenv("GPU_MODE"):
            config.deployment_mode = DeploymentMode.GPU
        
        if os.getenv("COMPETITION_MODE"):
            config.deployment_mode = DeploymentMode.COMPETITION
            config.performance_profile = PerformanceProfile.COMPETITION
        
        # Apply optimizations
        config.apply_optimizations()
        
        return config


# Global configuration instance
config = VuenCodeConfig.load_for_environment()


def get_config() -> VuenCodeConfig:
    """Get the global configuration instance."""
    return config


def reload_config(env_file: Optional[str] = None) -> VuenCodeConfig:
    """Reload configuration from environment."""
    global config
    config = VuenCodeConfig.load_for_environment(env_file)
    return config


# Environment-specific configuration files
def get_config_path(mode: DeploymentMode) -> str:
    """Get the configuration file path for a specific deployment mode."""
    config_files = {
        DeploymentMode.LOCAL: "configs/local.env",
        DeploymentMode.GPU: "configs/gpu.env", 
        DeploymentMode.COMPETITION: "configs/competition.env"
    }
    return config_files.get(mode, "configs/local.env")