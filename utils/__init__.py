"""
VuenCode utilities package.
Core utilities for configuration, metrics, fallback handling, and system optimization.
"""

from .config import get_config, reload_config, VuenCodeConfig, DeploymentMode, PerformanceProfile
from .metrics import get_performance_tracker, initialize_tracker, track_performance, PerformanceTracker
from .fallback import get_fallback_handler, initialize_fallback_handler, with_fallback, FallbackHandler, FallbackResponse

__all__ = [
    # Configuration
    "get_config",
    "reload_config", 
    "VuenCodeConfig",
    "DeploymentMode",
    "PerformanceProfile",
    
    # Performance tracking
    "get_performance_tracker",
    "initialize_tracker",
    "track_performance",
    "PerformanceTracker",
    
    # Fallback handling
    "get_fallback_handler",
    "initialize_fallback_handler", 
    "with_fallback",
    "FallbackHandler",
    "FallbackResponse"
]