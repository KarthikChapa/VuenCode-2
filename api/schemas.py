"""
Pydantic schemas for VuenCode API requests and responses.
Optimized for competition performance with comprehensive validation.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, HttpUrl, field_validator
from enum import Enum
import time


class QueryCategory(str, Enum):
    """16 evaluation categories for competition scoring."""
    VIDEO_SUMMARIZATION = "video_summarization"
    TEMPORAL_REASONING = "temporal_reasoning" 
    SPATIAL_REASONING = "spatial_reasoning"
    OBJECT_DETECTION = "object_detection"
    ACTION_RECOGNITION = "action_recognition"
    SCENE_UNDERSTANDING = "scene_understanding"
    DIALOGUE_ANALYSIS = "dialogue_analysis"
    EMOTION_RECOGNITION = "emotion_recognition"
    COUNTING_QUANTIFICATION = "counting_quantification"
    CAUSAL_REASONING = "causal_reasoning"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    MULTI_MODAL_REASONING = "multi_modal_reasoning"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    CONTENT_MODERATION = "content_moderation"
    TECHNICAL_ANALYSIS = "technical_analysis"
    GENERAL_UNDERSTANDING = "general_understanding"


class InferenceRequest(BaseModel):
    """
    Request schema for video inference endpoint.
    Supports both URL and direct video data upload.
    """
    video_url: Optional[HttpUrl] = Field(
        None,
        description="URL to video file (MP4, AVI, MOV, etc.)"
    )
    
    video_data: Optional[str] = Field(
        None, 
        description="Base64 encoded video data (alternative to URL)"
    )
    
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language query about the video content"
    )
    
    category: Optional[QueryCategory] = Field(
        None,
        description="Optional query category for optimized processing"
    )
    
    max_frames: Optional[int] = Field(
        default=None,
        ge=1,
        le=128,
        description="Maximum number of frames to analyze (overrides default)"
    )
    
    quality: Optional[str] = Field(
        default="auto",
        description="Processing quality: 'fast', 'balanced', 'high', 'auto'"
    )
    
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results for identical requests"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-provided request ID for tracking"
    )
    
    @field_validator('video_data')
    @classmethod
    def validate_video_source(cls, v, info):
        """Ensure exactly one video source is provided."""
        if v is not None and info.data.get('video_url') is not None:
            raise ValueError("Provide either video_url or video_data, not both")
        return v
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate and clean query text."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://example.com/video.mp4",
                "query": "What are the main activities shown in this video?",
                "category": "action_recognition",
                "quality": "auto",
                "use_cache": True
            }
        }


class InferenceResponse(BaseModel):
    """
    Response schema for video inference endpoint.
    Includes comprehensive metadata for competition evaluation.
    """
    model_config = {"protected_namespaces": ()}
    
    content: str = Field(
        ...,
        description="Generated response content (plain text only)"
    )
    
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    
    category: Optional[QueryCategory] = Field(
        None,
        description="Detected or provided query category"
    )
    
    model_used: str = Field(
        ...,
        description="Model used for processing (e.g., gemini-2.5-flash)"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Response confidence score (0.0 - 1.0)"
    )
    
    frames_analyzed: int = Field(
        ...,
        ge=0,
        description="Number of video frames analyzed"
    )
    
    cached: bool = Field(
        default=False,
        description="Whether result was served from cache"
    )
    
    fallback_tier: Optional[str] = Field(
        None,
        description="Fallback tier used: 'primary', 'secondary', 'emergency'"
    )
    
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed performance breakdown"
    )
    


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(
        ...,
        description="Service health status: 'healthy', 'degraded', 'unhealthy'"
    )
    
    timestamp: str = Field(
        default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S UTC'),
        description="Health check timestamp"
    )
    
    deployment_mode: str = Field(
        ...,
        description="Current deployment mode: 'local', 'gpu', 'competition'"
    )
    
    version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    performance_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current performance metrics summary"
    )
    
    system_resources: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current system resource utilization"
    )
    
    fallback_status: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fallback system health status"
    )
    
    capabilities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Phase 2 enhanced capabilities status"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-20 15:30:45 UTC",
                "deployment_mode": "gpu",
                "version": "1.0.0",
                "performance_summary": {
                    "avg_latency_ms": 420.5,
                    "p95_latency_ms": 680.2,
                    "throughput_rps": 12.3,
                    "success_rate": 0.995
                },
                "system_resources": {
                    "cpu_percent": 45.2,
                    "memory_percent": 68.1,
                    "gpu_memory_mb": 8192.0,
                    "gpu_utilization": 78.5
                },
                "fallback_status": {
                    "primary_available": True,
                    "secondary_available": True,
                    "emergency_cache_size": 1247
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(
        ...,
        description="Error type or code"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request ID if available"
    )
    
    timestamp: str = Field(
        default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S UTC'),
        description="Error timestamp"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details for debugging"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "processing_failed",
                "message": "Unable to process video: unsupported format",
                "request_id": "req_1642684800124",
                "timestamp": "2024-01-20 15:31:20 UTC",
                "details": {
                    "video_format": "webm",
                    "supported_formats": ["mp4", "avi", "mov", "mkv"]
                }
            }
        }


# Category detection patterns for automatic classification
CATEGORY_PATTERNS = {
    QueryCategory.VIDEO_SUMMARIZATION: [
        "summarize", "summary", "overview", "what happens", "describe video",
        "main points", "key events", "brief description"
    ],
    QueryCategory.TEMPORAL_REASONING: [
        "when", "before", "after", "during", "sequence", "timeline",
        "chronological", "first", "last", "next", "previous"
    ],
    QueryCategory.SPATIAL_REASONING: [
        "where", "location", "position", "left", "right", "above", "below",
        "front", "back", "center", "corner", "edge", "distance"
    ],
    QueryCategory.OBJECT_DETECTION: [
        "what objects", "identify", "detect", "recognize", "see", "visible",
        "items", "things", "present", "appear"
    ],
    QueryCategory.ACTION_RECOGNITION: [
        "action", "activity", "doing", "happening", "movement", "behavior",
        "performing", "executing", "motion", "gesture"
    ],
    QueryCategory.SCENE_UNDERSTANDING: [
        "scene", "setting", "environment", "place", "location", "context",
        "background", "surroundings", "atmosphere"
    ],
    QueryCategory.COUNTING_QUANTIFICATION: [
        "how many", "count", "number", "quantity", "amount", "several",
        "few", "multiple", "total", "sum"
    ],
    QueryCategory.CAUSAL_REASONING: [
        "why", "because", "cause", "reason", "result", "effect", "due to",
        "leads to", "causes", "explains"
    ],
    QueryCategory.COMPARATIVE_ANALYSIS: [
        "compare", "difference", "similar", "different", "contrast", "versus",
        "better", "worse", "more", "less", "same", "unlike"
    ],
    QueryCategory.VISUAL_QUESTION_ANSWERING: [
        "what", "who", "which", "is", "are", "can", "does", "will",
        "question", "answer", "explain"
    ]
}


def detect_query_category(query: str) -> QueryCategory:
    """
    Automatically detect query category based on content patterns.
    
    Args:
        query: Natural language query
        
    Returns:
        Detected QueryCategory or GENERAL_UNDERSTANDING as fallback
    """
    query_lower = query.lower()
    
    # Score each category based on keyword matches
    category_scores = {}
    
    for category, patterns in CATEGORY_PATTERNS.items():
        score = sum(1 for pattern in patterns if pattern in query_lower)
        if score > 0:
            category_scores[category] = score
    
    # Return category with highest score
    if category_scores:
        return max(category_scores.keys(), key=lambda c: category_scores[c])
    
    return QueryCategory.GENERAL_UNDERSTANDING