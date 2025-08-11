"""
VuenCode API package.
FastAPI-based video understanding system optimized for competition performance.
"""

from .main import app
from .schemas import (
    InferenceRequest, InferenceResponse, HealthResponse, ErrorResponse,
    QueryCategory, detect_query_category
)

__all__ = [
    "app",
    "InferenceRequest",
    "InferenceResponse", 
    "HealthResponse",
    "ErrorResponse",
    "QueryCategory",
    "detect_query_category"
]