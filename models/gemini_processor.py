"""
Intelligent Gemini API processor with model selection and optimization.
Supports both local development (stub) and production (real API) modes.
Implements competition-winning strategies for sub-500ms latency.
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import httpx
import base64
from PIL import Image
import io

from .preprocessing import VideoFrame

# Set up logger
logger = logging.getLogger(__name__)

# Try absolute imports first, then fall back to relative or direct imports
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


class ModelComplexity(Enum):
    """Model complexity levels for intelligent routing."""
    SIMPLE = "simple"      # Use Flash model
    MODERATE = "moderate"  # Use Flash with enhanced prompts
    COMPLEX = "complex"    # Use Pro model


# Fallback QueryCategory enum to avoid circular imports
class QueryCategory(Enum):
    """Query categories for processing optimization."""
    VIDEO_SUMMARIZATION = "video_summarization"
    OBJECT_DETECTION = "object_detection"
    ACTION_RECOGNITION = "action_recognition"
    SCENE_UNDERSTANDING = "scene_understanding"
    TEMPORAL_REASONING = "temporal_reasoning"
    CAUSAL_REASONING = "causal_reasoning"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    SPATIAL_REASONING = "spatial_reasoning"
    MULTI_MODAL_REASONING = "multi_modal_reasoning"
    VISUAL_QUESTION_ANSWERING = "visual_question_answering"
    GENERAL_UNDERSTANDING = "general_understanding"
    
    @classmethod
    def get_or_create(cls, category_input):
        """Get QueryCategory from string or return existing enum."""
        if isinstance(category_input, cls):
            return category_input
        elif isinstance(category_input, str):
            # Try to match string to enum value
            for cat in cls:
                if cat.value == category_input:
                    return cat
            # Default fallback
            return cls.GENERAL_UNDERSTANDING
        else:
            return cls.GENERAL_UNDERSTANDING


@dataclass 
class GeminiResponse:
    """Standardized response from Gemini processing."""
    content: str
    model_used: str
    processing_time_ms: float
    confidence: float
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ComplexityAnalyzer:
    """
    Analyzes query complexity to select optimal Gemini model.
    Balances speed (Flash) vs capability (Pro) for competition optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Complexity indicators
        self.complex_keywords = {
            'high': ['analyze', 'compare', 'reasoning', 'explain why', 'cause', 'effect', 
                    'relationship', 'implication', 'logic', 'philosophy', 'ethics'],
            'moderate': ['describe', 'identify', 'list', 'count', 'find', 'locate', 
                        'when', 'where', 'who', 'what'],
            'simple': ['show', 'see', 'visible', 'appear', 'color', 'basic', 'simple']
        }
        
        self.complex_categories = {
            QueryCategory.CAUSAL_REASONING: ModelComplexity.COMPLEX,
            QueryCategory.COMPARATIVE_ANALYSIS: ModelComplexity.COMPLEX,
            QueryCategory.TEMPORAL_REASONING: ModelComplexity.COMPLEX,
            QueryCategory.MULTI_MODAL_REASONING: ModelComplexity.COMPLEX,
            QueryCategory.SPATIAL_REASONING: ModelComplexity.MODERATE,
            QueryCategory.ACTION_RECOGNITION: ModelComplexity.MODERATE,
            QueryCategory.SCENE_UNDERSTANDING: ModelComplexity.MODERATE,
            QueryCategory.OBJECT_DETECTION: ModelComplexity.SIMPLE,
            QueryCategory.VIDEO_SUMMARIZATION: ModelComplexity.SIMPLE,
            QueryCategory.VISUAL_QUESTION_ANSWERING: ModelComplexity.SIMPLE,
        }
    
    def analyze_complexity(self, query: str, category: QueryCategory, frame_count: int) -> ModelComplexity:
        """
        Analyze query complexity to determine optimal model selection.
        
        Args:
            query: User query text
            category: Query category
            frame_count: Number of frames to process
            
        Returns:
            ModelComplexity level for optimal model selection
        """
        query_lower = query.lower()
        
        # Start with category-based complexity
        base_complexity = self.complex_categories.get(category, ModelComplexity.MODERATE)
        
        # Analyze query text
        complexity_score = 0
        
        # Check for complex keywords
        for keyword in self.complex_keywords['high']:
            if keyword in query_lower:
                complexity_score += 2
        
        for keyword in self.complex_keywords['moderate']:
            if keyword in query_lower:
                complexity_score += 1
        
        for keyword in self.complex_keywords['simple']:
            if keyword in query_lower:
                complexity_score -= 1
        
        # Query length factor
        if len(query.split()) > 15:
            complexity_score += 1
        elif len(query.split()) < 5:
            complexity_score -= 1
        
        # Frame count factor (more frames = potentially more complex analysis)
        if frame_count > 32:
            complexity_score += 1
        elif frame_count < 8:
            complexity_score -= 1
        
        # Determine final complexity
        if complexity_score >= 3:
            final_complexity = ModelComplexity.COMPLEX
        elif complexity_score <= -2:
            final_complexity = ModelComplexity.SIMPLE
        else:
            final_complexity = base_complexity
        
        self.logger.debug(f"Complexity analysis: {query[:50]}... -> {final_complexity.value} (score: {complexity_score})")
        
        return final_complexity


class PromptOptimizer:
    """
    Competition-grade prompt optimizer with timestamp-anchored, structured prompting.
    Implements advanced prompt engineering for baseline-aligned responses.
    """
    
    def __init__(self):
        self.category_templates = {
            QueryCategory.VIDEO_SUMMARIZATION: {
                "flash": """You are a precise video understanding assistant.
Frame timestamps: {timestamps}
Task: Generate a chronological, timestamped summary of key events in the video. Be precise about who does what at each timestamp.
Example: At 00:00:05 – person picks up red ball from table.
Output format: Numbered list with timestamps.
Use deterministic style (temperature=0.2).""",
                
                "pro": """You are an expert video analyst providing detailed chronological analysis.
Frame timestamps: {timestamps}
Task: Create a comprehensive timestamped summary with detailed event descriptions.
For each key moment, provide:
1. Timestamp
2. Actor/subject
3. Specific action
4. Objects involved
5. Scene context
Output as structured timeline."""
            },
            
            QueryCategory.OBJECT_DETECTION: {
                "flash": """You are a precise object detection assistant.
Frame timestamps: {timestamps}
Example: At 00:00:05 – red ball on wooden table.
Task: Provide a numbered list of **distinct objects** with their **first-seen timestamp**.
Output format:
1. 00:00:05 – red ball
2. 00:00:20 – wooden chair
3. 00:00:35 – blue cup
Use deterministic style (temperature=0.2).""",
                
                "pro": """You are an expert object detection analyst.
Frame timestamps: {timestamps}
Task: Comprehensive object inventory with temporal tracking.
For each distinct object, provide:
1. First appearance timestamp
2. Object name and description
3. Location in scene
4. How object changes/moves over time
5. Interactions with other objects
Output as structured object catalog."""
            },
            
            QueryCategory.ACTION_RECOGNITION: {
                "flash": """You are a precise action recognition specialist.
Frame timestamps: {timestamps}
Task: For each action you identify, state the actor, action verb, object of interaction, and timestamp.
Example: 00:00:10 – person walking toward door.
Output format:
1. [timestamp] – [actor] [action verb] [object/target]
2. [timestamp] – [actor] [action verb] [object/target]
Use deterministic style (temperature=0.2).""",
                
                "pro": """You are an expert action analysis specialist.
Frame timestamps: {timestamps}
Task: Detailed action decomposition with temporal reasoning.
For each action sequence:
1. Timestamp range
2. Primary actor
3. Action category (locomotion/manipulation/interaction)
4. Target objects/environment
5. Action intent/purpose
6. Relationship to previous/next actions
Output as structured action timeline."""
            },
            
            QueryCategory.SCENE_UNDERSTANDING: {
                "flash": """You are a precise scene analysis assistant.
Frame timestamps: {timestamps}
Task: Describe the setting (location type, lighting, background elements) and how it changes over time.
Example: 00:00:00 – indoor kitchen, bright lighting, wooden cabinets visible.
Output format:
1. [timestamp] – [location] [lighting] [key elements]
Use deterministic style (temperature=0.2).""",
                
                "pro": """You are an expert environmental analysis specialist.
Frame timestamps: {timestamps}
Task: Comprehensive scene decomposition with temporal changes.
For each distinct scene/setting:
1. Time range
2. Location type and specific details
3. Lighting conditions and sources
4. Background/foreground elements
5. Spatial layout and arrangement
6. Environmental changes over time
Output as structured scene timeline."""
            },
            
            QueryCategory.TEMPORAL_REASONING: {
                "flash": """You are a temporal reasoning specialist.
Frame timestamps: {timestamps}
Task: Step-by-step temporal analysis.
Step 1: Describe what you see in the first frame at {first_timestamp}.
Step 2: Describe changes in the next frame at {second_timestamp}.
Step 3: Summarize the transition between these frames.
Continue this pattern for all frames.""",
                
                "pro": """You are an expert temporal logic analyst.
Frame timestamps: {timestamps}
Task: Comprehensive temporal reasoning with causal connections.
Analyze:
1. Chronological sequence of events
2. Cause-and-effect relationships
3. Temporal dependencies
4. Duration of actions/states
5. Timing patterns and rhythms
6. Temporal anomalies or disruptions
Output as detailed temporal analysis."""
            },
            
            QueryCategory.CAUSAL_REASONING: {
                "flash": """You are a causal reasoning specialist.
Frame timestamps: {timestamps}
Task: Identify clear cause-and-effect chains with timestamps.
Example: 00:00:05 – ball falls (cause) → 00:00:07 – person looks down (effect).
Output format:
1. [timestamp] – [cause event] → [timestamp] – [effect event]
Use deterministic style (temperature=0.2).""",
                
                "pro": """You are an expert causal analysis specialist.
Frame timestamps: {timestamps}
Task: Deep causal reasoning with logical chains.
For each causal relationship:
1. Initial conditions/context
2. Triggering event (timestamp)
3. Causal mechanism/process
4. Intermediate effects
5. Final outcome (timestamp)
6. Confidence in causal link
7. Alternative explanations
Output as structured causal analysis."""
            }
        }
        
        self.default_template = {
            "flash": """You are a precise video understanding assistant.
Frame timestamps: {timestamps}
Task: Answer the question with specific, timestamped observations.
Question: {query}
Use deterministic style (temperature=0.2).""",
            
            "pro": """You are an expert video analyst.
Frame timestamps: {timestamps}
Task: Comprehensive analysis to answer the question with detailed reasoning.
Question: {query}
Provide structured response with timestamps and evidence."""
        }
    
    def optimize_prompt(self, query: str, category: QueryCategory, model_type: str, frames: List[VideoFrame]) -> str:
        """
        Generate timestamp-anchored, optimized prompt for specific model and category.
        
        Args:
            query: Original user query
            category: Query category
            model_type: 'flash' or 'pro'
            frames: List of VideoFrame objects with timestamp data
            
        Returns:
            Optimized prompt string with embedded timestamps
        """
        # Extract timestamps from frames
        timestamps_list = []
        for frame in frames:
            # Convert frame timestamp to HH:MM:SS format
            total_seconds = frame.timestamp
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            timestamps_list.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Create timestamp string for embedding
        if len(timestamps_list) <= 3:
            timestamps_str = ", ".join(timestamps_list)
        else:
            # For many frames, show first, middle, and last
            timestamps_str = f"{timestamps_list[0]}, {timestamps_list[len(timestamps_list)//2]}, {timestamps_list[-1]}"
        
        # Get first and second timestamps for temporal reasoning
        first_timestamp = timestamps_list[0] if timestamps_list else "00:00:00"
        second_timestamp = timestamps_list[1] if len(timestamps_list) > 1 else timestamps_list[0]
        
        # Get category-specific template
        templates = self.category_templates.get(category, self.default_template)
        base_prompt = templates.get(model_type, templates["flash"])
        
        # Format the prompt with timestamp data
        formatted_prompt = base_prompt.format(
            query=query,
            frame_count=len(frames),
            timestamps=timestamps_str,
            first_timestamp=first_timestamp,
            second_timestamp=second_timestamp
        )
        
        # Add user query if not already embedded in template
        if "{query}" not in base_prompt:
            formatted_prompt += f"\n\nUser question: {query}"
        
        # Add performance optimization hints for competition
        if model_type == "flash":
            formatted_prompt += "\n\nProvide a direct, efficient response optimized for speed and deterministic output."
        else:  # pro model
            formatted_prompt += "\n\nProvide comprehensive analysis with detailed reasoning and evidence."
        
        return formatted_prompt


class GeminiProcessor:
    """
    Competition-optimized Gemini API processor.
    Supports intelligent model selection and local development stubs.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.complexity_analyzer = ComplexityAnalyzer()
        self.prompt_optimizer = PromptOptimizer()
        
        # API configuration
        self.api_key = self.config.gemini_api_key
        self.flash_model = self.config.gemini_flash_model
        self.pro_model = self.config.gemini_pro_model
        
        # HTTP client for API calls
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.request_timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        # Performance tracking
        self.model_usage_stats = {"flash": 0, "pro": 0, "stub": 0}
        
        # Determine operational mode - prioritize API key availability
        # If we have an API key, use API mode regardless of deployment_mode setting
        self.is_local_mode = not self.api_key
        
        if self.is_local_mode:
            self.logger.info("GeminiProcessor initialized in LOCAL/STUB mode")
        else:
            self.logger.info(f"GeminiProcessor initialized with API key: ...{self.api_key[-8:]}")
    
    @track_performance("gemini_processing")
    async def analyze_video(
        self,
        frames: List[VideoFrame],
        query: str,
        category = None,
        use_pro_model: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Analyze video frames using optimal Gemini model selection.
        
        Args:
            frames: List of video frames to analyze
            query: User query
            category: Query category for optimization
            use_pro_model: Force Pro model usage (overrides intelligent selection)
            
        Returns:
            GeminiResponse with analysis results
        """
        start_time = time.time()
        
        try:
            # Ensure category is QueryCategory type
            if category is None:
                category = QueryCategory.GENERAL_UNDERSTANDING
            else:
                category = QueryCategory.get_or_create(category)
                
            if self.is_local_mode:
                # Use local stub for development
                response = await self._process_stub(frames, query, category)
            else:
                # Use real Gemini API with intelligent model selection
                response = await self._process_api(frames, query, category, use_pro_model)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update usage statistics
            model_used = response.get("model_used", "unknown")
            if "flash" in model_used.lower():
                self.model_usage_stats["flash"] += 1
            elif "pro" in model_used.lower():
                self.model_usage_stats["pro"] += 1
            else:
                self.model_usage_stats["stub"] += 1
            
            self.logger.debug(f"Video analysis completed: {processing_time:.1f}ms, model: {model_used}")
            
            return {
                "content": response["content"],
                "model_used": model_used,
                "processing_time_ms": processing_time,
                "confidence": response.get("confidence", 0.9),
                "metadata": {
                    "category": category.value,
                    "frame_count": len(frames),
                    "query_length": len(query)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Video analysis failed: {e}")
            raise
    
    async def _process_stub(self, frames: List[VideoFrame], query: str, category: QueryCategory) -> Dict[str, Any]:
        """
        Local development stub with realistic processing simulation.
        """
        # Simulate processing time (competition-optimized: 50-150ms)
        base_delay = 0.05  # 50ms base
        frame_delay = len(frames) * 0.003  # 3ms per frame
        complexity_delay = len(query.split()) * 0.001  # 1ms per word
        
        total_delay = base_delay + frame_delay + complexity_delay
        await asyncio.sleep(min(total_delay, 0.15))  # Cap at 150ms
        
        # Extract timestamps for structured responses
        timestamps_list = []
        for frame in frames:
            total_seconds = frame.timestamp
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            timestamps_list.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Generate structured, timestamp-anchored stub responses
        stub_responses = {
            QueryCategory.VIDEO_SUMMARIZATION: f"""Video Summary with Timeline:
1. {timestamps_list[0] if timestamps_list else '00:00:00'} – Initial scene setup with primary subjects positioned in frame
2. {timestamps_list[len(timestamps_list)//3] if len(timestamps_list) > 2 else '00:00:01'} – Main activity begins with object interaction and movement
3. {timestamps_list[2*len(timestamps_list)//3] if len(timestamps_list) > 3 else '00:00:02'} – Progressive action development with environmental changes
4. {timestamps_list[-1] if timestamps_list else '00:00:03'} – Conclusion of sequence with final positioning""",
            
            QueryCategory.ACTION_RECOGNITION: f"""Action Timeline Analysis:
1. {timestamps_list[0] if timestamps_list else '00:00:00'} – person walking toward central area
2. {timestamps_list[len(timestamps_list)//2] if len(timestamps_list) > 1 else '00:00:01'} – person reaching for object on surface
3. {timestamps_list[-1] if timestamps_list else '00:00:02'} – person manipulating object with hands""",
            
            QueryCategory.OBJECT_DETECTION: f"""Object Inventory with First Appearance:
1. {timestamps_list[0] if timestamps_list else '00:00:00'} – wooden table
2. {timestamps_list[0] if timestamps_list else '00:00:00'} – metal chair
3. {timestamps_list[len(timestamps_list)//2] if len(timestamps_list) > 1 else '00:00:01'} – red container
4. {timestamps_list[-1] if timestamps_list else '00:00:02'} – person wearing blue shirt""",
            
            QueryCategory.SCENE_UNDERSTANDING: f"""Scene Analysis Timeline:
1. {timestamps_list[0] if timestamps_list else '00:00:00'} – indoor kitchen setting, bright fluorescent lighting, wooden cabinets visible
2. {timestamps_list[len(timestamps_list)//2] if len(timestamps_list) > 1 else '00:00:01'} – same kitchen, lighting remains consistent, counter space utilized
3. {timestamps_list[-1] if timestamps_list else '00:00:02'} – kitchen environment, stable lighting conditions, organized workspace""",
            
            QueryCategory.TEMPORAL_REASONING: f"""Temporal Analysis:
Step 1: At {timestamps_list[0] if timestamps_list else '00:00:00'} – subject enters frame from left side
Step 2: At {timestamps_list[len(timestamps_list)//2] if len(timestamps_list) > 1 else '00:00:01'} – subject approaches central object
Step 3: Transition shows deliberate movement pattern indicating planned action sequence""",
            
            QueryCategory.CAUSAL_REASONING: f"""Causal Chain Analysis:
1. {timestamps_list[0] if timestamps_list else '00:00:00'} – person notices object (cause) → {timestamps_list[len(timestamps_list)//3] if len(timestamps_list) > 2 else '00:00:01'} – person moves toward object (effect)
2. {timestamps_list[len(timestamps_list)//2] if len(timestamps_list) > 1 else '00:00:01'} – person reaches for object (cause) → {timestamps_list[-1] if timestamps_list else '00:00:02'} – object position changes (effect)""",
        }
        
        base_response = stub_responses.get(
            category,
            "This video contains visual content that can be analyzed for various elements including objects, actions, and scene composition. The content appears to be well-structured for analysis."
        )
        
        # Customize response based on query
        if "summarize" in query.lower():
            response = f"Video Summary: {base_response}"
        elif "what" in query.lower():
            response = f"In response to your question: {base_response}"
        elif "how many" in query.lower():
            response = f"Based on the video analysis: There appear to be multiple instances of the requested elements. {base_response}"
        else:
            response = base_response
        
        return {
            "content": response,
            "model_used": "gemini-stub-local",
            "confidence": 0.85,  # Good confidence for stub
        }
    
    async def _process_api(
        self,
        frames: List[VideoFrame],
        query: str,
        category,
        force_pro: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process using real Gemini API with intelligent model selection.
        """
        # Determine optimal model
        if force_pro is True:
            model_name = self.pro_model
            model_type = "pro"
            complexity = ModelComplexity.COMPLEX
        elif force_pro is False:
            model_name = self.flash_model
            model_type = "flash"
            complexity = ModelComplexity.SIMPLE
        else:
            # Intelligent model selection
            complexity = self.complexity_analyzer.analyze_complexity(query, category, len(frames))
            
            if complexity == ModelComplexity.COMPLEX:
                model_name = self.pro_model
                model_type = "pro"
            else:
                model_name = self.flash_model
                model_type = "flash"
        
        # Optimize prompt for selected model with timestamp anchoring
        optimized_prompt = self.prompt_optimizer.optimize_prompt(
            query, category, model_type, frames
        )
        
        # Prepare API request
        api_request = await self._prepare_api_request(frames, optimized_prompt, model_name)
        
        # Make API call with retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = await self._call_gemini_api(api_request, model_name)
                
                return {
                    "content": response["content"],
                    "model_used": f"{model_name}-{model_type}",
                    "confidence": response.get("confidence", 0.95),
                    "token_count": response.get("token_count")
                }
                
            except Exception as e:
                self.logger.warning(f"API attempt {attempt + 1} failed: {e}")
                
                if attempt == self.config.max_retries - 1:
                    raise
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    async def _prepare_api_request(self, frames: List[VideoFrame], prompt: str, model: str) -> Dict[str, Any]:
        """Prepare API request payload for Gemini."""
        # Convert frames to base64
        frame_data = []
        for frame in frames[:32]:  # Limit frames for API
            frame_data.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": frame.to_base64()
                }
            })
        
        return {
            "model": model,
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        *frame_data
                    ]
                }
            ],
            "generation_config": {
                "temperature": 0.1,  # Low temperature for consistency
                "max_output_tokens": 1000,
                "top_p": 0.8,
                "top_k": 40
            }
        }
    
    async def _call_gemini_api(self, request_data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Make actual API call to Gemini."""
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        response = await self.http_client.post(
            api_url,
            json=request_data,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Extract content from response
        if "candidates" in result and len(result["candidates"]) > 0:
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            return {
                "content": content,
                "confidence": 0.95,  # High confidence for API responses
                "token_count": result.get("usageMetadata", {}).get("totalTokenCount")
            }
        else:
            raise Exception("No valid response from Gemini API")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get model usage statistics."""
        total_requests = sum(self.model_usage_stats.values())
        
        return {
            "total_requests": total_requests,
            "model_usage": self.model_usage_stats.copy(),
            "usage_percentages": {
                model: (count / max(1, total_requests)) * 100
                for model, count in self.model_usage_stats.items()
            },
            "flash_usage_rate": (self.model_usage_stats["flash"] / max(1, total_requests)) * 100
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.http_client.aclose()


# Global processor instance
_gemini_processor: Optional[GeminiProcessor] = None


def get_gemini_processor() -> GeminiProcessor:
    """Get or create the global Gemini processor instance."""
    global _gemini_processor
    if _gemini_processor is None:
        _gemini_processor = GeminiProcessor()
    return _gemini_processor


def initialize_gemini_processor(config=None) -> GeminiProcessor:
    """Initialize the global Gemini processor with custom config."""
    global _gemini_processor
    _gemini_processor = GeminiProcessor(config)
    return _gemini_processor