"""
Pyramid of Context System for VuenCode
Handles 2-hour videos with precise temporal queries using hierarchical summarization.
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Import existing components
from .preprocessing import VideoFrame, get_video_preprocessor
from .gemini_processor import get_gemini_processor, QueryCategory
# from .vlm_processor import get_vlm_processor  # Temporarily disabled
from .audio_processor import get_audio_processor

logger = logging.getLogger(__name__)

@dataclass
class TemporalEvent:
    """Represents a specific event with precise timestamp."""
    timestamp: float  # seconds from start
    event_type: str   # scene_change, object_entry, action, dialogue, etc.
    description: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class VideoSegment:
    """Represents a 90-second video segment."""
    start_time: float
    end_time: float
    frames: List[VideoFrame]
    events: List[TemporalEvent]
    micro_summary: str
    audio_transcript: Optional[str] = None

@dataclass
class VideoChapter:
    """Represents a 5-minute chapter (10 segments)."""
    start_time: float
    end_time: float
    segments: List[VideoSegment]
    meso_summary: str
    key_events: List[TemporalEvent]

@dataclass
class VideoContext:
    """Complete video context with hierarchical structure."""
    duration: float
    chapters: List[VideoChapter]
    macro_summary: str
    temporal_index: Dict[float, List[TemporalEvent]]
    semantic_index: Dict[str, List[Tuple[float, float]]]  # concept -> (start, end) times

class PyramidContextProcessor:
    """
    Implements the Pyramid of Context system for 2-hour video processing.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.video_preprocessor = get_video_preprocessor()
        self.gemini_processor = get_gemini_processor()
        # self.vlm_processor = get_vlm_processor("llava-hf/llava-1.5-7b-hf")  # Temporarily disabled
        self.audio_processor = get_audio_processor()
        
        # Configuration
        self.segment_duration = 90.0  # 90 seconds
        self.chapter_duration = 300.0  # 5 minutes
        self.overlap_duration = 5.0    # 5 seconds overlap
        
        # Initialize VLM (temporarily disabled)
        # asyncio.create_task(self.vlm_processor.initialize())
        
        self.logger.info("PyramidContextProcessor initialized")
    
    async def process_video_pyramid(self, video_data: bytes, query: str) -> Dict[str, Any]:
        """
        Process video using Pyramid of Context approach.
        """
        start_time = time.time()
        
        try:
            # Step 1: Segmentation
            self.logger.info("Step 1: Segmenting video...")
            segments = await self._segment_video(video_data)
            
            # Step 2: Event Tagging
            self.logger.info("Step 2: Tagging events...")
            segments = await self._tag_events(segments)
            
            # Step 3: Hierarchical Summarization
            self.logger.info("Step 3: Creating hierarchical summaries...")
            context = await self._create_hierarchical_summaries(segments)
            
            # Step 4: Query-Specific Retrieval
            self.logger.info("Step 4: Retrieving relevant context...")
            relevant_context = await self._retrieve_relevant_context(context, query)
            
            # Step 5: Focused Response Generation
            self.logger.info("Step 5: Generating focused response...")
            response = await self._generate_focused_response(relevant_context, query)
            
            processing_time = time.time() - start_time
            
            return {
                "content": response,
                "processing_time_ms": processing_time * 1000,
                "context_used": {
                    "segments_retrieved": len(relevant_context.get("segments", [])),
                    "temporal_range": relevant_context.get("temporal_range"),
                    "confidence": relevant_context.get("confidence", 0.8)
                },
                "pyramid_stats": {
                    "total_segments": len(segments),
                    "total_events": sum(len(s.events) for s in segments),
                    "chapters": len(context.chapters)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pyramid processing failed: {e}")
            return {
                "content": f"Error processing video: {str(e)}",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }
    
    async def _segment_video(self, video_data: bytes) -> List[VideoSegment]:
        """Step 1: Break video into 90-second segments."""
        segments = []
        
        # Extract frames with temporal information
        frames = await self.video_preprocessor.extract_frames(
            video_data, 
            max_frames=1000,  # More frames for 2-hour video
            quality="balanced"
        )
        
        # Group frames into segments
        current_segment_start = 0.0
        current_frames = []
        
        for frame in frames:
            if frame.timestamp >= current_segment_start + self.segment_duration:
                # Create segment
                if current_frames:
                    segment = VideoSegment(
                        start_time=current_segment_start,
                        end_time=current_segment_start + self.segment_duration,
                        frames=current_frames,
                        events=[],
                        micro_summary=""
                    )
                    segments.append(segment)
                
                # Start new segment
                current_segment_start = frame.timestamp
                current_frames = [frame]
            else:
                current_frames.append(frame)
        
        # Add final segment
        if current_frames:
            segment = VideoSegment(
                start_time=current_segment_start,
                end_time=current_segment_start + self.segment_duration,
                frames=current_frames,
                events=[],
                micro_summary=""
            )
            segments.append(segment)
        
        self.logger.info(f"Created {len(segments)} segments")
        return segments
    
    async def _tag_events(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Step 2: Tag specific events in each segment."""
        for segment in segments:
            events = []
            
            # Use VLM for enhanced visual understanding (temporarily disabled)
            # if self.vlm_processor.initialized:
            #     vlm_result = await self.vlm_processor.analyze_frames(
            #         [frame.image for frame in segment.frames[:5]],  # Sample frames
            #         "Describe what is happening in this video segment with specific events and timestamps"
            #     )
            #     
            #     if vlm_result.get("vlm_analysis"):
            #         for analysis in vlm_result["vlm_analysis"]:
            #             event = TemporalEvent(
            #                 timestamp=segment.start_time + (analysis["frame_index"] * 6),  # Approximate
            #                 event_type="vlm_detected",
            #                 description=analysis["response"],
            #                 confidence=analysis["confidence"],
            #                 metadata={"source": "vlm"}
            #             )
            #             events.append(event)
            
            # Add scene change detection
            if len(segment.frames) > 1:
                # Simple scene change detection
                prev_frame = segment.frames[0].image
                for i, frame in enumerate(segment.frames[1:], 1):
                    # Calculate frame difference
                    diff = np.mean(np.abs(np.array(frame.image) - np.array(prev_frame)))
                    if diff > 50:  # Threshold for scene change
                        event = TemporalEvent(
                            timestamp=segment.start_time + (i * self.segment_duration / len(segment.frames)),
                            event_type="scene_change",
                            description="Scene change detected",
                            confidence=0.8,
                            metadata={"frame_diff": diff}
                        )
                        events.append(event)
                    prev_frame = frame.image
            
            segment.events = events
        
        return segments
    
    async def _create_hierarchical_summaries(self, segments: List[VideoSegment]) -> VideoContext:
        """Step 3: Create hierarchical summaries (micro/meso/macro)."""
        
        # Create micro-summaries for each segment
        for segment in segments:
            # Generate descriptions for frames if they don't have them
            frame_descriptions = []
            for frame in segment.frames:
                if not hasattr(frame, 'description') or not frame.description:
                    # Generate description for this frame
                    try:
                        # Use analyze_video method to get frame description
                        result = await self.gemini_processor.analyze_video(
                            [frame],
                            "Describe what you see in this image in detail",
                            QueryCategory.SCENE_UNDERSTANDING
                        )
                        frame.description = result.get('content', 'Frame description unavailable')
                    except Exception as e:
                        self.logger.warning(f"Failed to generate frame description: {e}")
                        frame.description = "Frame description unavailable"
                
                frame_descriptions.append(frame.description)
            
            # Combine frame descriptions and events
            event_descriptions = [event.description for event in segment.events]
            
            all_content = frame_descriptions + event_descriptions
            if all_content:
                try:
                    micro_summary = await self.gemini_processor.generate_summary(
                        all_content,
                        f"Create a concise summary of this {self.segment_duration}-second video segment"
                    )
                    segment.micro_summary = micro_summary
                except Exception as e:
                    self.logger.warning(f"Failed to generate micro-summary: {e}")
                    # Fallback: use first frame description or event description
                    if frame_descriptions:
                        segment.micro_summary = frame_descriptions[0]
                    elif event_descriptions:
                        segment.micro_summary = event_descriptions[0]
                    else:
                        segment.micro_summary = f"Video segment from {segment.start_time:.1f}s to {segment.end_time:.1f}s"
            else:
                # No content available, create basic summary
                segment.micro_summary = f"Video segment from {segment.start_time:.1f}s to {segment.end_time:.1f}s"
        
        # Group segments into chapters (5 minutes each)
        chapters = []
        current_chapter_start = 0.0
        current_segments = []
        
        for segment in segments:
            if segment.start_time >= current_chapter_start + self.chapter_duration:
                # Create chapter
                if current_segments:
                    chapter = await self._create_chapter(current_segments, current_chapter_start)
                    chapters.append(chapter)
                
                # Start new chapter
                current_chapter_start = segment.start_time
                current_segments = [segment]
            else:
                current_segments.append(segment)
        
        # Add final chapter
        if current_segments:
            chapter = await self._create_chapter(current_segments, current_chapter_start)
            chapters.append(chapter)
        
        # Create macro-summary
        chapter_summaries = [chapter.meso_summary for chapter in chapters if chapter.meso_summary]
        
        if chapter_summaries:
            macro_summary = await self.gemini_processor.generate_summary(
                chapter_summaries,
                "Create a high-level narrative summary of this entire video"
            )
        else:
            # Fallback: create macro summary from all available content
            all_content = []
            for chapter in chapters:
                for segment in chapter.segments:
                    # Collect frame descriptions
                    for frame in segment.frames:
                        if hasattr(frame, 'description') and frame.description:
                            all_content.append(frame.description)
                    
                    # Collect event descriptions
                    for event in segment.events:
                        if event.description:
                            all_content.append(event.description)
            
            if all_content:
                macro_summary = await self.gemini_processor.generate_summary(
                    all_content,
                    "Create a high-level narrative summary of this entire video based on the available content"
                )
            else:
                # Last resort
                macro_summary = f"Video with {len(chapters)} chapters covering {segments[-1].end_time if segments else 0:.1f} seconds."
        
        # Create temporal and semantic indexes
        temporal_index = self._create_temporal_index(segments)
        semantic_index = await self._create_semantic_index(segments)
        
        return VideoContext(
            duration=segments[-1].end_time if segments else 0,
            chapters=chapters,
            macro_summary=macro_summary,
            temporal_index=temporal_index,
            semantic_index=semantic_index
        )
    
    async def _create_chapter(self, segments: List[VideoSegment], start_time: float) -> VideoChapter:
        """Create a 5-minute chapter from segments."""
        end_time = start_time + self.chapter_duration
        
        # Collect all events
        all_events = []
        for segment in segments:
            all_events.extend(segment.events)
        
        # Create meso-summary
        segment_summaries = [s.micro_summary for s in segments if s.micro_summary]
        
        # If no micro-summaries available, create fallback content from frame descriptions and events
        if not segment_summaries:
            fallback_content = []
            for segment in segments:
                # Use frame descriptions if available
                frame_descriptions = []
                for frame in segment.frames:
                    if hasattr(frame, 'description') and frame.description:
                        frame_descriptions.append(frame.description)
                
                # Use event descriptions if available
                event_descriptions = [event.description for event in segment.events if event.description]
                
                # Combine all available content
                segment_content = frame_descriptions + event_descriptions
                if segment_content:
                    fallback_content.extend(segment_content)
            
            if fallback_content:
                meso_summary = await self.gemini_processor.generate_summary(
                    fallback_content,
                    f"Create a summary of this {self.chapter_duration}-second video chapter based on the available content"
                )
            else:
                # Last resort: create a basic summary from timing information
                meso_summary = f"Video chapter from {start_time:.1f}s to {end_time:.1f}s containing {len(segments)} segments with {len(all_events)} events."
        else:
            meso_summary = await self.gemini_processor.generate_summary(
                segment_summaries,
                f"Create a summary of this {self.chapter_duration}-second video chapter"
            )
        
        return VideoChapter(
            start_time=start_time,
            end_time=end_time,
            segments=segments,
            meso_summary=meso_summary,
            key_events=all_events
        )
    
    def _create_temporal_index(self, segments: List[VideoSegment]) -> Dict[float, List[TemporalEvent]]:
        """Create temporal index for quick time-based lookups."""
        temporal_index = {}
        
        for segment in segments:
            for event in segment.events:
                # Round timestamp to nearest second for indexing
                rounded_time = round(event.timestamp)
                if rounded_time not in temporal_index:
                    temporal_index[rounded_time] = []
                temporal_index[rounded_time].append(event)
        
        return temporal_index
    
    async def _create_semantic_index(self, segments: List[VideoSegment]) -> Dict[str, List[Tuple[float, float]]]:
        """Create semantic index for concept-based search."""
        semantic_index = {}
        
        # Extract concepts from summaries and events
        for segment in segments:
            text_content = segment.micro_summary + " " + " ".join([e.description for e in segment.events])
            
            # Extract key concepts (simplified - could use NER or keyword extraction)
            concepts = await self.gemini_processor.extract_concepts(text_content)
            
            for concept in concepts:
                if concept not in semantic_index:
                    semantic_index[concept] = []
                semantic_index[concept].append((segment.start_time, segment.end_time))
        
        return semantic_index
    
    async def _retrieve_relevant_context(self, context: VideoContext, query: str) -> Dict[str, Any]:
        """Step 4: Smart retrieval based on query analysis."""
        
        # Analyze query type
        query_analysis = await self._analyze_query(query)
        
        relevant_segments = []
        temporal_range = None
        confidence = 0.8
        
        if query_analysis["type"] == "temporal":
            # Time-based query
            target_time = query_analysis["target_time"]
            temporal_range = (max(0, target_time - 30), min(context.duration, target_time + 30))
            
            # Find segments in temporal range
            for chapter in context.chapters:
                for segment in chapter.segments:
                    if (segment.start_time <= temporal_range[1] and 
                        segment.end_time >= temporal_range[0]):
                        relevant_segments.append(segment)
        
        elif query_analysis["type"] == "semantic":
            # Concept-based query
            concepts = query_analysis["concepts"]
            
            for concept in concepts:
                if concept in context.semantic_index:
                    for start_time, end_time in context.semantic_index[concept]:
                        # Find segments in this range
                        for chapter in context.chapters:
                            for segment in chapter.segments:
                                if (segment.start_time <= end_time and 
                                    segment.end_time >= start_time):
                                    relevant_segments.append(segment)
        
        else:
            # General query - use macro summary and key segments
            relevant_segments = [chapter.segments[0] for chapter in context.chapters[:3]]
        
        return {
            "segments": relevant_segments,
            "temporal_range": temporal_range,
            "query_analysis": query_analysis,
            "confidence": confidence,
            "macro_summary": context.macro_summary
        }
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine type and extract parameters."""
        
        # Check for temporal patterns
        import re
        time_patterns = [
            r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
            r'(\d{1,2}):(\d{2})',          # MM:SS
            r'(\d+)\s*(?:minutes?|mins?)\s*(\d+)\s*(?:seconds?|secs?)',  # X minutes Y seconds
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query)
            if match:
                if len(match.groups()) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(int, match.groups())
                    target_time = hours * 3600 + minutes * 60 + seconds
                elif len(match.groups()) == 2:  # MM:SS
                    minutes, seconds = map(int, match.groups())
                    target_time = minutes * 60 + seconds
                else:
                    minutes, seconds = map(int, match.groups())
                    target_time = minutes * 60 + seconds
                
                return {
                    "type": "temporal",
                    "target_time": target_time,
                    "original_match": match.group()
                }
        
        # Extract concepts for semantic search
        concepts = await self.gemini_processor.extract_concepts(query)
        
        return {
            "type": "semantic",
            "concepts": concepts,
            "original_query": query
        }
    
    async def _generate_focused_response(self, relevant_context: Dict[str, Any], query: str) -> str:
        """Step 5: Generate focused response using retrieved context."""
        
        # Prepare context for response generation
        context_parts = []
        
        # Add macro summary for overall context
        if relevant_context.get("macro_summary"):
            context_parts.append(f"Overall video context: {relevant_context['macro_summary']}")
        
        # Add relevant segment details
        for segment in relevant_context.get("segments", []):
            segment_info = f"Time {segment.start_time:.1f}s - {segment.end_time:.1f}s: {segment.micro_summary}"
            if segment.events:
                events_info = "; ".join([f"{e.event_type}: {e.description}" for e in segment.events])
                segment_info += f" Events: {events_info}"
            context_parts.append(segment_info)
        
        # Add temporal context if available
        if relevant_context.get("temporal_range"):
            start, end = relevant_context["temporal_range"]
            context_parts.append(f"Focusing on time range: {start:.1f}s - {end:.1f}s")
        
        # Generate focused response
        full_context = "\n".join(context_parts)
        
        response = await self.gemini_processor.generate_response(
            query,
            full_context,
            "Provide a precise, detailed answer based on the specific video context provided. Include relevant timestamps and details."
        )
        
        return response

def get_pyramid_context_processor(config=None) -> PyramidContextProcessor:
    """Get PyramidContextProcessor instance."""
    return PyramidContextProcessor(config)

def initialize_pyramid_context_processor(config=None) -> PyramidContextProcessor:
    """Initialize and return PyramidContextProcessor."""
    return PyramidContextProcessor(config)
