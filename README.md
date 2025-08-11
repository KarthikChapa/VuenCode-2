# VuenCode - Competition-Winning Video Understanding System

**High-Performance Video Analysis & Conversational AI**

A modular, scalable video understanding system designed for sub-500ms latency with maximum quality. Built for both local development and GPU-accelerated production deployment.

## Features
• Video Event Recognition & Summarization with Guideline Adherence:
    ◦ Accepts a video stream as input.
    ◦ Identifies specific events within the video.
    ◦ Summarizes video content, highlighting key events and any detected guideline adherence or violations.
    ◦ Example Scenario: Given a traffic scene video, the assistant can identify vehicle movements, pedestrian crossings, and traffic light changes, then summarize traffic violations (e.g., "Vehicle X ran a red light at timestamp Y," "Pedestrian crossed against the signal at timestamp Z").
• Multi-Turn Conversations:
    ◦ Supports natural, multi-turn interactions, retaining context from previous turns to understand follow-up questions and provide coherent responses.
    ◦ An agentic workflow is preferred for this functionality.
• Video Input Processing:
    ◦ Round 1: Processes input video streams with a maximum duration of 2 minutes.
    ◦ Round 2 (Advanced Feature): Processes input video streams with a maximum duration of 120 minutes at a frame rate of 90 frames per second (fps).
## Architectural Design
Our system is designed with a modular pipeline to ensure strong, balanced performance across different areas and reasoning types.
Key components include:
• API Inference Layer: A FastAPI endpoint routes requests to appropriate backend modules.
• GPU-accelerated Frame Extractor: For efficient video processing.
• Specialized Modules (optimized for specific reasoning types):
    ◦ Abstraction: For high-level reasoning.
    ◦ Physics Reasoning Engine: Uses explicit motion vectors (e.g., from a SlowFast backbone) and constraint-checkers for physical conservation (e.g., mass/quantity). Optimizations include simplified rigid-body motion models and precompiled physics rules.
    ◦ Semantics & Colour: Handles scene semantics, color recognition, and object attributes through semantic segmentation and attribute classifiers.
    ◦ Object & State: For identifying objects and their states.
    ◦ Action & Motion: For recognizing actions and temporal movements.
    ◦ Counterfactual & Predictive: For reasoning about hypothetical situations or future events, potentially leveraging "Flash vs. full LLM routing".
    ◦ Adversarial, Change Detection: For detecting subtle or deliberate changes.
• Multi-turn Memory and Summarization Components.
• Reasoning Engines integrated with prompt templates.
 Key Technologies & Justification
We have the freedom to choose any backend technology stack and VLM/AI models, with a strong emphasis on solutions optimized for GPU infrastructure, real-time processing, and scalability.
• Backend Tech Stack:
    ◦ FastAPI: Chosen for its efficiency in handling API requests, crucial for routing video understanding tasks to specialized modules. Its suitability for AI/ML workloads and scalability under load makes it ideal for high-performance requirements.
    ◦ Redis: A powerful, fast, and feature-rich cache, data structure server, and vector query engine. Redis is considered for AI applications and can be utilized for caching intermediate video processing results or managing conversational context for multi-turn interactions, enhancing real-time responsiveness.
    ◦ Prometheus: An open-source monitoring system and time series database. It can be integrated for collecting and querying metrics to monitor application performance, aiding in identifying and optimizing bottlenecks.
• VLM/AI Models:
    ◦ Open-source models for fine-tuning or custom optimization are highly encouraged.
    ◦ Tarsier2 (ByteDance Research): A state-of-the-art Large Vision-Language Model (LVLM) excelling in detailed video descriptions and general video understanding. Its three-stage training (pre-training, supervised fine-tuning with fine-grained temporal alignment, and Direct Preference Optimization) helps in achieving high accuracy and reducing hallucinations. It


## Project Structure

```
VuenCode/
├── api/                    # FastAPI application
├── models/                 # Video processing & AI models
├── utils/                  # Configuration & utilities
├── tests/                  # Comprehensive test suite
├── docker/                 # Container configurations
├── deploy/                 # Deployment automation
└── configs/                # Environment configurations
```

## Quick Start

### Phase 1: Local Development
```bash
# Clone and setup
cd VuenCode-2
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows (PowerShell):
# .\venv\Scripts\Activate.ps1
pip install -r docker/requirements-local.txt

# Run locally
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/health
pytest tests/
```

### Phase 2: GPU Deployment
```bash
# Build and deploy
docker-compose -f docker/docker-compose.yml up --build
python deploy/deploy.py

# Benchmark
python deploy/benchmark.py --url <ngrok-url>
```

## API Endpoints

### `/health` - Health Check
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "mode": "local", "latency_avg_ms": 50}
```

### `/infer` - Video Analysis
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "query": "Summarize this video"}' \
  http://localhost:8000/infer
```

## Competition Optimization

- **Latency Target**: <500ms (50% better than requirement)
- **Throughput**: 10+ concurrent requests
- **Quality**: SOTA on all 16 evaluation categories
- **Reliability**: 99.9% uptime with fallbacks

## Performance Metrics

Real-time tracking of:
- End-to-end latency (p50, p95, p99)
- Processing throughput
- Model accuracy by category
- System resource utilization

---

**Built for Vuencode Competition**