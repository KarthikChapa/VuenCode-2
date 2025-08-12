# VuenCode Competition Submission

## 🏆 Competition Ready Video Understanding System

VuenCode is a high-performance video understanding system designed for the competition with sub-500ms latency and comprehensive multimodal analysis capabilities.

## 📋 Competition Requirements Compliance

### ✅ API Endpoint Format
- **Endpoint**: `POST /infer`
- **Content-Type**: `multipart/form-data`
- **Required Fields**:
  - `video`: Uploaded video file (MP4, AVI, MOV, MKV, WEBM, M4V)
  - `prompt`: String prompt or question
- **Response Format**: Plain text output only
- **Headers**: `accept: text/plain`

### ✅ Example cURL Request
```bash
curl -X POST "http://<your-api-url>/infer" \
  -H "accept: text/plain" \
  -F "video=@/path/to/video.mp4" \
  -F "prompt=What action is happening in this clip?"
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r docker/requirements-local.txt

# Set environment variables
export VUENCODE_GEMINI_API_KEY="your_gemini_api_key"
export VUENCODE_API_HOST="127.0.0.1"
export VUENCODE_API_PORT="8000"
```

### 2. Start the Server
```bash
# From the VuenCode directory
python -m api.main
```

### 3. Test the Endpoint
```bash
# Test with curl (bash script)
chmod +x test_curl_competition.sh
./test_curl_competition.sh

# Or test with Python
python test_competition_endpoint.py
```

## 🏗️ System Architecture

### Core Components
- **Enhanced Video Processor**: Multimodal fusion with VST compression
- **Gemini 2.0 Flash**: Fast inference with `gemini-2.0-flash-exp`
- **Pyramid Context System**: Hierarchical video processing for long videos
- **Fallback System**: Intelligent degradation for reliability
- **Performance Tracking**: Real-time latency and throughput monitoring

### Key Features
- **Sub-500ms Latency**: Optimized for competition performance
- **Multimodal Analysis**: Video + Audio + Text fusion
- **Smart Caching**: Intelligent result caching for repeated queries
- **Error Handling**: Graceful degradation and fallback mechanisms
- **Competition Compliance**: Exact format matching requirements

## 📊 Performance Characteristics

### Model Configuration
- **Primary Model**: `gemini-2.0-flash-exp` (fast inference)
- **Fallback Model**: `gemini-2.0-flash-exp` (reliability)
- **Frame Sampling**: 32 frames per video (configurable)
- **Target Latency**: 500ms

### Supported Video Formats
- MP4, AVI, MOV, MKV, WEBM, M4V
- Maximum file size: Configurable
- Frame rate: Adaptive sampling

## 🧪 Testing

### Local Testing
```bash
# Health check
curl -X GET "http://127.0.0.1:8000/health"

# Competition endpoint test
curl -X POST "http://127.0.0.1:8000/infer" \
  -H "accept: text/plain" \
  -F "video=@test_video.mp4" \
  -F "prompt=What action is happening in this clip?"
```

### Automated Tests
```bash
# Run Python test suite
python test_competition_endpoint.py

# Run curl test suite
./test_curl_competition.sh
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
VUENCODE_GEMINI_API_KEY=your_api_key

# Optional (with defaults)
VUENCODE_API_HOST=127.0.0.1
VUENCODE_API_PORT=8000
VUENCODE_TARGET_LATENCY_MS=500
VUENCODE_MAX_FRAMES_PER_VIDEO=32
VUENCODE_GEMINI_FLASH_MODEL=gemini-2.0-flash-exp
VUENCODE_GEMINI_PRO_MODEL=gemini-2.0-flash-exp
```

### Configuration Files
- `configs/local.env`: Local development settings
- `configs/competition.env`: Competition deployment settings

## 🚀 Deployment

### Local Development
```bash
python -m api.main
```

### Production Deployment
```bash
# Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Using Docker (if available)
docker build -t vuencode .
docker run -p 8000:8000 -e VUENCODE_GEMINI_API_KEY=your_key vuencode
```

### Remote GPU Deployment
```bash
# Copy files to remote server
scp -P 44152 -i ~/.ssh/vuencode_team08_id_ed25519 -r . root@38.128.232.8:/opt/vuencode/

# SSH to server and start
ssh -p 44152 -i ~/.ssh/vuencode_team08_id_ed25519 root@38.128.232.8
cd /opt/vuencode
python -m api.main
```

## 📈 Monitoring

### Health Endpoint
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

### Performance Metrics
```bash
# Available in local mode only
curl -X GET "http://127.0.0.1:8000/metrics"
```

### Response Headers
- `X-Request-ID`: Unique request identifier
- `X-Process-Time-MS`: Processing time in milliseconds
- `X-Deployment-Mode`: Current deployment mode

## 🛠️ Troubleshooting

### Common Issues
1. **Port 8000 in use**: Change `VUENCODE_API_PORT` in config
2. **Missing API key**: Set `VUENCODE_GEMINI_API_KEY` environment variable
3. **Video format not supported**: Check allowed extensions in code
4. **Server not starting**: Check Python path and dependencies

### Debug Mode
```bash
# Enable debug logging
export VUENCODE_LOG_LEVEL=DEBUG
python -m api.main
```

## 📝 Competition Submission Checklist

- [x] **API Endpoint**: `POST /infer` implemented
- [x] **Content-Type**: `multipart/form-data` supported
- [x] **Required Fields**: `video` and `prompt` fields
- [x] **Response Format**: Plain text output
- [x] **Headers**: `accept: text/plain` supported
- [x] **Error Handling**: Graceful error responses
- [x] **Performance**: Sub-500ms target latency
- [x] **Testing**: Comprehensive test suite
- [x] **Documentation**: Complete setup instructions

## 🎯 Competition Advantages

### Technical Excellence
- **Fast Inference**: Gemini 2.0 Flash for speed
- **Reliable Processing**: Multi-tier fallback system
- **Smart Caching**: Intelligent result caching
- **Comprehensive Analysis**: Multimodal video understanding

### Competition Ready
- **Exact Format Compliance**: Matches competition requirements precisely
- **Robust Error Handling**: Graceful degradation under load
- **Performance Monitoring**: Real-time metrics and health checks
- **Easy Deployment**: Simple setup and configuration

## 📞 Support

For competition-related issues or questions:
- Check the troubleshooting section above
- Review the test scripts for examples
- Ensure all environment variables are set correctly
- Verify the server is running and accessible

---

**Ready for Competition Submission! 🚀**
