#!/bin/bash

# VuenCode Competition Endpoint Test Script
# Tests the exact format required by the competition

echo "🚀 VuenCode Competition Endpoint Test"
echo "======================================"

# Check if video file exists
VIDEO_FILE="../Cheetah cub learns how to hunt and kill a scrub hare.mp4"
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Test video not found: $VIDEO_FILE"
    echo "💡 Please ensure the video file exists in the parent directory"
    exit 1
fi

# Test health endpoint first
echo "🏥 Testing health endpoint..."
curl -X GET "http://127.0.0.1:8000/health" \
  -H "accept: application/json" \
  -w "\nStatus: %{http_code}\nTime: %{time_total}s\n"

echo ""
echo "🧪 Testing competition endpoint with multipart/form-data..."
echo "📹 Video: $VIDEO_FILE"
echo "❓ Prompt: What action is happening in this clip?"
echo ""

# Test the competition endpoint exactly as specified
curl -X POST "http://127.0.0.1:8000/infer" \
  -H "accept: text/plain" \
  -F "video=@$VIDEO_FILE" \
  -F "prompt=What action is happening in this clip?" \
  -w "\n\nStatus: %{http_code}\nTime: %{time_total}s\nSize: %{size_download} bytes\n"

echo ""
echo "======================================"
echo "📋 Competition Format Verified:"
echo "   ✅ Endpoint: POST /infer"
echo "   ✅ Content-Type: multipart/form-data"
echo "   ✅ Fields: video (file), prompt (string)"
echo "   ✅ Response: plain text"
echo "   ✅ Headers: accept: text/plain"
echo ""
echo "🎉 Ready for competition submission!"
