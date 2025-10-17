#!/bin/bash

# Quick test script for Aider setup
# Verifies Ollama connection and model availability

set -e

echo "=========================================="
echo "Testing Aider Setup"
echo "=========================================="
echo ""

# Check if Ollama is running
echo "1. Checking Ollama service..."
if docker ps | grep -q ollama; then
    echo "✅ Ollama container is running"
else
    echo "❌ Ollama container is not running"
    echo "Please start services: make start"
    exit 1
fi
echo ""

# Check if Ollama API is accessible
echo "2. Testing Ollama API..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama API is accessible"
else
    echo "❌ Cannot connect to Ollama API"
    exit 1
fi
echo ""

# Check for Qwen2.5-Coder models
echo "3. Checking for Qwen2.5-Coder models..."
MODELS=$(docker exec ollama ollama list 2>/dev/null || echo "")

if echo "$MODELS" | grep -q "qwen2.5-coder:7b"; then
    echo "✅ qwen2.5-coder:7b is available"
else
    echo "⚠️  qwen2.5-coder:7b not found"
    echo "   Run: make setup-aider"
fi

if echo "$MODELS" | grep -q "qwen2.5-coder:32b"; then
    echo "✅ qwen2.5-coder:32b is available"
else
    echo "⚠️  qwen2.5-coder:32b not found"
    echo "   Run: make setup-aider"
fi
echo ""

# Check if Aider is installed
echo "4. Checking Aider installation..."
if command -v aider &> /dev/null; then
    AIDER_VERSION=$(aider --version 2>&1 | head -n1 || echo "unknown")
    echo "✅ Aider is installed: $AIDER_VERSION"
else
    echo "⚠️  Aider not installed"
    echo "   Run: make setup-aider"
fi
echo ""

# Test Ollama connection from Python
echo "5. Testing Ollama connection..."
python3 << 'PYEOF'
import requests
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        print("✅ Successfully connected to Ollama from Python")
    else:
        print(f"⚠️  Unexpected status code: {response.status_code}")
except Exception as e:
    print(f"❌ Failed to connect: {e}")
PYEOF
echo ""

echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "To start using Aider:"
echo "  make aider       # Fast 7B model"
echo "  make aider-32b   # Powerful 32B model"
echo ""
