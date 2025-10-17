#!/bin/bash

# Setup Aider - AI Pair Programming Tool
# Works with local Ollama models

set -e

echo "=========================================="
echo "Setting up Aider with Ollama"
echo "=========================================="
echo ""

# Check if Ollama is running
if ! docker ps | grep -q ollama; then
    echo "❌ Ollama container is not running!"
    echo "Please start services first: make start"
    exit 1
fi

echo "✅ Ollama is running"
echo ""

# Pull recommended Qwen2.5-Coder models
echo "📦 Pulling Qwen2.5-Coder models..."
echo ""

echo "1/2 Pulling qwen2.5-coder:7b (fast, ~4.7GB)..."
docker exec ollama ollama pull qwen2.5-coder:7b
echo "✅ qwen2.5-coder:7b ready!"
echo ""

echo "2/2 Pulling qwen2.5-coder:32b (powerful, ~19GB)..."
docker exec ollama ollama pull qwen2.5-coder:32b
echo "✅ qwen2.5-coder:32b ready!"
echo ""

# Install aider if not already installed
if ! command -v aider &> /dev/null; then
    echo "📦 Installing Aider..."
    pip install aider-chat
    echo "✅ Aider installed!"
else
    echo "✅ Aider already installed"
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Available coding models:"
echo "  - qwen2.5-coder:7b   (fast, 4.7GB VRAM)"
echo "  - qwen2.5-coder:32b  (powerful, 19GB VRAM)"
echo ""
echo "Start coding with:"
echo "  make aider          # Use 7B model (faster)"
echo "  make aider-32b      # Use 32B model (more capable)"
echo ""
echo "Or run directly:"
echo "  aider --model ollama/qwen2.5-coder:7b"
echo ""
