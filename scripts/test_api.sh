#!/bin/bash

# API Testing Script
# Quick tests for the DevOps AI Assistant API

API_URL="${1:-http://localhost:8000}"

echo "Testing DevOps AI Assistant API at $API_URL"
echo "=========================================="

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Health Check
echo -e "\n${YELLOW}Test 1: Health Check${NC}"
HEALTH=$(curl -s "$API_URL/api/health")
if echo "$HEALTH" | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "$HEALTH" | python3 -m json.tool
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "$HEALTH"
fi

# Test 2: List Models
echo -e "\n${YELLOW}Test 2: List Models${NC}"
MODELS=$(curl -s "$API_URL/api/models")
if echo "$MODELS" | grep -q "models"; then
    echo -e "${GREEN}✓ Models endpoint working${NC}"
    echo "$MODELS" | python3 -m json.tool | head -20
else
    echo -e "${RED}✗ Models endpoint failed${NC}"
    echo "$MODELS"
fi

# Test 3: Stats
echo -e "\n${YELLOW}Test 3: Vector Database Stats${NC}"
STATS=$(curl -s "$API_URL/api/stats")
if echo "$STATS" | grep -q "vectors_count"; then
    echo -e "${GREEN}✓ Stats endpoint working${NC}"
    echo "$STATS" | python3 -m json.tool
else
    echo -e "${RED}✗ Stats endpoint failed${NC}"
    echo "$STATS"
fi

# Test 4: Chat (simple query)
echo -e "\n${YELLOW}Test 4: Chat Query${NC}"
CHAT_RESPONSE=$(curl -s -X POST "$API_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Kubernetes?",
    "model": "llama3.1:8b",
    "use_rag": true
  }')

if echo "$CHAT_RESPONSE" | grep -q "response"; then
    echo -e "${GREEN}✓ Chat endpoint working${NC}"
    echo "$CHAT_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print('Response:', data['response'][:200] + '...')"
    echo -e "\nContext used: $(echo "$CHAT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['context_used'])")"
else
    echo -e "${RED}✗ Chat endpoint failed${NC}"
    echo "$CHAT_RESPONSE"
fi

echo -e "\n${YELLOW}========================================${NC}"
echo "API tests complete!"
