#!/bin/bash

# Setup Verification Script
# Checks all prerequisites before running the AI RAG stack

set -e

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DevOps AI Assistant - Setup Verification${NC}"
echo -e "${BLUE}========================================${NC}\n"

ERRORS=0
WARNINGS=0

# Check 1: Docker
echo -e "${YELLOW}[1/7] Checking Docker...${NC}"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}✓${NC} Docker installed: $DOCKER_VERSION"
else
    echo -e "${RED}✗${NC} Docker not found"
    ERRORS=$((ERRORS+1))
fi

# Check 2: Docker Compose
echo -e "\n${YELLOW}[2/7] Checking Docker Compose...${NC}"
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    echo -e "${GREEN}✓${NC} Docker Compose installed: $COMPOSE_VERSION"
else
    echo -e "${RED}✗${NC} Docker Compose not found"
    ERRORS=$((ERRORS+1))
fi

# Check 3: NVIDIA Driver
echo -e "\n${YELLOW}[3/7] Checking NVIDIA Driver...${NC}"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
    echo -e "${GREEN}✓${NC} NVIDIA Driver: $DRIVER_VERSION"
    echo -e "${GREEN}✓${NC} GPU: $GPU_NAME"
    echo -e "${GREEN}✓${NC} VRAM: $GPU_MEMORY"
else
    echo -e "${RED}✗${NC} nvidia-smi not found"
    ERRORS=$((ERRORS+1))
fi

# Check 4: Docker GPU Access
echo -e "\n${YELLOW}[4/7] Testing Docker GPU Access...${NC}"
if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker can access GPU"
else
    echo -e "${RED}✗${NC} Docker cannot access GPU"
    echo -e "  ${YELLOW}Fix: sudo apt install nvidia-container-toolkit && sudo systemctl restart docker${NC}"
    ERRORS=$((ERRORS+1))
fi

# Check 5: Disk Space
echo -e "\n${YELLOW}[5/7] Checking Disk Space...${NC}"
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -ge 50 ]; then
    echo -e "${GREEN}✓${NC} Available disk space: ${AVAILABLE_GB}GB"
else
    echo -e "${YELLOW}⚠${NC} Available disk space: ${AVAILABLE_GB}GB (recommended: 50GB+)"
    WARNINGS=$((WARNINGS+1))
fi

# Check 6: RAM
echo -e "\n${YELLOW}[6/7] Checking RAM...${NC}"
TOTAL_RAM_GB=$(free -g | grep Mem | awk '{print $2}')
AVAILABLE_RAM_GB=$(free -g | grep Mem | awk '{print $7}')
echo -e "${GREEN}✓${NC} Total RAM: ${TOTAL_RAM_GB}GB"
if [ "$AVAILABLE_RAM_GB" -ge 8 ]; then
    echo -e "${GREEN}✓${NC} Available RAM: ${AVAILABLE_RAM_GB}GB"
else
    echo -e "${YELLOW}⚠${NC} Available RAM: ${AVAILABLE_RAM_GB}GB (consider closing applications)"
    WARNINGS=$((WARNINGS+1))
fi

# Check 7: Required Ports
echo -e "\n${YELLOW}[7/7] Checking Required Ports...${NC}"
PORTS_OK=true
for PORT in 3000 8000 6333 11434; do
    if sudo lsof -i :$PORT &> /dev/null; then
        echo -e "${YELLOW}⚠${NC} Port $PORT is in use"
        PORTS_OK=false
        WARNINGS=$((WARNINGS+1))
    fi
done
if $PORTS_OK; then
    echo -e "${GREEN}✓${NC} All required ports available (3000, 8000, 6333, 11434)"
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s) - you can proceed but may encounter issues${NC}"
    fi
    echo -e "\n${GREEN}You're ready to start!${NC}"
    echo -e "Run: ${BLUE}make setup && make start${NC}"
else
    echo -e "${RED}✗ $ERRORS error(s) found - please fix before proceeding${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s)${NC}"
    fi
    exit 1
fi

echo -e "\n${BLUE}========================================${NC}"
