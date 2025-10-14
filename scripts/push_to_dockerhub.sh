#!/bin/bash

# Script to build, tag, and push Docker images to Docker Hub
# Usage: ./scripts/push_to_dockerhub.sh [version]
# Example: ./scripts/push_to_dockerhub.sh v1.0.0

set -e  # Exit on error

# Configuration
DOCKER_USERNAME="jconover"
BACKEND_IMAGE="ai-rag-backend"
FRONTEND_IMAGE="ai-rag-frontend"

# Get version from argument or use 'latest' as default
VERSION="${1:-latest}"

echo "=================================================="
echo "Building and Pushing AI RAG Stack to Docker Hub"
echo "=================================================="
echo "Docker Hub Username: $DOCKER_USERNAME"
echo "Version: $VERSION"
echo ""

# Check if logged in to Docker Hub
echo "Checking Docker Hub authentication..."
if ! docker info | grep -q "Username: $DOCKER_USERNAME"; then
    echo "Not logged in to Docker Hub. Please login:"
    docker login
fi

echo ""
echo "Step 1/4: Building Backend Image..."
echo "-----------------------------------"
docker build -t ${DOCKER_USERNAME}/${BACKEND_IMAGE}:${VERSION} ./backend
docker tag ${DOCKER_USERNAME}/${BACKEND_IMAGE}:${VERSION} ${DOCKER_USERNAME}/${BACKEND_IMAGE}:latest

echo ""
echo "Step 2/4: Building Frontend Image..."
echo "------------------------------------"
docker build -t ${DOCKER_USERNAME}/${FRONTEND_IMAGE}:${VERSION} ./frontend
docker tag ${DOCKER_USERNAME}/${FRONTEND_IMAGE}:${VERSION} ${DOCKER_USERNAME}/${FRONTEND_IMAGE}:latest

echo ""
echo "Step 3/4: Pushing Backend Image..."
echo "----------------------------------"
docker push ${DOCKER_USERNAME}/${BACKEND_IMAGE}:${VERSION}
docker push ${DOCKER_USERNAME}/${BACKEND_IMAGE}:latest

echo ""
echo "Step 4/4: Pushing Frontend Image..."
echo "-----------------------------------"
docker push ${DOCKER_USERNAME}/${FRONTEND_IMAGE}:${VERSION}
docker push ${DOCKER_USERNAME}/${FRONTEND_IMAGE}:latest

echo ""
echo "=================================================="
echo "✅ Successfully pushed images to Docker Hub!"
echo "=================================================="
echo ""
echo "Backend Images:"
echo "  - ${DOCKER_USERNAME}/${BACKEND_IMAGE}:${VERSION}"
echo "  - ${DOCKER_USERNAME}/${BACKEND_IMAGE}:latest"
echo ""
echo "Frontend Images:"
echo "  - ${DOCKER_USERNAME}/${FRONTEND_IMAGE}:${VERSION}"
echo "  - ${DOCKER_USERNAME}/${FRONTEND_IMAGE}:latest"
echo ""
echo "View on Docker Hub:"
echo "  - https://hub.docker.com/r/${DOCKER_USERNAME}/${BACKEND_IMAGE}"
echo "  - https://hub.docker.com/r/${DOCKER_USERNAME}/${FRONTEND_IMAGE}"
echo ""
