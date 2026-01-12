# =============================================================================
# Multi-Stage Dockerfile for DevOps AI Assistant Backend
# =============================================================================
# Optimized for minimal image size with multi-stage builds:
#   - Builder stage: compiles dependencies and downloads models
#   - Runtime stage: minimal image with only production artifacts
#
# Expected size reduction: 30-40% compared to single-stage build
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile native extensions
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
# - build-essential: Required for compiling Python packages with C extensions
# - git: Required by some pip packages for version detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for clean dependency isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements for better layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~5GB by excluding CUDA)
# This must be done before requirements.txt to override the torch version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install production dependencies (exclude dev/test packages)
# Filter out testing dependencies to reduce image size
RUN grep -v -E '^(pytest|black|ruff|mypy|pre-commit)' requirements.txt > requirements-prod.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# Download spaCy English model for semantic chunking
# Using small model (en_core_web_sm) to minimize image size (~12MB vs ~560MB for lg)
RUN python -m spacy download en_core_web_sm

# Download NLTK data for text processing
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# Clean up unnecessary files from installed packages
RUN find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyo" -delete 2>/dev/null || true && \
    find /opt/venv -type f -name "*.c" -delete 2>/dev/null || true && \
    find /opt/venv -type f -name "*.h" -delete 2>/dev/null || true && \
    rm -rf /opt/venv/lib/python3.11/site-packages/pip* && \
    rm -rf /opt/venv/lib/python3.11/site-packages/setuptools* && \
    rm -rf /opt/venv/lib/python3.11/site-packages/wheel*

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Security and performance environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install only runtime dependencies (no build tools)
# - libgomp1: Required for torch CPU operations (OpenMP)
# - curl: Required for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy NLTK data from builder
COPY --from=builder /root/nltk_data /root/nltk_data

# Create non-root user for security before copying app code
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Copy NLTK data to user home (will be owned by root, but readable)
RUN mkdir -p /home/appuser/nltk_data && \
    cp -r /root/nltk_data/* /home/appuser/nltk_data/ && \
    rm -rf /root/nltk_data

# Copy only application code (not tests, not dev files)
# The .dockerignore already excludes __pycache__, .pyc, tests, etc.
COPY --chown=appuser:appuser app/ ./app/

# Set ownership of nltk_data
RUN chown -R appuser:appuser /home/appuser/nltk_data

# Switch to non-root user
USER appuser

# Set NLTK data path for the user
ENV NLTK_DATA=/home/appuser/nltk_data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
# Using 4 workers for production; adjust based on CPU cores
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
