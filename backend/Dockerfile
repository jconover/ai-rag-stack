FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# - build-essential: Required for compiling some Python packages
# - curl, git: General utilities
# - libgomp1: Required for torch CPU operations
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~5GB by excluding CUDA)
# This must be done before requirements.txt to override the torch version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
# Note: torch is already installed above, pip will skip it
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model for semantic chunking (Phase 2)
# Using small model (en_core_web_sm) to minimize image size (~12MB vs ~560MB for lg)
RUN python -m spacy download en_core_web_sm

# Download NLTK data for text processing (Phase 2)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
# Using 4 workers for production; adjust based on CPU cores
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
