# ============================================
# Pneumonia Detection API - Production Docker Build
# Optimized for Azure / Koyeb Deployment
# ============================================

# Stage 1: Builder - Install Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements_api.txt .

# Create wheels for faster installation in final stage
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements_api.txt


# Stage 2: Production Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user for security (important for cloud deployments)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder and install (faster than pip install from scratch)
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels && \
    rm -rf /root/.cache/pip && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY config.py .
COPY api.py .
COPY xray_validator.py .

# Copy trained model and configuration
COPY results/models/ ./results/models/
COPY results/training_config.json ./results/training_config.json

# Copy frontend files for static serving
COPY frontend/ ./frontend/

# Create startup script for running both API and serving frontend
COPY <<EOF /app/start.sh
#!/bin/bash
set -e

echo "============================================"
echo "  🫁 PNEUMONIA DETECTION API"
echo "  Platform: \${PLATFORM:-cloud}"
echo "============================================"

# Start the FastAPI application
exec python api.py
EOF

RUN chmod +x /app/start.sh

# Set ownership of all files to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# ============================================
# Environment Variables (can be overridden at runtime)
# ============================================

# Server Configuration
ENV HOST=0.0.0.0
# PORT: 7860 for Hugging Face Spaces, 8000 for other platforms
# Can be overridden at runtime
ENV PORT=7860

# TensorFlow Configuration (CPU-only for cloud deployment)
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=-1

# Python Configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Logging
ENV LOG_LEVEL=INFO

# Model paths (relative to /app)
ENV MODEL_PATH=/app/results/models/model_final.keras
ENV CONFIG_PATH=/app/results/training_config.json

# CORS - Allow all origins by default (override in production)
ENV ALLOWED_ORIGINS=*

# ============================================
# Expose Port & Health Check
# ============================================

# Expose the API port (7860 for HF Spaces, override with -p for other platforms)
EXPOSE 7860

# Health check for container orchestration (Azure, Koyeb)
# - interval: How often to check (30s)
# - timeout: How long to wait for response (10s)
# - start-period: Grace period for startup (120s for model loading)
# - retries: How many failures before unhealthy (3)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ============================================
# Start Command
# ============================================

# Use the startup script
CMD ["/app/start.sh"]
# Trigger rebuild Sun Dec 21 01:18:24 EET 2025
