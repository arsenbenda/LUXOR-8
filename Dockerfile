# LUXOR Trading Bot - Production Dockerfile
# Base: Python 3.11 slim (stable + TA-Lib wheels available)
# Build tested: 2026-01-13

FROM python:3.11-slim

# Metadata
LABEL maintainer="arsenbenda"
LABEL version="7.1.0"
LABEL description="LUXOR Trading Bot - BTC Strategy with TA-Lib"

# Environment Variables (MUST be defined BEFORE usage)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    APP_HOME=/app \
    PYTHONPATH=/app

# Create app directory
WORKDIR $APP_HOME

# System dependencies
# TA-Lib: binary wheels available on PyPI, no build deps needed
# curl: for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# ⚠️ Order matters: pip -> requirements -> cleanup
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy application code
COPY . .

# Security: Non-root user
RUN useradd -m -u 1000 luxor && \
    chown -R luxor:luxor $APP_HOME
USER luxor

# Expose port
EXPOSE 8000

# Healthcheck (validates app is responding)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command (production mode)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
