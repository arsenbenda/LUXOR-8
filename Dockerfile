# LUXOR-8 Trading Bot - Dockerfile v7.2 STABLE
# NO TA-LIB - Pure Python Implementation
# Python 3.11+ - Alpine Linux (minimal footprint)

FROM python:3.11-alpine

# Metadata
LABEL maintainer="arsenbenda" \
      version="7.2.0" \
      description="LUXOR-8 Trading Bot - Zero Native Dependencies"

# Set working directory
WORKDIR /app

# Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install system dependencies (minimal - only runtime)
RUN apk add --no-cache \
    bash \
    curl \
    ca-certificates \
    tzdata && \
    rm -rf /var/cache/apk/*

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser -D -u 1000 luxor && \
    chown -R luxor:luxor /app

# Switch to non-root user
USER luxor

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Start application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
