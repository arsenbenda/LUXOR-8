# ============================================
# LUXOR v7.1 - PRODUCTION DOCKERFILE
# Optimized for Coolify Deployment
# Python 3.11 + TA-Lib Binary Wheels
# ============================================

FROM python:3.11-slim

# ============================================
# 1. SYSTEM DEPENDENCIES
# ============================================

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (NO TA-Lib compilation needed - use binary wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# 2. PYTHON ENVIRONMENT SETUP
# ============================================

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip==25.3

# Set working directory
WORKDIR /app

# ============================================
# 3. PYTHON PATH CONFIGURATION (FIX)
# ============================================

# Define PYTHONPATH BEFORE using it
ENV PYTHONPATH=/app:${PYTHONPATH:-}

# ============================================
# 4. INSTALL DEPENDENCIES
# ============================================

# Copy requirements first (Docker layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# Note: TA-Lib==0.5.1 usa binary wheels (no compilation)
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# 5. COPY APPLICATION CODE
# ============================================

# Copy all project files
COPY . /app/

# Ensure libs directory exists with __init__.py
RUN mkdir -p /app/libs && \
    touch /app/libs/__init__.py

# ============================================
# 6. RUNTIME CONFIGURATION
# ============================================

# Expose FastAPI port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# ============================================
# 7. HEALTHCHECK
# ============================================

# Use curl for healthcheck (already installed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ============================================
# 8. STARTUP COMMAND
# ============================================

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
