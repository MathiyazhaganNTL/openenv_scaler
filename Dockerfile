# ──────────────────────────────────────────────────────────────
# Customer Support Environment — Dockerfile
# Compatible with HuggingFace Spaces (Docker SDK).
# HF Spaces expects port 7860 and a non-root user.
#
# Local usage:
#   docker build -t customer-support-env .
#   docker run -p 7860:7860 customer-support-env
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and enable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a non-root user (required by HF Spaces)t
RUN useradd -m -u 1000 user
USER user

# Set environment
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH="/app"

WORKDIR /app

# Copy application code (as non-root user)
COPY --chown=user:user . .

# Expose port (local default matching README)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run server on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
