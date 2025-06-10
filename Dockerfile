# Stage 1: Build stage
FROM python:3.12-slim-bookworm as builder

WORKDIR /app

# Install system dependencies (combine commands to reduce layers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (with pip cache disabled)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgeos-c1v5 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create appuser before copying files to ensure proper ownership
RUN useradd -m appuser && \
    mkdir -p /app/static && \
    mkdir -p /app/app && \
    chown -R appuser:appuser /app

# Switch to appuser for file operations
USER appuser

# Copy application code (now as appuser)
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create static subdirectories with correct permissions
RUN mkdir -p /app/static/{basemap,coastline,eez,legend,maps,pacificnames,thredds,tide}

EXPOSE 8000

# Optimized Uvicorn command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]