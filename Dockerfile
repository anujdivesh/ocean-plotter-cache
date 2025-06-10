# Stage 1: Build stage
FROM python:3.12-slim-bookworm as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgeos-c1v5 && \
    rm -rf /var/lib/apt/lists/*

# Create appuser first
RUN useradd -u 1000 -g 1000 -m appuser

# Create all static directories with correct permissions in one layer
RUN mkdir -p /app/static && \
    mkdir -p /app/static/{maps,legend,tide,basemap,eez,pacificnames,thredds,coastline} && \
    chown -R appuser:appuser /app/static && \
    chmod -R 775 /app/static

# After copying all files, ensure appuser owns everything
RUN chown -R appuser:appuser /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Ensure appuser owns the /app directory
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]