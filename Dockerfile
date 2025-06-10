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

# Create and activate virtual environment (better than --user install)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (with pip cache disabled)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn
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

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create static directory with correct permissions
RUN mkdir -p /app/static && \
    chmod 755 /app/static

RUN mkdir -p /app/static/basemap && \
    chmod 755 /app/static/basemap

RUN mkdir -p /app/static/coastline && \
    chmod 755 /app/static/coastline
RUN mkdir -p /app/static/eez && \
    chmod 755 /app/static/eez
RUN mkdir -p /app/static/legend && \
    chmod 755 /app/static/legend
RUN mkdir -p /app/static/maps && \
    chmod 755 /app/static/maps
RUN mkdir -p /app/static/pacificnames && \
    chmod 755 /app/static/pacificnames
RUN mkdir -p /app/static/thredds && \
    chmod 755 /app/static/thredds
RUN mkdir -p /app/static/tide && \
    chmod 755 /app/static/tide



EXPOSE 8000

# Optimized Uvicorn command
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--threads", "2", \
     "--timeout", "120", \
     "--keep-alive", "60", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "app.main:app"]