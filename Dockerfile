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

# Create appuser FIRST
RUN useradd -m appuser

# Create directory structure with correct ownership
RUN mkdir -p /app/static/{basemap,coastline,eez,legend,maps,pacificnames,thredds,tide,plots} && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app/static

# Switch to appuser before copying files
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]