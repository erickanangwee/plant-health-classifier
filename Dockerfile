# Stage 1: builder — install Python packages into /install
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime — lean image with only what inference needs
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime system libraries for PyTorch / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code and configuration
COPY api/        ./api/
COPY src/        ./src/
COPY params.yaml ./params.yaml

# Copy trained champion model + leaf centroid
# These must exist locally before `docker build` is run.
# If they don't, run `dvc repro` first.
COPY models/champion/ ./models/champion/

# Run as a non-root user (security best practice)
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Liveness check: probe /health every 30s, allow 90s for cold start
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]