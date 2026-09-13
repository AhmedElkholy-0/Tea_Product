# syntax=docker/dockerfile:1

# ==========================================
# Stage 1: builder
# Installs dependencies into a virtual environment using uv.
# ==========================================
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system

# Install uv (the package manager used by this project)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dependency manifests first. Docker caches this layer, so
# `uv sync` only reruns when pyproject.toml or uv.lock actually change,
# not every time application code changes.
COPY pyproject.toml uv.lock README.md ./

# Install dependencies only (no dev extras) into /app/.venv, without
# installing the project itself yet (--no-install-project keeps this
# layer cacheable even before the source code is copied).
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the application source and install the project itself.
COPY src ./src
RUN uv sync --frozen --no-dev

# ==========================================
# Stage 2: runtime
# Copies only the virtual environment and the files needed to run the app.
# ==========================================
FROM python:3.13-slim AS runtime

# opencv needs these shared libraries at runtime even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Bring over the ready-to-use virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Copy the application code, configuration, and trained model.
COPY src ./src
COPY configs ./configs
COPY runs/detect/train-2/weights/best.onnx ./runs/detect/train-2/weights/best.onnx

# Run as a non-root user for safety.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# Bind to 0.0.0.0 so the app is reachable from outside the container.
CMD ["uvicorn", "src.tea_product.api:app", "--host", "0.0.0.0", "--port", "8000"]