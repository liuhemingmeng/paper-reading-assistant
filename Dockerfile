FROM python:3.11-slim AS base

# Optional PyPI mirror passed at build time (e.g. a China mirror for faster
# installs). Defaults to upstream PyPI so the image stays portable.
ARG PIP_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=$PIP_INDEX_URL \
    FRONTEND_DIR=/app/frontend

WORKDIR /app

# System deps: build headers are not required for the pure-Python stack, but
# libpq is needed if a real PostgreSQL driver is later wired in.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy source BEFORE installing the project: `pip install .` builds the wheel
# from src/, so the package tree must already be present in the build context.
COPY pyproject.toml ./
COPY src ./src
COPY frontend ./frontend

RUN pip install --upgrade pip && \
    pip install . && \
    pip install "gunicorn>=21,<23" "uvicorn[standard]>=0.30,<1.0"

# Secret-free image: all keys come from the container environment at runtime.
EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "--preload", "-b", "0.0.0.0:8000", "paper_api.api:app", "--timeout", "120"]
