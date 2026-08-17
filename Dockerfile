# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: build headers are not required for the pure-Python stack, but
# libpq is needed if a real PostgreSQL driver is later wired in.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install . && \
    pip install "gunicorn>=21,<23" "uvicorn[standard]>=0.30,<1.0"

COPY src ./src
COPY frontend ./frontend

# Secret-free image: all keys come from the container environment at runtime.
EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "paper_api.api:app", "--timeout", "120"]
