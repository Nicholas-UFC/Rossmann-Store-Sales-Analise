# syntax=docker/dockerfile:1

FROM python:3.14-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.18

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group api --no-install-project

COPY api ./api
COPY models ./models
COPY parameters ./parameters

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
