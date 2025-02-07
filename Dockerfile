# FROM python:3.12-slim-bookworm
FROM mcr.microsoft.com/playwright:v1.50.0-noble

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
RUN uv sync
RUN uv run playwright install chromium

COPY scripts scripts/
COPY srt srt/

# Create directories for uploads and outputs
RUN mkdir -p data output
VOLUME ["/app/data", "/app/output"]

EXPOSE 8000

CMD ["uv", "run", "gunicorn", "--timeout", "1200", "-w", "8", "-b", "0.0.0.0:8000", "scripts.web:app"]
