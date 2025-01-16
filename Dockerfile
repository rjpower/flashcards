FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml .
COPY README.md .
RUN uv sync

COPY scripts scripts/
COPY srt srt/

# Create directories for uploads and outputs
RUN mkdir -p data output
VOLUME ["/app/data", "/app/output"]

EXPOSE 8000

CMD ["uv", "run", "gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "scripts.web:app"]
