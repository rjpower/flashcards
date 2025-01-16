FROM python/3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY scripts scripts/
COPY srt srt/

RUN pip install .

# Create directories for uploads and outputs
RUN mkdir -p data output
VOLUME ["/app/data", "/app/output"]

EXPOSE 8000

CMD ["python", "scripts/web.py"]
