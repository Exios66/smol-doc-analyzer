FROM python:3.12-slim

WORKDIR /workspace

# System deps for PyMuPDF / pillow wheels; keep image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY taxonomy ./taxonomy
COPY discord ./discord

RUN pip install --no-cache-dir -e ".[discord]" \
    && chmod +x discord/smol-doc-analyzer/entrypoint.sh

ENTRYPOINT ["discord/smol-doc-analyzer/entrypoint.sh"]
CMD ["discord/smol-doc-analyzer"]
