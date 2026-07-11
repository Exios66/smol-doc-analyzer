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

RUN pip install --no-cache-dir -e ".[discord]"

CMD ["python", "-m", "src.discord_bot", "--config-dir", "discord/smol-doc-analyzer"]
