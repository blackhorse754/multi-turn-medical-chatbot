FROM python:3.11-slim

WORKDIR /app

# build-essential covers any deps that need compiling on install
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Writable runtime dirs: session DB, logs, FAISS index, chat memory, source PDFs
RUN mkdir -p /app/data/log /app/interim /app/input/docs /app/scripts/sessionMemory

ENV DB_PATH=/app/data/session_history.db \
    FALLBACK_DIR=/app/data/log/fallback.db \
    PYTHONUNBUFFERED=1

# This is a multi-turn CLI chat loop (reads from stdin) — must be run with `docker run -it`
CMD ["python", "scripts/conversational_bot.py"]
