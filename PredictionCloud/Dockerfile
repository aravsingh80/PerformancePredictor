# Use Python 3.11 (TensorFlow is happiest here)
FROM python:3.11-slim

# Reduce logs + avoid TF trying GPU
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

# System deps (certs + basic build tooling just in case wheels need it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app + artifacts
COPY server.py .
COPY model_final.keras .
COPY nfl_player_gamelogs.parquet .

# Render provides $PORT
EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
