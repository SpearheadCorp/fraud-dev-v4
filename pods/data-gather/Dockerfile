# Build context: repo root (docker build -f pods/data-gather/Dockerfile .)
FROM nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.11

WORKDIR /app

COPY pods/data-gather/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pods/data-gather/gather.py .

# Seed data baked in at build time (optional — falls back to hardcoded defaults)
COPY seed-data/ /app/seed-data/

CMD ["python", "gather.py"]
