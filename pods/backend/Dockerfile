FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# docker.sock is mounted in docker-compose.yaml
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
