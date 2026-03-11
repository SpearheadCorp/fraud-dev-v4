FROM nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.11

WORKDIR /app

COPY pods/model-train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pods/model-train/train_continuous.py .

CMD ["python", "train_continuous.py"]
