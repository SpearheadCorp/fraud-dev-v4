FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

WORKDIR /app

COPY pods/model-build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pods/model-build/train.py .

CMD ["python", "train.py"]
