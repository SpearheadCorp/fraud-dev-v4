FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

WORKDIR /app

COPY pods/data-prep/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pods/data-prep/prepare.py .

CMD ["python", "prepare.py"]
