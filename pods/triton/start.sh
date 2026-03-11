#!/bin/bash
set -e

MODEL_REPO="${MODEL_REPO:-/data/models}"

echo "[INFO] Waiting for GNN model (fraud_gnn_gpu)..."
until [ -s "${MODEL_REPO}/fraud_gnn_gpu/1/state_dict_gnn.pth" ] && \
      [ -f "${MODEL_REPO}/fraud_gnn_gpu/config.pbtxt" ]; do
    echo "[INFO] Model not ready yet, waiting 10s..."
    sleep 10
done

echo "[INFO] GNN model ready. Starting Triton Inference Server..."
exec tritonserver \
    --model-repository="${MODEL_REPO}" \
    --model-control-mode=poll \
    --repository-poll-secs=30 \
    --strict-model-config=false \
    --log-verbose=0 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
