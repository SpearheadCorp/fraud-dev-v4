# Fraud Detection Demo v3.1

Production-quality fraud detection demo using XGBoost, Triton Inference Server, and a real-time web dashboard.

Baseline from Pure Storage · Best-of from SpearheadCorp v3 · NVIDIA blueprint gaps closed.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Docker Compose                           │
│                                                              │
│  [data-gather] ──> [data-prep] ──> [model-build] ──> [inference]  (profile: pipeline)
│                                                              │
│  [backend]  ←→  WebSocket  ←→  Browser Dashboard            │
│  [prometheus] ←─ DCGM exporter                              │
└──────────────────────────────────────────────────────────────┘
```

| Service | Image | Purpose |
|---|---|---|
| `data-gather` | python:3.11-slim | Synthetic transaction generator (Kaggle-seeded) |
| `data-prep` | rapidsai/base (CUDA) | Feature engineering — CPU path + GPU path (cudf) |
| `model-build` | rapidsai/base (CUDA) | XGBoost CPU + GPU training + SHAP |
| `inference` | tritonserver:24.02 | Triton FIL backend serving both models |
| `backend` | python:3.11-slim | FastAPI + WebSocket dashboard |
| `prometheus` | prom/prometheus | Metrics scraping |
| `dcgm-exporter` | dcgm-exporter | NVIDIA GPU metrics → Prometheus |

---

## Quick Start

### Prerequisites

- Docker + Docker Compose v2
- NVIDIA driver + Container Toolkit (for GPU acceleration)
- NVIDIA NGC API key (to pull `nvcr.io` images)

```bash
# 1. Authenticate with NVIDIA NGC
docker login nvcr.io -u '$oauthtoken' --password <YOUR_NGC_API_KEY>

# 2. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local dev)

# 3. Build images
make build

# 4. Start dashboard + monitoring
make up
# → Dashboard: http://localhost:8080
# → Prometheus: http://localhost:9090

# 5. Run the ML pipeline
make pipeline
```

### Using the seed data

If you have the Sparkov credit card dataset (`credit_card_transactions.csv` or `.csv.zip`):

```bash
# In .env:
KAGGLE_SEED_PATH=/path/to/credit_card_transactions.csv.zip
```

The generator fits statistical distributions from the seed file and generates unlimited synthetic data with those distributions. The raw seed data is never used for training.

---

## Dashboard

Open `http://localhost:8080` after `make up`.

**Tabs:**
- **Business** — fraud exposure, transaction counts, category breakdown, live alerts
- **Infrastructure** — CPU/GPU utilization charts, pipeline stage progress, storage stats
- **Model/SHAP** — F1/AUC-PR metrics, SHAP feature importance chart, CPU vs GPU speedup

**Controls:**
- **▶ Start** — trigger the ML pipeline
- **■ Stop** — stop pipeline services
- **⚡ Stress** — activate stress mode (3–4× data throughput spike)
- **↺ Reset** — stop pipeline and clear raw/feature data (models preserved)

---

## Testing the Triton Inference Endpoint

After the pipeline completes and Triton is healthy:

```bash
# Health check
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models

# Single inference (21 features as FP32)
curl -X POST http://localhost:8000/v2/models/fraud_xgboost_gpu/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input__0",
      "shape": [1, 21],
      "datatype": "FP32",
      "data": [[1.2, 0.3, 14, 2, 0, 0, 45.2, 6, 5, 1, 8.2, 1,
                89.5, 37.5, -122.1, 85000, 1388000000,
                37.7, -122.4, 94105.0, 94102]]
    }]
  }'
```

---

## File Structure

```
fraud-det-v31/
├── .env.example
├── docker-compose.yaml
├── Makefile
├── README.md
├── pods/
│   ├── data-gather/   gather.py, Dockerfile, requirements.txt
│   ├── data-prep/     prepare.py, Dockerfile, requirements.txt
│   ├── model-build/   train.py, Dockerfile, requirements.txt
│   ├── inference/     start.sh, Dockerfile
│   └── backend/       backend.py, metrics.py, pipeline.py, Dockerfile,
│                      requirements.txt, static/dashboard.html
├── monitoring/
│   ├── prometheus.yml
│   └── dcgm-config.yaml
└── seed-data/         (optional) credit_card_transactions.csv.zip
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Orchestration | Docker Compose | No kubectl bugs; simpler |
| Queue | Shared Docker volumes | Native; no Redis |
| GPU metrics | DCGM + Prometheus | Industry standard |
| SHAP | XGBoost native `pred_contribs=True` | No extra library |
| Train/test split | Temporal (70/15/15) | Prevents data leakage |
| GNN | Not in v3.1 | Deferred to v4.0 |

---

## Roadmap

- **v4.0** — GraphSAGE GNN on IBM TabFormer dataset
- **v4.0** — Multi-GPU Dask CUDA training
- **v4.1** — Kubernetes manifests with FlashBlade CSI
