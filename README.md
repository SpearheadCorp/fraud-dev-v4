# Fraud Detection Demo v4

Real-time GPU-accelerated fraud detection pipeline running on Kubernetes with NVIDIA L40S GPUs and Pure FlashBlade NFS storage. Demonstrates high GPU utilization through mega-batch processing, Graph Neural Networks, and GPU-accelerated I/O.

---

## Solution Overview

This demo detects fraudulent credit card transactions in real-time using a multi-stage GPU pipeline. Synthetic transactions are generated offline, then processed through feature engineering, GNN+XGBoost model training, and Triton-served inference — all orchestrated by a web dashboard.

### What Makes This Demo Interesting

- **GPU Saturation**: Mega-batch processing concatenates millions of rows into single GPU kernel launches that fill L40S streaming multiprocessors (99% peak utilization)
- **GNN Fraud Detection**: GraphSAGE-based Graph Neural Network builds a tri-partite graph (User - Transaction - Merchant) to capture relational fraud patterns, combined with XGBoost for tabular features
- **FlashBlade NFS Performance**: All pipeline stages read/write through NFS, demonstrating Pure FlashBlade's low-latency parallel I/O under GPU workloads
- **Pipelined I/O**: GPU processes batch N+1 while batch N writes to NFS in parallel threads, maximizing GPU utilization
- **Live Dashboard**: Real-time WebSocket updates at 200ms showing GPU utilization (DCGM profiling metrics), FlashBlade latency, fraud alerts, and pipeline throughput

---

## Architecture

```
  Pre-demo (offline)                  Live Demo Pipeline
  ==================                  ==================

  data-gather (GPU)                   data-prep (GPU)
  Synthetic transactions              Mega-batch feature engineering
  cuDF/cuPy generation                40M rows per batch
        |                             sort + groupby + merge on GPU
        v                                    |
  /data/raw/*.parquet                        v
  (~750 files, 5M rows each)         /data/features/*.parquet
                                             |
                                      +------+------+
                                      |             |
                                      v             v
                                scoring (GPU)   model-train (GPU)
                                Batch inference  Continuous GNN+XGBoost
                                via Triton       retraining (64 epochs)
                                      |             |
                                      v             v
                                /data/scores   /data/models
                                               (Triton hot-reloads)

  Backend (CPU) — FastAPI + WebSocket dashboard
  Reads pod logs for telemetry, queries Prometheus for GPU/CPU metrics,
  queries FlashBlade REST API for storage latency
```

### GPU Layout (4x NVIDIA L40S across 2 nodes)

| Node | GPU 0 | GPU 1 |
|------|-------|-------|
| Worker .44 | data-prep (feature engineering) | triton (inference server) |
| Worker .40 | model-train (GNN retraining) | scoring (fraud scoring) |

### Pipeline Stages

| Stage | Image Base | GPU Work | Throughput |
|-------|-----------|----------|-----------|
| **data-gather** | RAPIDS 24.12 | cuDF/cuPy synthetic generation | ~5M rows/file |
| **data-prep** | RAPIDS 24.12 | Mega-batch: haversine, groupby, sort, merge, rank | ~7M rows/sec feature engineering |
| **model-train** | PyTorch 2.7 | 2-layer GraphSAGE (35->64->32) + XGBoost | 64 epochs per cycle |
| **triton** | Triton 25.04 | GNN + XGBoost ensemble inference | Python backend |
| **scoring** | RAPIDS 24.12 | Batch cuDF reads + Triton client | 8 files per batch |
| **backend** | Python 3.11 | None (CPU only) | WebSocket at 200ms |

### Storage (Pure FlashBlade NFS)

| Volume | Size | Purpose |
|--------|------|---------|
| /data/raw | 500Gi | Raw synthetic transactions |
| /data/features | 100Gi | GPU-engineered features |
| /data/scores | 100Gi | Fraud scores from scoring |
| /data/models | 50Gi | Triton model repo + training metrics |

All volumes are ReadWriteMany NFS mounts. Pipeline coordination uses atomic NFS rename (file-queue protocol).

---

## ML Model

### Graph Neural Network (GraphSAGE)
- **Architecture**: 2-layer SAGEConv (35 input -> 64 hidden -> 32 output)
- **Graph**: Tri-partite — User nodes, Transaction nodes, Merchant nodes
- **Edges**: User-Transaction and Merchant-Transaction (bidirectional)
- **Training**: 64 epochs per cycle on up to 500K transactions

### XGBoost Ensemble
- **Input**: 35 tabular features + 32 GNN embeddings = 67 features
- **Features include**: amount stats, temporal (hour/day/weekend/night), haversine distance, per-customer velocity, per-category z-scores, per-merchant z-scores, percentile ranks
- **Output**: Fraud probability (0-1)

### Inference
- Triton serves both models via a Python backend
- Scoring pod maintains a sliding-window graph for context
- New transactions are scored against the most recent graph state

---

## Dashboard

**URL:** `http://<worker-node-ip>:30880`

### Business Tab
- Total transactions processed, TX/s throughput
- Fraud detection rate, flagged transactions, exposure estimate
- Category-level breakdown, live fraud alerts table

### Infrastructure Tab
- **GPU Utilization**: Per-GPU engine activity (DCGM profiling at 1s resolution) for all 4 GPUs
- **CPU Utilization**: Per-node CPU % from node-exporter
- **FlashBlade Latency**: Read/write latency from FlashBlade REST API
- **NFS Storage**: File counts and sizes across all volumes

### Model Tab
- Training metrics (F1, AUC-PR, loss curves)
- Architecture diagram
- Feature importance from latest training cycle

### Controls
- **Start** — scales up 4 GPU pipeline pods
- **Stop** — scales all pods to 0
- **Reset** — stops pipeline, re-queues raw data, clears features/scores

---

## Quick Start

See **[HOW-TO.md](HOW-TO.md)** for complete build, deploy, and troubleshooting instructions.

### TL;DR

```bash
# 1. Apply K8s manifests
kubectl apply -f k8s/namespace.yaml -f k8s/rbac.yaml -f k8s/storage.yaml
kubectl apply -f k8s/deployments.yaml -f k8s/services.yaml

# 2. Build all images on build VM (see HOW-TO.md for details)

# 3. Generate raw data (offline, pre-demo)
kubectl -n fraud-det-v31 scale deployment/data-gather --replicas=1
# Wait for ~750 files, then:
kubectl -n fraud-det-v31 scale deployment/data-gather --replicas=0

# 4. Build initial model
kubectl apply -f k8s/jobs/model-build.yaml

# 5. Open dashboard and click Start
# http://<worker-node-ip>:30880
```

---

## File Structure

```
fraud-det-v4/
├── k8s/                          Kubernetes manifests
│   ├── namespace.yaml            Namespace (fraud-det-v31)
│   ├── rbac.yaml                 ServiceAccount + ClusterRole
│   ├── storage.yaml              FlashBlade NFS PV/PVC
│   ├── deployments.yaml          All deployments + Triton service
│   ├── services.yaml             Backend NodePort
│   ├── jobs/model-build.yaml     Initial model build Job
│   ├── dcgm-servicemonitor.yaml  GPU metrics scraping
│   └── dcgm-custom-metrics.yaml  Custom DCGM metrics config
├── pods/
│   ├── backend/                  FastAPI backend + dashboard
│   │   ├── backend.py            HTTP/WebSocket server
│   │   ├── metrics.py            Metrics collection (K8s logs, Prometheus, FlashBlade API)
│   │   ├── pipeline.py           Pipeline control (K8s deployment scaling)
│   │   └── static/dashboard.html Single-page dashboard (Chart.js)
│   ├── data-gather/              Synthetic data generator (GPU)
│   │   └── gather.py             cuDF/cuPy transaction generation
│   ├── data-prep/                Feature engineering (GPU)
│   │   ├── prepare.py            Orchestrator (file-queue, GPU worker subprocess)
│   │   └── gpu_worker.py         Mega-batch cuDF feature engineering
│   ├── scoring/                  Fraud scoring (GPU)
│   │   └── scorer.py             Batch cuDF reads + Triton inference + graph
│   ├── model-train/              Continuous training (GPU)
│   │   └── train_continuous.py   GraphSAGE + XGBoost training loop
│   ├── model-build/              Initial model build (Job)
│   │   └── train.py              One-shot model training
│   └── triton/                   Triton Inference Server
│       ├── Dockerfile            Triton 25.04 + model polling
│       └── start.sh              Startup script with NFS model polling
├── CHANGELOG.md                  Development history + GPU utilization journey
├── HOW-TO.md                     Build, deploy, run, and troubleshoot guide
└── README.md                     This file
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Orchestration | Kubernetes (bare-metal) | Matches customer environments; GPU scheduling |
| Storage | FlashBlade NFS (ReadWriteMany) | Demo target product; shared across all pods |
| Pipeline coordination | Atomic NFS rename (file-queue) | No Redis/Kafka dependency; race-safe |
| GPU framework | RAPIDS cuDF/cuPy | NFS->GPU direct reads; GIL-free for parallel I/O |
| Batch strategy | Mega-batch (40M+ rows) | Single large kernel fills L40S SMs vs many tiny launches |
| GNN | GraphSAGE (2-layer) | Captures relational fraud patterns; fast training |
| Inference | Triton Python backend | Serves GNN+XGBoost ensemble; hot model reload |
| Dashboard | Single HTML + Chart.js CDN | Zero build step; corporate proxy compatible |
| GPU metrics | DCGM profiling (1s resolution) | Shows actual kernel activity, not 30s averages |
| Write pipelining | Background thread pool | GPU processes next batch while NFS writes complete |
