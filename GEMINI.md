# GEMINI.md - Fraud Detection Demo v4

## Project Overview
**Fraud Detection Demo v4** is a high-performance, real-time fraud detection pipeline designed to showcase GPU acceleration (NVIDIA L40S) and high-speed parallel I/O (Pure FlashBlade NFS). It uses a multi-stage distributed architecture to process credit card transactions using Graph Neural Networks (GNN) and XGBoost.

### Core Technologies
- **Runtime:** Kubernetes (Bare-metal)
- **Languages:** Python 3.11+
- **GPU Frameworks:** RAPIDS (cuDF, cuPy), PyTorch Geometric (GraphSAGE), XGBoost
- **Inference:** NVIDIA Triton Inference Server
- **Storage:** Pure FlashBlade NFS (ReadWriteMany)
- **Monitoring:** Prometheus, Grafana, DCGM (GPU Profiling)
- **Backend:** FastAPI, WebSockets, Chart.js

### Architecture
1.  **data-gather (GPU):** Generates synthetic transaction data using cuDF/cuPy.
2.  **data-prep (GPU):** Performs mega-batch feature engineering (40M+ rows) to saturate GPU streaming multiprocessors.
3.  **model-train (GPU):** Continuously retrains a 2-layer GraphSAGE GNN + XGBoost ensemble.
4.  **triton (GPU):** Serves the GNN and XGBoost models for real-time inference.
5.  **scoring (GPU):** Performs batch fraud scoring by maintaining a sliding-window graph of recent transactions.
6.  **backend (CPU):** FastAPI dashboard that orchestrates the pipeline by scaling K8s deployments and scraping telemetry.

---

## Building and Running

### Build & Deploy Workflow
The project uses a `Makefile` to manage builds on a remote Build VM and deployment to the K8s cluster.

```bash
# 1. Build all images (clones repo on Build VM, builds via Docker)
make build

# 2. Push images to the private registry
make push

# 3. Apply Kubernetes manifests (namespace, rbac, storage, deployments, services)
make deploy
```

### Operational Lifecycle
1.  **Generate Raw Data (Offline):**
    ```bash
    kubectl -n fraud-det-v31 scale deployment/data-gather --replicas=1
    # Wait for ~750 files in /data/raw, then scale to 0
    kubectl -n fraud-det-v31 scale deployment/data-gather --replicas=0
    ```
2.  **Build Initial Model:**
    ```bash
    kubectl apply -f k8s/jobs/model-build.yaml
    ```
3.  **Start Pipeline:**
    Access the dashboard at `http://<worker-node-ip>:30880` and click **Start**, or use:
    ```bash
    make start
    ```
4.  **Reset Environment:**
    ```bash
    make reset
    ```

---

## Development Conventions

### 1. GPU Processing & Subprocesses
- **Context Isolation:** To avoid CUDA context corruption in Python's main process, GPU-intensive work (cuDF/cuPy) is often delegated to a persistent subprocess using the `fork` multiprocessing context.
- **Mega-batching:** Logic should prioritize processing large batches (e.g., 20+ files, 40M+ rows) in a single GPU kernel launch to maximize L40S utilization.

### 2. Coordination (File-Queue Protocol)
The pipeline uses **Atomic NFS Renames** for coordination to avoid external dependencies like Redis or Kafka:
- `*.parquet`: Unclaimed file.
- `*.parquet.processing`: Claimed by a worker.
- `*.parquet.done`: Successfully processed.

### 3. Telemetry & Monitoring
- **Logging:** All pipeline stages must emit telemetry to `stdout` using the format:
  `[TELEMETRY] stage=<name> chunk_id=<id> rows=<count> ...`
- **Heartbeats:** Pods should touch `/tmp/.healthy` every loop iteration for Kubernetes liveness probes.
- **DCGM:** GPU metrics are scraped at 1s resolution for high-fidelity performance tracking.

### 4. Code Style
- **Type Hinting:** Extensive use of Python type hints is encouraged.
- **Environment Variables:** Configuration is strictly driven by environment variables (e.g., `INPUT_PATH`, `OUTPUT_PATH`, `BATCH_FILES`).
- **Error Handling:** GPU workers should be monitored by the parent; if a worker fails, the pod should exit to trigger a K8s restart.

### 5. Storage Layout
- `/data/raw/`: Raw parquet files from `data-gather`.
- `/data/features/`: Engineered features from `data-prep`.
- `/data/scores/`: Final fraud scores from `scoring`.
- `/data/models/`: Triton model repository and training artifacts.
