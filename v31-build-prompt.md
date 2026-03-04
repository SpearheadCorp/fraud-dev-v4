# Agent Prompt: Build Fraud Detection Demo v3.1

## Your Mission

Build **fraud-det-v3.1** — a clean, working, production-quality fraud detection demo that:
1. Starts from the **Pure Storage repo** as the base (clean, working baseline)
2. Cherry-picks the **best ideas** from the SpearheadCorp v3 repo (dashboard, stress load, dual volumes, fraud signal injection)
3. Closes the **key NVIDIA blueprint gaps** (real data seeding, temporal splits, SHAP explainability)
4. Avoids every known bug from SpearheadCorp v3 (documented below)

You are building this from scratch into a new directory. Do not clone any repo — I will give you the reference code details in this prompt. Write clean, tested, minimal code.

---

## Reference Repos (Read These First)

You should fetch and study these repos before writing any code:

- **NVIDIA blueprint**: `https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection`
- **Pure Storage base**: `https://github.com/PureStorage-OpenConnect/fraud-detection-demo`
- **SpearheadCorp v3** (for dashboard/backend patterns only): `https://github.com/SpearheadCorp/fraud-dev-v3`

---

## Architecture Decisions (Already Made — Do Not Revisit)

| Decision | Choice | Reason |
|---|---|---|
| Orchestration | **Docker Compose** (not K8s) | Simpler, works, no kubectl bugs, closer to NVIDIA |
| Pipeline model | **Batch with continuous trigger** | GNN-compatible, no streaming complexity |
| Queue mechanism | **Shared Docker volumes** (no file queue, no Redis) | Docker Compose native; simpler than SpearheadCorp's file queue |
| Backend control | **FastAPI** (Docker Compose native) | Clean rewrite; uses `docker compose` CLI not kubectl |
| GPU metrics | **DCGM Exporter + Prometheus** | Industry standard; feeds dashboard real GPU % |
| SHAP | **XGBoost native SHAP** (not Captum) | XGBoost 2.x has built-in `.predict(pred_contribs=True)` — no extra library needed |
| GNN | **Not in v3.1** — marked as v4.0 future work | Too complex for this sprint; XGBoost is sufficient |
| Data | **Kaggle-seeded synthetic** (see below) | Real statistical distributions without licensing issues |
| Train/test split | **Temporal** (not random 80/20) | Prevents data leakage; production-realistic |
| Image registry | **Local build only** | No Docker Hub push required |

---

## Repository Structure to Create

```
fraud-det-v31/
├── .env                          # Single source of truth for all config
├── .env.example                  # Committed template with placeholder values
├── docker-compose.yaml           # All services
├── Makefile                      # Convenience targets
├── README.md
│
├── pods/
│   ├── data-gather/              # Pod 1: Kaggle-seeded synthetic data generator
│   │   ├── Dockerfile
│   │   ├── gather.py
│   │   └── requirements.txt
│   │
│   ├── data-prep/                # Pod 2: Feature engineering (CPU then GPU)
│   │   ├── Dockerfile
│   │   ├── prepare.py
│   │   └── requirements.txt
│   │
│   ├── model-build/              # Pod 3: XGBoost training (CPU then GPU) + SHAP
│   │   ├── Dockerfile
│   │   ├── train.py
│   │   └── requirements.txt
│   │
│   ├── inference/                # Pod 4: Triton inference server
│   │   ├── Dockerfile            # Extends tritonserver with FIL backend
│   │   └── start.sh              # Writes config.pbtxt, starts tritonserver
│   │
│   └── backend/                  # Pod 5: FastAPI control plane + dashboard
│       ├── Dockerfile
│       ├── backend.py            # FastAPI app
│       ├── metrics.py            # Prometheus + DCGM + local psutil collection
│       ├── pipeline.py           # docker compose up/down/scale logic
│       ├── static/
│       │   └── dashboard.html    # Single-file dashboard (no npm, no build step)
│       └── requirements.txt
│
└── monitoring/
    ├── prometheus.yml            # Prometheus scrape config
    └── dcgm-config.yaml          # DCGM exporter config (if custom needed)
```

---

## Pod 1 — Data Gather (`pods/data-gather/gather.py`)

### What it does
Generates realistic synthetic transaction data using the **Kaggle Credit Card Fraud Detection dataset as a statistical seed**, then uses those learned distributions to generate unlimited volume via NumPy.

### Kaggle Seeding Pattern (CRITICAL — implement exactly this)

The Kaggle dataset (`creditcard.csv`) has columns: `Time`, `V1`–`V28` (PCA-anonymized), `Amount`, `Class` (fraud label).

**Do not use the Kaggle data directly as training data.** Use it only to fit statistical parameters:

```python
import pandas as pd
import numpy as np
from scipy import stats

def load_seed_distributions(seed_csv_path: str) -> dict:
    """
    Fit statistical distributions from Kaggle data.
    Returns distribution params used by generator — NOT the raw data.
    """
    df = pd.read_csv(seed_csv_path)

    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]

    return {
        'fraud_rate': len(fraud) / len(df),           # ~0.00172 (real rate)
        'amount_legit': stats.lognorm.fit(legit['Amount'].clip(lower=0.01)),
        'amount_fraud': stats.lognorm.fit(fraud['Amount'].clip(lower=0.01)),
        'hour_dist': np.histogram(df['Time'] % 86400 / 3600, bins=24, density=True),
        # Fraud is 3x more likely in hours 0-6 (real pattern from Kaggle)
        'fraud_hour_dist': np.histogram(fraud['Time'] % 86400 / 3600, bins=24, density=True),
    }
```

If `seed_csv_path` is not found (env var `KAGGLE_SEED_PATH` not set), fall back to hardcoded reasonable defaults (lognormal amount, flat fraud rate 0.5%).

### Schema to Generate (23 columns — matches Pure Storage schema)
`trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `first`, `last`, `gender`, `street`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`, `trans_num`, `unix_time`, `merch_lat`, `merch_long`, `is_fraud`, `merch_zipcode`

### Fraud Signal Injection (from SpearheadCorp — better ML signal)
Do NOT use flat random fraud. Inject realistic patterns:
- Categories `shopping_net`, `travel`, `misc_net` get **3x higher fraud rate**
- Fraud transactions have amounts drawn from `amount_fraud` distribution (higher mean)
- Fraud more likely in hours 0–5 (night)
- Fraud more likely when `distance_km` between customer and merchant > 500km

### Execution Model
Support two modes via `RUN_MODE` env var:
- `RUN_MODE=once` — generate `TARGET_ROWS` rows, write Parquet files, exit (for pipeline batch mode)
- `RUN_MODE=continuous` — loop indefinitely, write batches of `CHUNK_SIZE` rows, sleep briefly between batches (for dashboard demo mode)

### Parallelism
- Use `multiprocessing.Pool` with `NUM_WORKERS` workers (default: `os.cpu_count() // 2`)
- Each worker generates a chunk of `CHUNK_SIZE` rows (default: 100,000)
- Each chunk written as a separate Parquet file to `OUTPUT_PATH` (FlashBlade mount)
- Uncompressed output (`compression=None`) for max write throughput

### Telemetry (required — backend parses these lines)
Every 5 seconds emit to stdout:
```
[TELEMETRY] stage=gather rows_generated=500000 throughput_mbps=245.3 files_written=5 workers=8 fraud_rate=0.0051
```

### Stress Mode
When env var `STRESS_MODE=true`, multiply `NUM_WORKERS` by 4 and `CHUNK_SIZE` by 2. This is triggered by the backend when user hits the "Stress" button on the dashboard.

### Environment Variables
```
OUTPUT_PATH=/data/raw          # FlashBlade mount for raw Parquet files
NUM_WORKERS=8
CHUNK_SIZE=100000
FRAUD_RATE=0.005               # Overridden by Kaggle seed if available
TARGET_ROWS=1000000            # Used in 'once' mode only
RUN_MODE=once                  # 'once' or 'continuous'
STRESS_MODE=false              # Set to 'true' by backend during stress demo
KAGGLE_SEED_PATH=              # Optional path to creditcard.csv
```

### Dependencies (`requirements.txt`)
```
pandas>=2.1.0
numpy>=1.26.0
pyarrow>=14.0.0
faker>=22.0.0
scipy>=1.11.0
psutil>=5.9.0
```

### Dockerfile
Base: `python:3.11-slim`

---

## Pod 2 — Data Prep (`pods/data-prep/prepare.py`)

### What it does
Reads raw Parquet files from Pod 1, engineers features, writes feature Parquet files. Runs **CPU path first, then GPU path**, prints comparison table. Saves only GPU output (or CPU if no GPU available).

### Feature Engineering (21 features — matching Pure Storage)
From the 23 raw columns, produce:
```python
features = [
    'amt_log',           # log1p(amt)
    'amt_scaled',        # (amt - mean) / std
    'hour_of_day',       # from unix_time
    'day_of_week',       # from unix_time
    'is_weekend',        # int8 0/1
    'is_night',          # int8: hour in [0,5]
    'distance_km',       # haversine(lat, long, merch_lat, merch_long)
    'category_encoded',  # int8 ordinal (0-13)
    'state_encoded',     # int8 ordinal (0-49)
    'gender_encoded',    # int8: M=0, F=1
    'city_pop_log',      # log1p(city_pop)
    'zip_region',        # zip // 10000
    'amt',               # original
    'lat', 'long',
    'city_pop',
    'unix_time',
    'merch_lat', 'merch_long',
    'merch_zipcode', 'zip',
    'is_fraud',          # label — keep in feature file, train.py will split it out
]
```

### CPU vs GPU Pattern (from Pure Storage — keep exactly this)
```python
# Phase 1: CPU
cpu_result, cpu_timing = process_cpu(input_files)
print_timing("CPU", cpu_timing)

# Phase 2: GPU (graceful fallback if no GPU)
try:
    gpu_result, gpu_timing = process_gpu(input_files)
    print_timing("GPU", gpu_timing)
    print_speedup_table(cpu_timing, gpu_timing)
    output = gpu_result
except Exception as e:
    print(f"[WARN] GPU unavailable ({e}), using CPU output")
    output = cpu_result
```

### GPU Implementation
Use `cudf` for GPU DataFrame operations. Mirror every pandas operation 1:1 in cudf. Use `try/import` to fail gracefully:
```python
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
```

### Temporal Split (NVIDIA gap — CRITICAL)
After feature engineering, split data temporally (not randomly):
```python
def temporal_split(df):
    # Sort by unix_time
    df = df.sort_values('unix_time')
    n = len(df)
    train = df.iloc[:int(n * 0.70)]   # earliest 70%
    val   = df.iloc[int(n * 0.70):int(n * 0.85)]  # next 15%
    test  = df.iloc[int(n * 0.85):]   # most recent 15%
    return train, val, test
```
Write three output files: `features_train.parquet`, `features_val.parquet`, `features_test.parquet`

### Telemetry
```
[TELEMETRY] stage=prep files_read=10 rows_processed=1000000 cpu_time_s=45.2 gpu_time_s=3.8 speedup=11.9x output_size_mb=124
```

### Environment Variables
```
INPUT_PATH=/data/raw
OUTPUT_PATH=/data/features
```

### Dependencies
```
pandas>=2.1.0
numpy>=1.26.0
pyarrow>=14.0.0
cudf-cu12>=23.10.*   # installed in base image, not pip
```

### Dockerfile
Base: `nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10`

---

## Pod 3 — Model Build (`pods/model-build/train.py`)

### What it does
Trains XGBoost on features from Pod 2. Runs **CPU XGBoost first, then GPU XGBoost**. Produces two trained models for Triton. Computes **SHAP values** on test set. Writes Triton model repository.

### Training (CPU then GPU — from Pure Storage pattern)
```python
FEATURE_COLS = [col for col in train_df.columns if col != 'is_fraud']
LABEL_COL = 'is_fraud'

xgb_params = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': None,  # auto-calculated from class imbalance
    'eval_metric': 'aucpr',
    'early_stopping_rounds': 10,
}

# CPU training
cpu_model = train_xgboost(X_train, y_train, X_val, y_val, device='cpu', params=xgb_params)

# GPU training
gpu_model = train_xgboost(X_train, y_train, X_val, y_val, device='cuda', params=xgb_params)
```

### SHAP Values (NVIDIA gap — use XGBoost native, NOT Captum)
```python
def compute_shap(model, X_test, feature_names):
    """XGBoost native SHAP — no extra library needed."""
    # Get SHAP values (n_samples x n_features)
    shap_values = model.predict(xgb.DMatrix(X_test), pred_contribs=True)
    # Last column is bias term — drop it
    shap_values = shap_values[:, :-1]

    # Feature importance by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs_shap.tolist()))

    return {
        'top_features': sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10],
        'shap_values_sample': shap_values[:100].tolist()  # first 100 for dashboard
    }
```

Save SHAP results to `model_repository/shap_summary.json`.

### Triton Model Repository Output
After training, write:
```
model_repository/
├── fraud_xgboost_cpu/
│   ├── 1/
│   │   └── model.json          # XGBoost CPU model
│   └── config.pbtxt            # Triton FIL config (KIND_CPU)
├── fraud_xgboost_gpu/
│   ├── 1/
│   │   └── model.json          # XGBoost GPU model
│   └── config.pbtxt            # Triton FIL config (KIND_GPU)
└── shap_summary.json           # SHAP feature importance + sample values
```

### `config.pbtxt` Generation (must be correct — SpearheadCorp had bugs here)
```python
def write_triton_config(model_dir: str, model_name: str, kind: str, n_features: int):
    config = f"""name: "{model_name}"
backend: "fil"
max_batch_size: 8192
input [{{
  name: "input__0"
  data_type: TYPE_FP32
  dims: [ {n_features} ]
}}]
output [{{
  name: "output__0"
  data_type: TYPE_FP32
  dims: [ 1 ]
}}]
instance_group [{{ kind: {kind} count: 1 }}]
parameters [
  {{ key: "model_type"    value: {{ string_value: "xgboost_json" }} }},
  {{ key: "predict_proba" value: {{ string_value: "true" }} }},
  {{ key: "output_class"  value: {{ string_value: "false" }} }},
  {{ key: "threshold"     value: {{ string_value: "0.5" }} }}
]
"""
    with open(os.path.join(model_dir, 'config.pbtxt'), 'w') as f:
        f.write(config)
```

### Evaluation Metrics to Log
```python
metrics = {
    'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...,
    'auc_roc': ..., 'auc_pr': ...,
    'tp': ..., 'fp': ..., 'fn': ..., 'tn': ...,
    'threshold': 0.5,
    'n_train': len(X_train), 'n_test': len(X_test),
    'fraud_rate_train': y_train.mean(),
    'fraud_rate_test': y_test.mean(),
}
```
Save to `model_repository/training_metrics.json`.

### Telemetry
```
[TELEMETRY] stage=train cpu_train_time_s=42.1 gpu_train_time_s=4.3 speedup=9.8x f1_cpu=0.921 f1_gpu=0.924 auc_pr=0.887 shap_computed=true
```

### Environment Variables
```
INPUT_PATH=/data/features
MODEL_REPO=/data/models
MAX_SAMPLES=500000         # Cap training set if dataset is very large
```

### Dependencies
```
xgboost>=2.0.0
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.3.0
pyarrow>=14.0.0
psutil>=5.9.0
```

### Dockerfile
Base: `nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10`

---

## Pod 4 — Inference (`pods/inference/`)

### What it does
Runs NVIDIA Triton Inference Server serving both `fraud_xgboost_cpu` and `fraud_xgboost_gpu` models via the FIL backend. No custom Python code beyond the startup script.

### `start.sh`
```bash
#!/bin/bash
set -e

MODEL_REPO=${MODEL_REPO:-/data/models}

# Wait for models to exist (training pod may not be done yet)
echo "[INFO] Waiting for model repository at $MODEL_REPO..."
until [ -f "$MODEL_REPO/fraud_xgboost_gpu/1/model.json" ]; do
    echo "[INFO] Models not ready yet, waiting 10s..."
    sleep 10
done

echo "[INFO] Models found. Starting Triton..."
exec tritonserver \
    --model-repository="$MODEL_REPO" \
    --model-control-mode=poll \
    --repository-poll-secs=30 \
    --strict-model-config=false \
    --log-verbose=0 \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002
```

### Dockerfile
```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.02-py3
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
```

### Health Check
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
  interval: 10s
  timeout: 5s
  retries: 12
  start_period: 30s
```

---

## Pod 5 — Backend (`pods/backend/`)

### What it does
FastAPI app that:
1. Serves `dashboard.html` at `GET /`
2. Controls the pipeline (start/stop/reset/stress) via `docker compose` CLI subprocess calls
3. Streams real-time metrics to the dashboard via **WebSocket**
4. Collects metrics from: pod stdout (`[TELEMETRY]` lines), Prometheus (GPU via DCGM), local psutil (CPU/RAM)

### CRITICAL Implementation Notes

**Docker Compose native — NOT kubectl:**
```python
import subprocess

COMPOSE_FILE = os.environ.get('COMPOSE_FILE', '/app/docker-compose.yaml')
PROJECT_NAME = os.environ.get('COMPOSE_PROJECT', 'fraud-det-v31')

def compose_cmd(*args) -> str:
    """Run a docker compose command, return stdout."""
    result = subprocess.run(
        ['docker', 'compose', '-f', COMPOSE_FILE, '-p', PROJECT_NAME, *args],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"compose failed: {result.stderr}")
    return result.stdout
```

**No kubectl. No K8s. No pscp. No Redis.**

### API Endpoints (keep it focused — fewer is better)

```
GET  /                          → serves dashboard.html
GET  /api/status                → pipeline running status + pod health
POST /api/control/start         → start the pipeline (compose up data-gather data-prep model-build inference)
POST /api/control/stop          → stop pipeline (compose stop those services)
POST /api/control/reset         → stop + clear /data/raw and /data/features (preserve /data/models)
POST /api/control/stress        → set STRESS_MODE=true in data-gather, scale up (for demo spike)
POST /api/control/stress-stop   → return to normal load
GET  /api/metrics/current       → latest snapshot of all metrics (for polling fallback)
GET  /api/metrics/shap          → SHAP feature importance from model_repository/shap_summary.json
WS   /ws/dashboard              → WebSocket stream, emits JSON every 1s
GET  /metrics                   → Prometheus text format (for Prometheus scraping)
```

### Metrics Collection (`metrics.py`)

**Three sources — collect all, merge into one dict:**

```python
class MetricsCollector:
    def collect(self) -> dict:
        return {
            'telemetry': self._parse_telemetry(),     # from docker logs
            'system': self._collect_system(),          # psutil CPU/RAM
            'gpu': self._collect_gpu(),                # Prometheus/DCGM
            'business': self._compute_kpis(),          # derived from telemetry
            'timestamp': time.time(),
            'is_running': self.state.is_running,
        }

    def _parse_telemetry(self) -> dict:
        """
        Parse [TELEMETRY] lines from docker compose logs.
        Run: docker compose logs --no-log-prefix --tail=50 data-gather data-prep model-build inference
        Parse lines matching: [TELEMETRY] key=value key=value ...
        """

    def _collect_gpu(self) -> dict:
        """
        Query Prometheus for DCGM metrics.
        PromQL: DCGM_FI_DEV_GPU_UTIL, DCGM_FI_DEV_MEM_COPY_UTIL
        Fall back to zeros if Prometheus unreachable.
        """

    def _collect_system(self) -> dict:
        """psutil.cpu_percent(), psutil.virtual_memory()"""

    def _compute_kpis(self) -> dict:
        """
        Derive business KPIs from telemetry:
        - Total rows generated (sum across gather telemetry)
        - Fraud flagged count and rate
        - Throughput MB/s (from gather telemetry)
        - Estimated fraud exposure $ (fraud_count * avg_fraud_amount)
        """
```

**Docker logs telemetry collection:**
```python
def _parse_telemetry(self) -> dict:
    try:
        output = compose_cmd('logs', '--no-log-prefix', '--tail=100',
                             'data-gather', 'data-prep', 'model-build')
        result = {}
        for line in output.splitlines():
            if '[TELEMETRY]' in line:
                parts = line.split('[TELEMETRY]')[1].strip().split()
                kv = {}
                for part in parts:
                    if '=' in part:
                        k, v = part.split('=', 1)
                        try: kv[k] = float(v)
                        except: kv[k] = v
                stage = kv.pop('stage', 'unknown')
                result[stage] = kv
        return result
    except Exception:
        return {}
```

### Stress Mode Implementation
```python
@app.post("/api/control/stress")
async def start_stress():
    """Scale up data-gather to demonstrate GPU/CPU spike."""
    # Update running container env via docker compose up --scale or restart with new env
    # Simplest approach: write STRESS_MODE=true to a runtime config file that gather.py watches
    state.stress_mode = True
    stress_config_path = Path(os.environ['STRESS_CONFIG_PATH'])
    stress_config_path.write_text('STRESS_MODE=true\nNUM_WORKERS=32\nCHUNK_SIZE=200000\n')
    return {"status": "stress mode activated"}

@app.post("/api/control/stress-stop")
async def stop_stress():
    state.stress_mode = False
    stress_config_path = Path(os.environ['STRESS_CONFIG_PATH'])
    stress_config_path.write_text('STRESS_MODE=false\nNUM_WORKERS=8\nCHUNK_SIZE=100000\n')
    return {"status": "stress mode deactivated"}
```

In `gather.py`, poll a mounted `stress.conf` file every 5 seconds and reload `NUM_WORKERS`/`CHUNK_SIZE` dynamically without restarting.

### WebSocket Stream
```python
@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            metrics = collector.collect()
            await websocket.send_json(metrics)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
```

### WebSocket Payload Structure
```json
{
  "is_running": true,
  "elapsed_sec": 1234,
  "timestamp": 1700000000,
  "stress_mode": false,
  "system": {
    "cpu_percent": 67.4,
    "ram_percent": 54.2,
    "ram_used_gb": 8.7
  },
  "gpu": {
    "gpu_0_util_pct": 82.3,
    "gpu_0_mem_pct": 44.1,
    "gpu_1_util_pct": 0.0
  },
  "pipeline": {
    "gather": {
      "rows_generated": 1500000,
      "throughput_mbps": 312.4,
      "files_written": 15,
      "workers": 8,
      "fraud_rate": 0.0051
    },
    "prep": {
      "rows_processed": 1400000,
      "cpu_time_s": 45.2,
      "gpu_time_s": 3.8,
      "speedup": 11.9
    },
    "train": {
      "cpu_train_time_s": 42.1,
      "gpu_train_time_s": 4.3,
      "speedup": 9.8,
      "f1_gpu": 0.924,
      "auc_pr": 0.887,
      "shap_computed": true
    }
  },
  "business": {
    "total_transactions": 1500000,
    "fraud_flagged": 7650,
    "fraud_rate_pct": 0.51,
    "fraud_exposure_usd": 892340,
    "projected_annual_savings_usd": 4200000
  },
  "storage": {
    "raw_files": 15,
    "raw_size_gb": 2.4,
    "features_files": 3,
    "models_ready": true
  }
}
```

### Dependencies (`requirements.txt`)
```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
websockets>=12.0
psutil>=5.9.0
requests>=2.31.0
python-dotenv>=1.0.0
```

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Mount docker socket so we can run docker compose commands
# docker.sock is mounted in docker-compose.yaml
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
```

**Mount `/var/run/docker.sock` in docker-compose.yaml** so the backend can run `docker compose` commands.

---

## Dashboard (`pods/backend/static/dashboard.html`)

Single HTML file. No build step, no npm, no webpack. Pure HTML + CSS + vanilla JS + Chart.js CDN.

### Layout (3-tab design)
```
┌────────────────────────────────────────────────────────┐
│  Fraud Detection Demo v3.1   [LIVE ●]  [▶ Start] [■ Stop] [⚡ Stress] [↺ Reset]  │
│  00:20:34 elapsed                                       │
├─[Business]──[Infrastructure]──[Model/SHAP]─────────────┤
│                                                         │
│  BUSINESS TAB (default):                               │
│  ┌──────────┬──────────┬──────────┬──────────┐        │
│  │ $892,340 │  1.5M    │  7,650   │  0.51%   │        │
│  │ Exposure │  Txns    │ Flagged  │ Fraud %  │        │
│  └──────────┴──────────┴──────────┴──────────┘        │
│                                                         │
│  [Risk Score Distribution - bar chart]                  │
│  [Fraud by Category - horizontal bars]                  │
│  [Recent High-Risk Alerts - scrolling table]            │
│                                                         │
│  INFRASTRUCTURE TAB:                                    │
│  ┌────────────────────┬───────────────────────┐        │
│  │ CPU Utilization    │ GPU Utilization        │        │
│  │ [line chart]       │ [line chart]           │        │
│  └────────────────────┴───────────────────────┘        │
│  [Pipeline stage progress: Gather → Prep → Train → Infer]│
│  [Throughput: CPU vs GPU speedup table]                 │
│  [Storage: FlashBlade read/write MB/s gauges]           │
│                                                         │
│  MODEL/SHAP TAB:                                        │
│  [Model metrics: F1, AUC-PR, Precision, Recall]        │
│  [SHAP Feature Importance - horizontal bar chart]       │
│  [CPU vs GPU training time comparison]                  │
└────────────────────────────────────────────────────────┘
```

### WebSocket Connection (JS)
```javascript
const ws = new WebSocket(`ws://${location.host}/ws/dashboard`);
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateDashboard(data);
};
ws.onclose = () => {
    // Reconnect after 2s
    setTimeout(connectWS, 2000);
};
```

### Stress Mode Visual
When `data.stress_mode === true`:
- Flash a bright orange "⚡ STRESS MODE ACTIVE" banner
- CPU and GPU charts should visibly spike (because actual workload increases)
- The stress button becomes "Stop Stress" (red)

### Control Button Implementation
```javascript
async function startPipeline() {
    await fetch('/api/control/start', {method: 'POST'});
}
async function stressLoad() {
    await fetch('/api/control/stress', {method: 'POST'});
    document.getElementById('stress-btn').textContent = '⚡ Stop Stress';
    document.getElementById('stress-btn').onclick = stopStress;
}
```

### SHAP Visualization
On the Model/SHAP tab, call `GET /api/metrics/shap` once when tab is opened.
Render top 10 features as a horizontal bar chart (Chart.js) with feature names on Y axis, mean |SHAP| on X axis.

### Libraries (CDN only — no npm)
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
```

---

## Docker Compose (`docker-compose.yaml`)

```yaml
services:

  data-gather:
    build: ./pods/data-gather
    volumes:
      - raw_data:/data/raw
      - stress_config:/data/stress   # shared config file for hot-reload
    environment:
      - OUTPUT_PATH=/data/raw
      - STRESS_CONFIG_PATH=/data/stress/stress.conf
      - NUM_WORKERS=${NUM_WORKERS:-8}
      - CHUNK_SIZE=${CHUNK_SIZE:-100000}
      - RUN_MODE=${RUN_MODE:-once}
      - KAGGLE_SEED_PATH=${KAGGLE_SEED_PATH:-}
    profiles: ["pipeline"]           # only starts when explicitly included

  data-prep:
    build: ./pods/data-prep
    volumes:
      - raw_data:/data/raw:ro
      - features_data:/data/features
    environment:
      - INPUT_PATH=/data/raw
      - OUTPUT_PATH=/data/features
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    profiles: ["pipeline"]
    depends_on:
      data-gather:
        condition: service_completed_successfully

  model-build:
    build: ./pods/model-build
    volumes:
      - features_data:/data/features:ro
      - model_repo:/data/models
    environment:
      - INPUT_PATH=/data/features
      - MODEL_REPO=/data/models
      - MAX_SAMPLES=${MAX_SAMPLES:-500000}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    profiles: ["pipeline"]
    depends_on:
      data-prep:
        condition: service_completed_successfully

  inference:
    build: ./pods/inference
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - model_repo:/data/models:ro
    environment:
      - MODEL_REPO=/data/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-sf", "http://localhost:8000/v2/health/ready"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 60s

  backend:
    build: ./pods/backend
    ports:
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # docker compose control
      - raw_data:/data/raw:ro
      - features_data:/data/features:ro
      - model_repo:/data/models:ro
      - stress_config:/data/stress
      - ./docker-compose.yaml:/app/docker-compose.yaml:ro
    environment:
      - COMPOSE_FILE=/app/docker-compose.yaml
      - COMPOSE_PROJECT=${COMPOSE_PROJECT:-fraud-det-v31}
      - PROMETHEUS_URL=http://prometheus:9090
      - STRESS_CONFIG_PATH=/data/stress/stress.conf
    depends_on:
      - prometheus

  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "9400:9400"
    cap_add:
      - SYS_ADMIN

volumes:
  raw_data:        # FlashBlade NFS mount in production; Docker volume for local dev
  features_data:
  model_repo:
  stress_config:

networks:
  default:
    name: fraud-net
```

**Notes on profiles:**
- `docker compose up backend prometheus dcgm-exporter` — starts always-on services
- `docker compose --profile pipeline up` — triggers the full ML pipeline
- Backend calls `docker compose --profile pipeline up` when user clicks Start

---

## Makefile

```makefile
.PHONY: help build up pipeline stop reset stress clean

help:
	@echo "fraud-det-v3.1"
	@echo "  make build      - Build all Docker images"
	@echo "  make up         - Start backend + monitoring"
	@echo "  make pipeline   - Run the ML pipeline once"
	@echo "  make stop       - Stop all services"
	@echo "  make reset      - Stop and clear data volumes"
	@echo "  make stress     - Trigger stress mode via API"
	@echo "  make logs       - Tail all logs"
	@echo "  make clean      - Remove all containers and volumes"

build:
	docker compose build

up:
	docker compose up -d backend prometheus dcgm-exporter
	@echo "Dashboard: http://localhost:8080"
	@echo "Prometheus: http://localhost:9090"

pipeline:
	docker compose --profile pipeline up

stop:
	docker compose stop

reset:
	docker compose stop
	docker compose run --rm backend python -c "import shutil, os; [shutil.rmtree(p, ignore_errors=True) for p in ['/data/raw', '/data/features']]"

stress:
	curl -s -X POST http://localhost:8080/api/control/stress | python3 -m json.tool

stress-stop:
	curl -s -X POST http://localhost:8080/api/control/stress-stop | python3 -m json.tool

logs:
	docker compose logs -f

clean:
	docker compose down -v --remove-orphans
```

---

## `.env` / `.env.example`

```bash
# --- FlashBlade (production) or local Docker volumes (dev) ---
# Set these to NFS mount paths on FlashBlade to use real storage
# Leave blank to use Docker named volumes (for local dev)
FB_RAW_PATH=           # e.g., /mnt/flashblade/fraud/raw
FB_FEATURES_PATH=      # e.g., /mnt/flashblade/fraud/features
FB_MODELS_PATH=        # e.g., /mnt/flashblade/fraud/models

# --- Data Generation ---
NUM_WORKERS=8
CHUNK_SIZE=100000
TARGET_ROWS=1000000
RUN_MODE=once          # 'once' for pipeline run, 'continuous' for demo mode
KAGGLE_SEED_PATH=      # Optional: path to creditcard.csv from Kaggle

# --- Model Training ---
MAX_SAMPLES=500000

# --- Docker Compose ---
COMPOSE_PROJECT=fraud-det-v31

# --- NVIDIA ---
NVIDIA_API_KEY=        # For pulling from nvcr.io (ngc.nvidia.com)
```

---

## Prometheus Configuration (`monitoring/prometheus.yml`)

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']

  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8080']
    metrics_path: '/metrics'

  - job_name: 'triton'
    static_configs:
      - targets: ['inference:8002']
```

---

## Bugs to Explicitly Avoid (from SpearheadCorp v3 — do not repeat these)

1. **Missing `import os` in config files** — every Python file must import all stdlib modules it uses. Run `pylint` or `flake8` on every file before finishing.

2. **StorageClass/AccessMode mismatch** — N/A here (we use Docker Compose, not K8s), but: never use a volume `driver` that doesn't support the access pattern you need.

3. **GPU inference path untested** — test the Triton FIL backend config before calling the feature complete. Use `curl` to call the inference endpoint with a synthetic payload as part of README.

4. **Stale metrics after reset** — the `/api/control/reset` endpoint must clear the `MetricsCollector` state object in-memory, not just delete files.

5. **Pod naming inconsistencies** — pick names once (`data-gather`, `data-prep`, `model-build`, `inference`, `backend`) and use them consistently everywhere: docker-compose service names, API endpoint strings, dashboard dropdown values, telemetry stage names.

6. **`config.pbtxt` path mismatch** — the model JSON file must be at exactly `model_name/1/model.json` (not `xgboost.json`, not `model.json` in the wrong directory). Validate this in `train.py` before writing.

7. **FlashBlade saturation** — `gather.py` must check available disk space every 10 chunks. If usage > 80%, log a warning and pause. If > 95%, stop writing and emit `[TELEMETRY] stage=gather status=PAUSED reason=disk_full`.

8. **Random 80/20 split instead of temporal** — do not use `train_test_split(random_state=42)`. Use the temporal split function defined above.

9. **Docker socket not mounted** — the backend needs `/var/run/docker.sock` mounted to call `docker compose`. This is already in the docker-compose.yaml above — do not remove it.

10. **No graceful GPU fallback** — every GPU code path must have a `try/except` that falls back to CPU and logs a clear warning. Never let an `ImportError` on `cudf` crash a pod.

---

## Coding Standards

- **Python 3.11** throughout
- All files must be **importable without errors** (`python -c "import module"` must succeed)
- Every environment variable must have a **default value** in the script so it works locally without a `.env`
- **No bare `except:`** — always catch specific exceptions or `Exception` at minimum
- Use `pathlib.Path` for all file paths — no string concatenation for paths
- Log with **structured prefixes**: `[INFO]`, `[WARN]`, `[ERROR]`, `[TELEMETRY]`
- No print statements for errors — use `sys.stderr.write()` or Python `logging`
- Each pod script must handle `SIGTERM` gracefully (write any buffered output, close files)

---

## Testing Checklist (verify each before declaring done)

1. `docker compose build` completes with no errors
2. `make up` starts backend, prometheus, dcgm-exporter — dashboard loads at `http://localhost:8080`
3. `make pipeline` runs all 4 pods to completion with no errors
4. `model_repository/fraud_xgboost_gpu/1/model.json` exists after pipeline
5. `model_repository/shap_summary.json` exists with top 10 features
6. Triton health check passes: `curl http://localhost:8000/v2/health/ready`
7. Triton can serve inference: `curl -X POST http://localhost:8000/v2/models/fraud_xgboost_gpu/infer -d '...'`
8. Dashboard WebSocket connects and data streams update in real-time
9. Start button triggers pipeline, metrics update in dashboard
10. Stress button causes visible CPU/GPU utilization increase in Infrastructure tab
11. Stop Stress returns metrics to baseline
12. Reset button stops pipeline and clears raw/features data (models preserved)
13. SHAP tab shows feature importance chart
14. No pod name string mismatches between docker-compose, backend, and dashboard

---

## What v3.1 Does NOT Include (save for later)

- **GNN / GraphSAGE** — planned for v4.0; requires IBM TabFormer dataset and graph construction pipeline
- **Kubernetes** — Docker Compose only for v3.1; K8s manifest generation is a later milestone
- **S3 / FlashBlade S3** — not implemented
- **Multi-GPU Dask** — single GPU per pod; Dask CUDA cluster is future work
- **Real Kaggle training data** — used only as statistical seed for distribution fitting; raw Kaggle data not required at runtime
- **LLM / NIM microservices** — not in scope

---

## Recommended Build Order

1. Write and test `pods/data-gather/gather.py` — validate Parquet output shape
2. Write and test `pods/data-prep/prepare.py` — validate feature count and temporal split
3. Write and test `pods/model-build/train.py` — validate Triton model repo structure and SHAP JSON
4. Write `pods/inference/start.sh` + Dockerfile — validate Triton starts and serves
5. Write `docker-compose.yaml` — validate `make pipeline` runs end-to-end
6. Write `pods/backend/backend.py` + `metrics.py` — validate WebSocket stream
7. Write `pods/backend/static/dashboard.html` — validate charts render and controls work
8. Write `monitoring/prometheus.yml` — validate DCGM metrics appear in dashboard GPU charts
9. Write `Makefile`, `.env.example`, `README.md`
10. Run full testing checklist

---

## Final Note

This is a **demo first, production second** system. When in doubt, prefer simplicity over cleverness. A dashboard that clearly shows CPU at 20% jumping to 80% when Stress is clicked is worth more than a sophisticated metric aggregation pipeline that has bugs. Ship working code that tells a clear story.
