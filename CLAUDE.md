# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

GPU-accelerated fraud detection demo pipeline running on Kubernetes with NVIDIA L40S GPUs and Pure FlashBlade NFS storage. Seven containerized Python services coordinate via atomic NFS file renaming (no Redis/Kafka).

## Deployment Overview

**Branch-based isolation** — the Makefile and manual commands auto-target the right environment:

| Branch | Namespace       | NodePort | Image tag | Dashboard                   |
| ------ | --------------- | -------- | --------- | --------------------------- |
| `main` | `fraud-det-v31` | `30880`  | `latest`  | `http://10.23.181.44:30880` |
| `dev`  | `fraud-det-dev` | `30881`  | `dev`     | `http://10.23.181.44:30881` |

Build VM (all image builds happen here): `tduser@10.23.181.247`

---

## Git Branching Model

### Branch roles

| Branch              | Role                                                                                               |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| `main`              | Production — always deployable, only branch pushed to remote origin                                |
| `dev`               | Integration branch for completed feature/bugfix work                                               |
| `feature/<desc>`    | Created off `dev`, merged back to `dev` via PR                                                     |
| `bugfix/<desc>`     | Non-urgent fix — created off `dev`, merged back to `dev` via PR                                    |
| `hotfix/<desc>`     | Urgent production fix — created off `main`, merged to `main` via PR, then cherry-picked into `dev` |
| `release/<version>` | Stabilization branch cut from `dev`, merged into `main` via PR                                     |

### PR rules

- **All merges are done via GitHub PRs** — never `git merge` locally
- PRs into `main` require review + CI pass
- `feature/*` and `bugfix/*` → PR into `dev`
- `hotfix/*` and `release/*` → PR into `main`, then cherry-pick fix back into `dev`
- Never delete branches after merge

### Quick reference

**Feature / Bugfix (dev)**

```bash
git switch dev
git switch -c feature/<desc>   # or bugfix/<desc>
# work & commit
git push -u origin feature/<desc>
gh pr create --base dev --head feature/<desc> --title "feat: ..." --body "..."
```

**Hotfix (production)**

```bash
git switch main && git pull origin main
git switch -c hotfix/<desc>
# fix & commit
git push -u origin hotfix/<desc>
gh pr create --base main --head hotfix/<desc> --title "fix: ..." --body "..."
# After PR merges — cherry-pick into dev:
git switch dev
git cherry-pick <fix-commit-sha>
```

**Release (dev → main)**

```bash
git switch dev
git switch -c release/<version>
# stabilize, then:
git push -u origin release/<version>
gh pr create --base main --head release/<version> --title "Release <version>" --body "..."
# After PR merges — cherry-pick release fixes into dev:
git switch dev
git cherry-pick <release-fix-shas>
```

### Commit prefixes

`feat:` · `fix:` · `chore:` · `refactor:`

### Build VM remote setup

The build VM (`tduser@10.23.181.247`) has two remotes:

- `origin` → `https://github.com/SpearheadCorp/fraud-dev-v31.git`
- `v4` → `https://github.com/SpearheadCorp/fraud-dev-v4.git`

To pull a branch from v4 on the build VM:

```bash
git fetch v4 && git checkout -b <branch> v4/<branch>
# or if branch exists locally:
git checkout <branch> && git pull v4 <branch>
```

---

## Production Deployment (Mac/Linux — `make` available)

```bash
# Full build all images + push + deploy (slowest, use when changing multiple pods)
make up

# Individual steps if needed
make build    # SSH to build VM, git pull, docker build all 7 images
make push     # Push all images to registry
make deploy   # kubectl apply all k8s/ manifests with namespace/tag substitution
```

```bash
# Pipeline control
make start    # POST /api/control/start — scales all pods 0→1
make stop     # POST /api/control/stop
make reset    # POST /api/control/reset

# Observability
make status   # kubectl get pods -n fraud-det-v31
make logs     # stream backend logs
```

---

## Dev Deployment (Windows PowerShell — no `make`)

> Build VM: `10.23.181.247` | Namespace: `fraud-det-dev` | Dashboard: `http://10.23.181.44:30881`

### Which pods need rebuilding?

| Changed files               | Action                                                 |
| --------------------------- | ------------------------------------------------------ |
| `pods/backend/**`           | Rebuild `backend`                                      |
| `pods/scoring/**`           | Rebuild `scoring`                                      |
| `pods/data-prep/**`         | Rebuild `data-prep`                                    |
| `pods/model-train/**`       | Rebuild `model-train`                                  |
| `pods/triton/**`            | Rebuild `triton`                                       |
| `pods/data-gather/**`       | Rebuild `data-gather`                                  |
| `k8s/deployments.yaml` only | **No rebuild** — just reapply manifests + restart pods |

### Step 1 — Free space on build VM (if needed)

```bash
ssh tduser@10.23.181.247
df -h /   # need ~30GB free; scoring/triton images are ~10-20GB each

# Remove ONLY the image being rebuilt (never prune :latest prod images)
docker rmi 10.23.181.247:5000/fraud-det-dev/<pod>:dev
docker builder prune -f
```

### Step 2 — Pull latest code on build VM

```bash
cd /home/tduser/fraud-det-v31
git fetch v4 dev && git checkout dev && git pull v4 dev
```

### Step 3 — Build and push changed image(s)

```bash
# Replace <pod> with: backend | scoring | data-prep | model-train | triton | data-gather
docker build --no-cache -t 10.23.181.247:5000/fraud-det-dev/<pod>:dev -f pods/<pod>/Dockerfile . && \
docker push 10.23.181.247:5000/fraud-det-dev/<pod>:dev
```

### Step 4 — Apply k8s manifests (from Windows PowerShell)

Always run after any change to `k8s/` files, even if no image was rebuilt:

```powershell
# Apply demo-config ConfigMap first (required by backend — must exist before pod starts)
(Get-Content k8s/demo-config.yaml) `
  -replace 'fraud-det-v31','fraud-det-dev' | kubectl apply -f -

(Get-Content k8s/deployments.yaml) `
  -replace 'fraud-det-v31','fraud-det-dev' `
  -replace 'fraud-det-v4','fraud-det-dev' `
  -replace ':latest',':dev' | kubectl apply -f -
```

### Step 5 — Restart changed pods

```powershell
# Replace with the pods you actually changed
kubectl rollout restart deployment/scoring deployment/backend -n fraud-det-dev
kubectl rollout status deployment/scoring deployment/backend -n fraud-det-dev
```

### Step 6 — Verify

```powershell
kubectl get pods -n fraud-det-dev -o wide

# Check logs for errors (replace <pod> with the pod name)
kubectl logs -n fraud-det-dev deployment/<pod> --tail=30

# Confirm scoring pipeline is writing files end-to-end
kubectl exec -n fraud-det-dev deployment/scoring -- ls /data/scores/

# Check metrics API
curl http://10.23.181.44:30881/api/metrics | python3 -m json.tool
```

### Pipeline control (dev)

```powershell
curl -X POST http://10.23.181.44:30881/api/control/start
curl -X POST http://10.23.181.44:30881/api/control/stop
curl -X POST http://10.23.181.44:30881/api/control/reset
```

### Common issues

| Symptom                                      | Cause                              | Fix                                                                                                                          |
| -------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `No space left on device` on build VM        | Docker image cache full            | `docker rmi` the pod being rebuilt + `docker builder prune -f` (never use `docker image prune -a` — deletes prod images too) |
| Pod stuck `Pending`                          | GPU still held by old pod          | `kubectl delete pod -l app=<pod> -n fraud-det-dev --force --grace-period=0`                                                  |
| `CreateContainerError` on node .40           | containerd snapshot corruption     | Stop kubelet+containerd, `rm -rf /var/lib/containerd/*`, restart both                                                        |
| Scoring `RESOURCE_EXHAUSTED` gRPC errors     | GNN payload exceeds 2GB gRPC limit | `BATCH_FILES` must stay ≤ 2 in deployments.yaml — larger values exceed gRPC's 2GB message cap                                |
| FBms not showing on dashboard                | `FLASHBLADE_API_TOKEN` missing     | Verify env var is set in backend deployment in deployments.yaml                                                              |
| Triton crashes immediately showing help text | Unknown CLI flag passed            | Check triton deployment for invalid command overrides in deployments.yaml                                                    |

## Pipeline Control

The dashboard at `http://10.23.181.44:30880` (prod) or `:30881` (dev) provides start/stop/reset controls via UI.

## Local Development (Docker Compose)

```bash
docker-compose up --profile=pipeline
```

Uses Docker named volumes instead of NFS. GPU allocation via `resources.reservations.devices`.

## Architecture

### Data Flow

**Pre-demo (offline):** `data-gather` generates ~750 parquet files (5M rows each) → `/data/raw` (500Gi NFS)

**Live demo pipeline:**

```
data-prep (GPU) → /data/features → scoring (GPU) → Triton (GPU)
                               └→ model-train (GPU) → /data/models → Triton (hot-reload)
```

All pipeline coordination is via **atomic NFS rename** — a pod claims a file by renaming it to `*.processing`, preventing double-processing without external queue services.

### Pod Responsibilities

| Pod           | GPU           | Role                                         |
| ------------- | ------------- | -------------------------------------------- |
| `data-gather` | Node .40 GPU0 | Synthetic transaction generation (cuDF/cuPy) |
| `data-prep`   | Node .44 GPU0 | Feature engineering mega-batch (cuDF)        |
| `model-train` | Node .40 GPU0 | Continuous GraphSAGE + XGBoost retraining    |
| `triton`      | Node .44 GPU1 | Model serving via Triton Python backend      |
| `scoring`     | Node .40 GPU1 | Batch fraud scoring via Triton inference     |
| `backend`     | CPU only      | FastAPI control plane + WebSocket dashboard  |
| `model-build` | K8s Job       | One-shot initial model build (pre-demo)      |

### Key Design: Persistent GPU Worker Subprocess

`data-prep` splits into two processes to avoid cuDF SIGSEGV crashes from CUDA context reuse:

- `prepare.py` — orchestrator, file-queue management, spawns worker via `subprocess.Popen`
- `gpu_worker.py` — long-lived GPU worker in isolated CUDA context, receives jobs via stdin JSON, responds via stdout JSON

When adding GPU processing logic to `data-prep`, all cuDF/cuPy operations must go in `gpu_worker.py`.

### ML Model

GraphSAGE (2-layer GNN) + XGBoost ensemble:

- Graph captures tri-partite relationships: `card → merchant`, `card → transaction`, `merchant → transaction`
- XGBoost uses GNN embeddings + tabular features
- Triton serves both via a single Python backend (`pods/triton/`) that loads from `/data/models`

### Backend Metrics Architecture

`pods/backend/metrics.py` collects from three sources:

1. **Kubernetes API** — pod status, replica counts
2. **Prometheus** — DCGM GPU metrics, pipeline throughput
3. **FlashBlade REST API** — NFS read/write latency

Metrics are broadcast to all WebSocket clients every 200ms.

## NFS Storage Layout

| Mount            | Size  | Consumer                                      |
| ---------------- | ----- | --------------------------------------------- |
| `/data/raw`      | 500Gi | data-gather writes, data-prep reads           |
| `/data/features` | 100Gi | data-prep writes, scoring + model-train read  |
| `/data/scores`   | 100Gi | scoring writes                                |
| `/data/models`   | 50Gi  | model-train + model-build write, triton reads |

All volumes are `ReadWriteMany` (NFS). PV/PVC defined in `k8s/storage.yaml`.

## Kubernetes Specifics

- **Namespace:** `fraud-det-v31` (prod) / `fraud-det-dev` (dev)
- **Node pinning:** GPU pods use `nodeSelector` to pin to specific nodes for GPU scheduling control
- **Liveness probes:** All GPU pods use file-touch heartbeat at `/tmp/.healthy` (exec probe every 30s)
- **`data-prep` uses `Recreate` strategy** (not `RollingUpdate`) — cannot run >1 replica due to GPU context
- **RBAC:** `backend-sa` ServiceAccount needs `ClusterRole` with deployment scaling permissions (`k8s/rbac.yaml`)

## Monitoring

- **Prometheus:** `http://10.23.181.44:30090` — DCGM scrape interval is 200ms
- **DCGM Exporter:** GPU metrics at 1s resolution, custom metrics in `monitoring/dcgm-config.yaml`
- **Dashboard tabs:** Business metrics, Infrastructure (GPU/storage), Model performance

## Technology Stack

- **GPU data processing:** RAPIDS 24.12 (cuDF, cuPy)
- **ML:** PyTorch 2.7 + PyTorch Geometric 2.6.1 (GraphSAGE), XGBoost 3.0.2
- **Inference:** Triton Inference Server 25.04
- **Backend:** FastAPI 0.110+, Uvicorn, websockets 12.0+
- **Dashboard:** Single HTML file with Chart.js (CDN, no build step)
- **Base images:** `nvcr.io/nvidia/rapidsai/base:24.12-cuda12.0-py3.11` for GPU pods; `python:3.11-slim` for backend
