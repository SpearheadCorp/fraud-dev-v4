# How-To: Build, Deploy, and Run the Fraud Detection Demo (v4)

Step-by-step guide to deploy and operate the GPU-accelerated fraud detection pipeline on a bare-metal Kubernetes cluster with NVIDIA L40S GPUs and Pure FlashBlade NFS storage.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Cluster Setup](#2-cluster-setup)
3. [Build All Images](#3-build-all-images)
4. [Deploy to Kubernetes](#4-deploy-to-kubernetes)
5. [Pre-Demo: Generate Raw Data](#5-pre-demo-generate-raw-data)
6. [Pre-Demo: Build Initial Model](#6-pre-demo-build-initial-model)
7. [Running the Demo](#7-running-the-demo)
8. [Resetting Between Demos](#8-resetting-between-demos)
9. [Troubleshooting](#9-troubleshooting)
10. [Architecture Reference](#10-architecture-reference)

---

## 1. Prerequisites

### Hardware
- **2x GPU worker nodes** with 2x NVIDIA L40S each (4 GPUs total)
- **1x Build VM** with Docker and a private registry
- **Pure FlashBlade** NFS storage (or any NFS server)
- **K8s control plane** (kubectl access from your workstation)

### Software
- Kubernetes 1.28+ with NVIDIA GPU Operator installed
- Docker on the build VM
- `kubectl` configured on your workstation
- `ssh` access to the build VM
- `gh` CLI (optional, for GitHub operations)

### Network
- Private Docker registry running on build VM (HTTP, port 5000)
- All K8s nodes must have the registry configured as insecure:
  ```
  # /etc/containerd/config.toml or /etc/docker/daemon.json
  # Add: "insecure-registries": ["<BUILD_VM_IP>:5000"]
  ```

### Environment Variables (used throughout this guide)
```bash
export REGISTRY=10.23.181.247:5000/fraud-det-v4    # Private registry
export BUILD_VM=tduser@10.23.181.247                # Build VM SSH
export SSH_KEY=~/.ssh/id_rsa                        # SSH key
export NS=fraud-det-v31                             # K8s namespace
```

---

## 2. Cluster Setup

### 2a. Verify GPU Operator
```bash
# Check GPU operator is running
kubectl get pods -n gpu-operator

# Verify GPUs are visible on each node
kubectl describe node <node-name> | grep nvidia.com/gpu

# Should show:
#   nvidia.com/gpu: 2
# for each worker node
```

### 2b. Verify DCGM Exporter (GPU metrics)
```bash
# Check DCGM exporter is scraping
kubectl get pods -n gpu-operator -l app=nvidia-dcgm-exporter

# Test metrics endpoint
kubectl exec -n gpu-operator <dcgm-pod> -- curl -s localhost:9400/metrics | grep DCGM_FI_PROF_GR_ENGINE_ACTIVE
```

### 2c. Verify Prometheus
```bash
# Check Prometheus is running
kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus

# Test a GPU query from your workstation
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
# Then open: http://localhost:9090 and query: DCGM_FI_PROF_GR_ENGINE_ACTIVE
```

### 2d. Create NFS Directories on FlashBlade
```bash
# One-time setup: create the NFS directory structure
kubectl run setup -n $NS --image=busybox --restart=Never --rm -it \
  --overrides='{"spec":{"volumes":[{"name":"fb","nfs":{"server":"10.23.181.65","path":"/financial-fraud-detection-demo"}}],"containers":[{"name":"setup","image":"busybox","command":["sh"],"volumeMounts":[{"name":"fb","mountPath":"/fb"}]}]}}' \
  -- sh -c "mkdir -p /fb/v31/raw /fb/v31/features /fb/v31/features-cpu /fb/v31/models && chmod -R 777 /fb/v31"
```

---

## 3. Build All Images

### 3a. Clone the Repo on Build VM
```bash
ssh -i $SSH_KEY $BUILD_VM "cd /home/tduser && git clone https://github.com/SpearheadCorp/fraud-dev-v4.git"
```

### 3b. Build and Push All Images
```bash
ssh -i $SSH_KEY $BUILD_VM 'cd /home/tduser/fraud-det-v4 && git pull && \
  REGISTRY=10.23.181.247:5000/fraud-det-v4 && \
  docker build -t $REGISTRY/backend:latest       -f pods/backend/Dockerfile . && \
  docker build -t $REGISTRY/data-gather:latest   -f pods/data-gather/Dockerfile . && \
  docker build -t $REGISTRY/data-prep:latest     -f pods/data-prep/Dockerfile . && \
  docker build -t $REGISTRY/triton:latest        -f pods/triton/Dockerfile . && \
  docker build -t $REGISTRY/scoring:latest       -f pods/scoring/Dockerfile . && \
  docker build -t $REGISTRY/model-train:latest   -f pods/model-train/Dockerfile . && \
  docker build -t $REGISTRY/model-build:latest   -f pods/model-build/Dockerfile . && \
  docker push $REGISTRY/backend:latest && \
  docker push $REGISTRY/data-gather:latest && \
  docker push $REGISTRY/data-prep:latest && \
  docker push $REGISTRY/triton:latest && \
  docker push $REGISTRY/scoring:latest && \
  docker push $REGISTRY/model-train:latest && \
  docker push $REGISTRY/model-build:latest'
```

### 3c. Build a Single Image (after code changes)
```bash
# Example: rebuild just backend after editing backend.py
git add <files> && git commit -m "message" && git push

ssh -i $SSH_KEY $BUILD_VM 'cd /home/tduser/fraud-det-v4 && git pull && \
  REGISTRY=10.23.181.247:5000/fraud-det-v4 && \
  docker build -t $REGISTRY/backend:latest -f pods/backend/Dockerfile . && \
  docker push $REGISTRY/backend:latest'

kubectl -n $NS rollout restart deployment/backend
```

### 3d. Verify Registry Contents
```bash
# List all images in registry
curl -s http://10.23.181.247:5000/v2/_catalog | python3 -m json.tool

# List tags for a specific image
curl -s http://10.23.181.247:5000/v2/fraud-det-v4/backend/tags/list

# Check build VM disk space
ssh -i $SSH_KEY $BUILD_VM "df -h /"

# Free build cache (safe — does NOT break registry)
ssh -i $SSH_KEY $BUILD_VM "docker builder prune -af"

# WARNING: NEVER run "docker image prune -af" — breaks registry layer references!
```

---

## 4. Deploy to Kubernetes

### 4a. Apply All Manifests
```bash
# Apply in order: namespace first, then RBAC, storage, deployments, services
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/rbac.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/deployments.yaml
kubectl apply -f k8s/services.yaml

# Verify PVs are bound
kubectl get pv,pvc -n $NS
```

### 4b. Set FlashBlade API Token (for latency metrics)
```bash
# Get your FlashBlade API token and set it as an env var on the backend
kubectl -n $NS set env deployment/backend FLASHBLADE_API_TOKEN="<your-token>"
```

### 4c. Verify Backend is Running
```bash
kubectl -n $NS get pods -l app=backend
# Should show 1/1 Running

# Check dashboard is accessible
curl -s http://<worker-node-ip>:30880/api/status
# Should return JSON with is_running: false
```

**Dashboard URL:** `http://<worker-node-ip>:30880`

---

## 5. Pre-Demo: Generate Raw Data

Data-gather runs **offline before the demo** to fill `/data/raw` with synthetic transactions. It does not run during the live demo.

### 5a. Scale Up Data-Gather
```bash
# Start 1 gather pod (each produces ~5M-row files at ~440MB each)
kubectl -n $NS scale deployment/data-gather --replicas=1

# Watch it generate files
kubectl -n $NS logs -f deployment/data-gather --tail=20
```

### 5b. Monitor Progress
```bash
# Count raw files and total size
kubectl -n $NS exec deployment/backend -- sh -c \
  'ls /data/raw/*.parquet 2>/dev/null | wc -l && du -sh /data/raw/'

# Target: ~750 files (5M rows each) for ~40 min pipeline runtime
```

### 5c. Stop Data-Gather When Done
```bash
kubectl -n $NS scale deployment/data-gather --replicas=0
```

---

## 6. Pre-Demo: Build Initial Model

The model-build Job creates the initial GNN+XGBoost model that Triton serves. You need at least a few feature files first.

### 6a. Run a Quick Data-Prep Batch
```bash
# Start data-prep to process a few raw files into features
kubectl -n $NS scale deployment/data-prep --replicas=1

# Wait for at least one batch to complete
kubectl -n $NS logs -f deployment/data-prep --tail=20
# Look for: "batch 000000: ... rows"

# Stop data-prep
kubectl -n $NS scale deployment/data-prep --replicas=0
```

### 6b. Run Model-Build Job
```bash
kubectl apply -f k8s/jobs/model-build.yaml

# Watch progress
kubectl -n $NS logs -f job/model-build --tail=50

# Verify model files exist
kubectl -n $NS exec deployment/backend -- ls -la /data/models/fraud_gnn_gpu/1/
# Should show: state_dict_gnn.pth, xgboost.json, config.pbtxt, model.py
```

### 6c. Re-Queue Raw Data for Demo
```bash
# Reset: renames .done files back to .parquet so the demo can process them
curl -X POST http://<worker-node-ip>:30880/api/control/reset
```

---

## 7. Running the Demo

### 7a. Open Dashboard
Navigate to `http://<worker-node-ip>:30880` in a browser.

### 7b. Start Pipeline
Click the **Start** button on the dashboard. This scales up 4 GPU pods:
- **data-prep** (GPU) — mega-batch feature engineering on 40M rows
- **triton** (GPU) — GNN+XGBoost inference server
- **scoring** (GPU) — batch fraud scoring with cuDF
- **model-train** (GPU) — continuous GNN retraining

### 7c. What to Watch
- **GPU Utilization**: data-prep should spike to 99% during mega-batch processing
- **TX/s**: ~7M rows/sec feature engineering throughput
- **FlashBlade Latency**: read/write latency from NFS storage
- **Fraud Rate**: percentage of transactions flagged as fraudulent
- **Queue Depth**: raw files pending vs processed

### 7d. Stop Pipeline
Click **Stop** on the dashboard. All pipeline pods scale to 0.

---

## 8. Resetting Between Demos

### 8a. Via Dashboard
Click **Reset** on the dashboard. This:
1. Stops all pipeline pods
2. Renames `.done`/`.processing` raw files back to `.parquet`
3. Clears `/data/features` and `/data/scores`

### 8b. Via CLI
```bash
curl -X POST http://<worker-node-ip>:30880/api/control/reset
```

### 8c. Full Manual Reset (nuclear option)
```bash
# Stop everything
kubectl -n $NS scale deployment/data-prep --replicas=0
kubectl -n $NS scale deployment/triton --replicas=0
kubectl -n $NS scale deployment/scoring --replicas=0
kubectl -n $NS scale deployment/model-train --replicas=0
kubectl -n $NS scale deployment/data-gather --replicas=0

# Re-queue raw data
kubectl -n $NS exec deployment/backend -- python3 -c "
from pathlib import Path
raw = Path('/data/raw')
n = 0
for suffix in ('.done', '.processing'):
    for f in raw.glob(f'*{suffix}'):
        f.rename(f.with_name(f.name[:-len(suffix)]))
        n += 1
print(f'Re-queued {n} files')
"

# Clear downstream data
kubectl -n $NS exec deployment/backend -- sh -c "rm -rf /data/features/* /data/scores/*"

# Verify
kubectl -n $NS exec deployment/backend -- sh -c \
  'echo "Raw: $(ls /data/raw/*.parquet 2>/dev/null | wc -l) files"; \
   echo "Features: $(ls /data/features/*.parquet 2>/dev/null | wc -l) files"; \
   echo "Scores: $(ls /data/scores/*.parquet 2>/dev/null | wc -l) files"'
```

---

## 9. Troubleshooting

### Pod Status and Logs

```bash
# Overview of all pods
kubectl -n $NS get pods -o wide

# Detailed pod status (events, conditions, node placement)
kubectl -n $NS describe pod <pod-name>

# Live logs (follow)
kubectl -n $NS logs -f deployment/data-prep --tail=50
kubectl -n $NS logs -f deployment/scoring --tail=50
kubectl -n $NS logs -f deployment/model-train --tail=50
kubectl -n $NS logs -f deployment/triton --tail=50
kubectl -n $NS logs -f deployment/backend --tail=50

# Previous container logs (after crash/restart)
kubectl -n $NS logs <pod-name> --previous

# All pods and their status at once
kubectl -n $NS get pods -o custom-columns='NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName,RESTARTS:.status.containerStatuses[0].restartCount,AGE:.metadata.creationTimestamp'
```

### GPU Issues

```bash
# Check GPU allocation across all pods
kubectl -n $NS get pods -o json | python3 -c "
import json,sys
pods = json.load(sys.stdin)['items']
for p in pods:
    name = p['metadata']['name']
    for c in p['spec'].get('containers', []):
        gpu = c.get('resources', {}).get('limits', {}).get('nvidia.com/gpu', '0')
        node = p['spec'].get('nodeName', 'pending')
        print(f'{name:40s} GPU={gpu} node={node}')
"

# Check GPU utilization via Prometheus (from workstation)
# Query: DCGM_FI_PROF_GR_ENGINE_ACTIVE
# Shows 0-1 range per GPU per node

# Check GPU utilization directly on a node
ssh -i $SSH_KEY tduser@10.23.181.44 "nvidia-smi"
ssh -i $SSH_KEY tduser@10.23.181.40 "nvidia-smi"

# Check GPU memory on a node
ssh -i $SSH_KEY tduser@10.23.181.44 "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"

# DCGM profiling metrics (1-second resolution GPU engine activity)
ssh -i $SSH_KEY tduser@10.23.181.44 "dcgmi dmon -e 1001,1002,1003 -d 1000"
# 1001 = GR Engine Active, 1002 = SM Active, 1003 = SM Occupancy
```

### Pod Stuck in Pending

```bash
# Usually means not enough GPU slots
kubectl -n $NS describe pod <pending-pod-name>
# Look for Events: "Insufficient nvidia.com/gpu"

# Check what's using GPUs
kubectl get pods --all-namespaces -o json | python3 -c "
import json,sys
pods = json.load(sys.stdin)['items']
for p in pods:
    for c in p['spec'].get('containers', []):
        gpu = c.get('resources', {}).get('limits', {}).get('nvidia.com/gpu', '0')
        if gpu != '0':
            print(f\"{p['metadata']['namespace']:20s} {p['metadata']['name']:40s} GPU={gpu} node={p['spec'].get('nodeName','pending')}\")
"

# If a terminated pod is still holding a GPU slot, delete it
kubectl -n $NS delete pod <stuck-pod-name>
```

### Liveness Probe Failures (CrashLoopBackOff)

```bash
# Check if the pod was killed by liveness probe
kubectl -n $NS describe pod <pod-name> | grep -A5 "Last State"
# Look for: "Reason: OOMKilled" or "Liveness probe failed"

# Check the liveness file timestamp from inside the pod
kubectl -n $NS exec <pod-name> -- python3 -c "
import os, time
mtime = os.path.getmtime('/tmp/.healthy')
age = time.time() - mtime
print(f'Last heartbeat: {age:.0f}s ago (threshold: 120s)')
"
```

### NFS / FlashBlade Issues

```bash
# Check NFS mounts inside a pod
kubectl -n $NS exec deployment/backend -- df -h /data/raw /data/features /data/scores /data/models

# Check file counts and sizes
kubectl -n $NS exec deployment/backend -- sh -c '
echo "=== Raw ===" && ls /data/raw/*.parquet 2>/dev/null | wc -l && du -sh /data/raw/
echo "=== Features ===" && ls /data/features/*.parquet 2>/dev/null | wc -l && du -sh /data/features/
echo "=== Scores ===" && ls /data/scores/*.parquet 2>/dev/null | wc -l && du -sh /data/scores/
echo "=== Models ===" && ls -la /data/models/fraud_gnn_gpu/1/ 2>/dev/null
'

# Check for stuck .processing files (claimed but never finished)
kubectl -n $NS exec deployment/backend -- sh -c \
  'ls /data/raw/*.processing 2>/dev/null | wc -l && ls /data/features/*.processing 2>/dev/null | wc -l'

# Manually re-queue stuck .processing files
kubectl -n $NS exec deployment/backend -- python3 -c "
from pathlib import Path
for d in ['/data/raw', '/data/features']:
    for f in Path(d).glob('*.processing'):
        f.rename(f.with_name(f.name.replace('.processing', '')))
        print(f'Re-queued: {f.name}')
"

# Check FlashBlade latency via REST API
kubectl -n $NS exec deployment/backend -- python3 -c "
import requests, os, urllib3
urllib3.disable_warnings()
ip = os.environ.get('FLASHBLADE_MGMT_IP', '10.23.181.60')
token = os.environ.get('FLASHBLADE_API_TOKEN', '')
if not token: print('No FLASHBLADE_API_TOKEN set'); exit()
r = requests.post(f'https://{ip}/api/login', headers={'api-token': token}, verify=False, timeout=5)
st = r.headers.get('X-Auth-Token')
r2 = requests.get(f'https://{ip}/api/2.24/file-systems/performance',
    headers={'X-Auth-Token': st}, params={'names': 'financial-fraud-detection-demo'}, verify=False, timeout=5)
item = r2.json()['items'][0]
print(f\"Read:  {item.get('usec_per_read_op', 0)/1000:.2f} ms\")
print(f\"Write: {item.get('usec_per_write_op', 0)/1000:.2f} ms\")
"
```

### Triton Issues

```bash
# Check Triton readiness
kubectl -n $NS exec deployment/triton -- curl -s localhost:8000/v2/health/ready
# Should return 200

# Check loaded models
kubectl -n $NS exec deployment/triton -- curl -s localhost:8000/v2/models | python3 -m json.tool

# Check model config
kubectl -n $NS exec deployment/triton -- curl -s localhost:8000/v2/models/fraud_gnn_gpu/config | python3 -m json.tool

# Check if model files exist on NFS
kubectl -n $NS exec deployment/backend -- ls -la /data/models/fraud_gnn_gpu/1/
# Must contain: state_dict_gnn.pth, xgboost.json, config.pbtxt, model.py
```

### Scoring Pod Can't Reach Triton

```bash
# Test DNS resolution
kubectl -n $NS exec <scoring-pod> -- python3 -c "import socket; print(socket.getaddrinfo('triton', 8000))"

# Test HTTP connectivity
kubectl -n $NS exec <scoring-pod> -- python3 -c "
import requests
r = requests.get('http://triton:8000/v2/health/ready', timeout=5)
print(f'Status: {r.status_code}')
"
```

### Dashboard / Backend Issues

```bash
# Check backend API
curl -s http://<node-ip>:30880/api/status | python3 -m json.tool
curl -s http://<node-ip>:30880/api/metrics | python3 -m json.tool

# Check WebSocket (wscat or browser console)
# ws://<node-ip>:30880/data/dashboard

# Check backend logs for errors
kubectl -n $NS logs deployment/backend --tail=100 | grep -i error

# Restart backend (picks up new image)
kubectl -n $NS rollout restart deployment/backend
kubectl -n $NS rollout status deployment/backend
```

### Deployment Rollout

```bash
# Restart a specific deployment (pulls latest image)
kubectl -n $NS rollout restart deployment/<name>

# Watch rollout progress
kubectl -n $NS rollout status deployment/<name>

# Check image being used
kubectl -n $NS get deployment <name> -o jsonpath='{.spec.template.spec.containers[0].image}'

# Force pull latest image (if imagePullPolicy is Always)
kubectl -n $NS delete pod -l app=<name>
```

### Telemetry / Metrics Debugging

```bash
# Check what telemetry the backend is seeing from pod logs
kubectl -n $NS logs deployment/data-prep --tail=20 | grep TELEMETRY
kubectl -n $NS logs deployment/scoring --tail=20 | grep TELEMETRY
kubectl -n $NS logs deployment/model-train --tail=20 | grep TELEMETRY

# Example telemetry line (data-prep):
# [TELEMETRY] stage=prep chunk_id=5 rows=40000000 gpu_time_s=21.985 feat_time_s=5.712 gpu_used=1 batch_files=8

# Check cached telemetry
kubectl -n $NS exec deployment/backend -- cat /data/models/last_telemetry.json | python3 -m json.tool
```

### Build VM Issues

```bash
# Check registry is running
curl -s http://10.23.181.247:5000/v2/_catalog

# Check disk space
ssh -i $SSH_KEY $BUILD_VM "df -h /"

# Free build cache (safe)
ssh -i $SSH_KEY $BUILD_VM "docker builder prune -af"

# Check running containers
ssh -i $SSH_KEY $BUILD_VM "docker ps"

# NEVER run "docker image prune -af" — breaks registry layer references
```

---

## 10. Architecture Reference

### Pipeline Flow
```
data-gather (offline) → /data/raw/*.parquet
                              ↓
data-prep (GPU) → mega-batch 40M rows → /data/features/*.parquet
                              ↓                    ↓
                     scoring (GPU) ←── triton (GPU)
                              ↓                    ↑
                     /data/scores/*.parquet   model-train (GPU)
                                              reads /data/features
                                              writes /data/models
```

### GPU Layout (4x L40S, 2 nodes)
```
Node .44 (slc6-lg-n3-b30-29):
  GPU 0 → data-prep    (mega-batch feature engineering, 99% peak)
  GPU 1 → triton        (GNN+XGBoost inference server)

Node .40 (slc6-lg-n3-b30-25):
  GPU 0 → model-train   (continuous GNN retraining)
  GPU 1 → scoring        (batch fraud scoring with cuDF)
```

### NFS Volumes (FlashBlade @ 10.23.181.65)
| PVC | Path | Size | Purpose |
|-----|------|------|---------|
| raw-data-pvc | /v31/raw | 500Gi | Raw parquet from data-gather |
| features-data-pvc | /v31/features | 100Gi | GPU features from data-prep |
| features-cpu-data-pvc | /v31/features-cpu | 100Gi | Scores from scoring (repurposed) |
| model-repo-pvc | /v31/models | 50Gi | Triton model repo + training metrics |

### Services
| Service | Type | Port | URL |
|---------|------|------|-----|
| backend | NodePort | 8080 -> 30880 | `http://<node-ip>:30880` |
| triton | ClusterIP | 8000/8001/8002 | `triton:8000` (internal) |

### File-Queue Protocol
All pipeline stages use atomic NFS rename for coordination:
- `chunk.parquet` — unclaimed file, ready for processing
- `chunk.parquet.processing` — claimed by a worker (atomic rename)
- `chunk.parquet.done` — finished processing
- If rename fails with ENOENT, another worker claimed it (race-safe)

### Key Environment Variables
| Variable | Pod | Default | Purpose |
|----------|-----|---------|---------|
| BATCH_FILES | data-prep | 10 | Files per mega-batch (10 x 5M = 50M rows) |
| CHUNK_SIZE | data-gather | 10000000 | Rows per raw file |
| GNN_EPOCHS | model-train | 64 | GraphSAGE training epochs |
| GNN_HIDDEN | model-train | 64 | GNN hidden dimension |
| GNN_OUT | model-train | 32 | GNN output dimension |
| TRAIN_INTERVAL_SEC | model-train | 30 | Seconds between training cycles |
| WINDOW_CHUNKS | scoring | 2 | Sliding window depth for graph |
| TRITON_RETRIES | scoring | 30 | Retries for Triton connection |
