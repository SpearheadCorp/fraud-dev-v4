# Changelog — fraud-det-v4

Development history from v3.1 baseline through v4 GPU-optimized demo.

---

## Phase 1: v3.1 Baseline + K8s Migration (Mar 3)

| Commit | Change |
|--------|--------|
| `79369c9` | **Initial commit**: v3.1 codebase — Docker Compose, 5 pods (gather, prep, train, triton, backend), XGBoost-only, Prometheus+DCGM |
| `32fdd41` | **K8s migration**: manifests, pipeline.py rewired to K8s API (scale deployments, read pod logs), metrics from pod logs |
| `a119e5e` | Fix Dockerfile COPY paths for repo-root build context |
| `0f17cef` | Pin all pods to node .44 — node .40 lacked insecure registry config |

**GPU utilization at this point:** Near-zero. Data-prep processed tiny CSV-to-parquet conversions; GPU was idle 95%+ of the time.

---

## Phase 2: Stabilization + Dashboard (Mar 4)

| Commit | Change |
|--------|--------|
| `8f3724e` | Switch to gpu-operator's dcgm-exporter (remove manual deployment) |
| `14daae4` | Fix numpy conflict in RAPIDS images |
| `be58a44`-`4517966` | Fix Triton model filename (`model.json` -> `xgboost.json`) |
| `878ec0c` | Add CPU pods for GPU vs CPU side-by-side comparison |
| `f7d25c6` | Code review pass: fix pipeline stability + demo UX |
| `de3e550` | Remove custom Prometheus; use cluster kube-prometheus-stack |
| `76f9a62` | Inference pod lifecycle management (start/stop, not always-on) |
| `93a828f` | Dashboard: TX/s chart, combined 4-line infra chart, pipeline funnel |
| `4ae284d`-`27da420` | **RAPIDS image fix**: tried 25.02/CUDA 12.6, settled on 24.12/CUDA 12.5 for driver 580 compatibility |
| `063022e`-`d4b0d70` | FlashBlade latency chart, Y-axis tuning (2-3ms range) |
| `a08377a` | Enable GPU metrics in dashboard, fix prep/train showing '--' |

**GPU utilization:** ~5%. Small chunk sizes (10K rows) meant GPU kernels finished instantly. Most time spent in Python overhead.

---

## Phase 3: v3.2 — GNN Architecture (Mar 5)

| Commit | Change |
|--------|--------|
| `30a9b9f` | **v3.2**: continuous pipeline with GraphSAGE GNN + XGBoost ensemble. Sliding-window tri-partite graph (User <-> Transaction <-> Merchant). Triton serves GNN+XGBoost via Python backend. |
| `ce7c278` | Add pandas+pyarrow to backend (metrics reads parquet scores) |

**GPU utilization:** ~10%. GNN added real GPU work but chunk sizes still tiny.

---

## Phase 4: cuDF SIGSEGV Investigation (Mar 6)

The most painful phase. cuDF crashed with SIGSEGV when the parent process had any CUDA state.

| Commit | Change |
|--------|--------|
| `6ea0da5` | Fix: NFS mount was readOnly, file-queue rename needs write |
| `a500102`-`bd3475e` | Fix NFS permissions: RAPIDS uid=1001 vs root-owned NFS dirs |
| `8202de0` | GPU probe to catch SIGSEGV in subprocess, fall back to CPU |
| `d603c81`-`f71967d` | Faulthandler diagnostics: crash in `.to_cupy()` via numba_cuda |
| `99832f2` | **Key fix**: persistent GPU worker subprocess isolates CUDA from parent |
| `d3b7ed1` | Warm up CUDA context in worker before signalling ready |
| `1429521` | Pre-encode strings in pandas, only pass numeric cols to GPU |
| `dc2a4b7` | **Final fix**: route GPU->CPU via Arrow (`gdf.to_arrow().to_pandas()`) instead of numba_cuda |
| `d4ed4cd` | GPU worker owns full file lifecycle (no data through Queue) |

**Key lesson:** `gdf.to_pandas()` triggers numba_cuda internally and SIGSEGVs if the parent process ever touched CUDA. Solution: `gdf.to_arrow().to_pandas()` bypasses numba entirely.

**GPU utilization:** ~15%. Worker process stable now, but chunks still small.

---

## Phase 5: Liveness Probes + Triton Stability (Mar 7-8)

| Commit | Change |
|--------|--------|
| `aaaada8` | Increase CHUNK_SIZE 10K -> 50K for meaningful GPU speedup |
| `b3b6246` | TRITON_RETRIES=30 to prevent restart during slow init |
| `2813a4d`-`97739e4` | Liveness probes: heartbeat thread + /tmp/.healthy path |
| `dd8ea96` | Fix GPU warmup killing pods: heartbeat thread + Numba cache |
| `755c6e6` | Scoring: background heartbeat before `_connect_triton` |
| `7ddcf3f` | Stress mode: hot-reload gather rate via NFS config file |
| `5fda635`-`3b3092c` | Dashboard: pod startup modal, action modals for all operations |

**GPU utilization:** ~20%. Pods stopped dying, but chunk sizes still not saturating L40S SMs.

---

## Phase 6: v4 Architecture — Dedicated GPUs (Mar 10)

| Commit | Change |
|--------|--------|
| `e389952` | **Architecture v4**: single dedicated GPU per pipeline pod, continuous model training, 4 L40S GPUs across 2 nodes |
| `2f10a59` | Dashboard redesign: dual infra charts, 200ms WebSocket updates |
| `92c839d` | Per-category amount caps in synthetic data (realistic distributions) |
| `7682657`-`841a436` | Pin pods to nodes for insecure registry access |
| `031eef3` | Add NFS storage tile to dashboard |

**GPU utilization:** ~25%. Now on dedicated GPUs but still processing small files.

---

## Phase 7: GPU Saturation Push (Mar 11)

The major push to fill L40S SMs with real work.

| Commit | Change |
|--------|--------|
| `06d0928` | Multi-file batching + prefetch in data-prep |
| `7f40384` | **Full GPU pipeline**: `cudf.read_parquet` + GPU encoding, BATCH_FILES=16 |
| `e1d48d9` | Parallel NFS I/O + per-customer GPU features (sort+groupby+merge) |
| `74d3ab4` | **128-file batches** + per-category/merchant/rank features |
| `b74d9ee` | GPU-accelerated data-gather on 4th L40S |
| `11d27dc` | **CHUNK_SIZE 1M -> 10M** rows per file for substantial GPU work |
| `c81b49a` | 128 concurrent pipes: overlap NFS I/O with GPU compute |

**GPU utilization:** Saw first spikes to 99% during kernel execution, but heavy sawtooth pattern (100% during compute, 0% during NFS I/O). Average ~40%.

---

## Phase 8: Time-Slicing Experiment (Mar 11-12)

Tried GPU time-slicing (5 pods per GPU, 20 total) to fill idle cycles.

| Commit | Change |
|--------|--------|
| `149c00f` | GPU time-slicing: 20 pods on 4 L40S (5 per GPU) |
| `57c7bcd` | Disable cupy memory pool for shared GPU |
| `4a6e512` | Tune: 10M chunks, 12 gather pods, RMM_ALLOCATOR=cuda |
| `58a02c6` | Settle on 12 gather pods with 5M chunks, RMM managed memory |

**GPU utilization:** OOM issues with 10M-row files on shared GPU. Time-slicing caused memory contention. Backed off to 5M chunks but utilization didn't improve — the small-kernel problem got worse with contention.

**Decision:** Abandoned time-slicing. Dedicated GPU per pod with mega-batch is better for this workload.

---

## Phase 9: Mega-Batch + Pipelined I/O (Mar 12-13)

The breakthrough architecture.

| Commit | Change |
|--------|--------|
| `c2eae7e` | **Mega-batch data-prep**: concat 8x 5M-row files into 40M-row dataframe, process features ONCE |
| `3900257` | **Remove time-slicing**, dedicated GPU per pod, mega-batch |
| `8c41a3b` | **Pipeline NFS writes**: GPU processes batch N+1 while batch N writes to NFS |
| `f0989c3` | Parallel NFS writes: 32 pipes split output for concurrent I/O |
| `dcb45ae` | Dashboard: DCGM profiling metrics (`DCGM_FI_PROF_GR_ENGINE_ACTIVE` at 1s resolution) |
| `407a078` | Fix reset: re-queue raw data, don't delete |
| `3b5955d` | Dashboard: show volume processed (not generated) |

**GPU utilization:** 99% during mega-batch kernel execution (sort+groupby+merge on 40M rows fills L40S SMs). Sawtooth pattern remains (drops during NFS read/write) but peaks are 99% and average is 60-70%.

---

## Phase 10: Demo Polish (Mar 13-14)

| Commit | Change |
|--------|--------|
| `b2f255e` | Dashboard cleanup: remove stress mode, dead metrics |
| `9704116` | **Code quality**: remove stress mode infra, hardcoded tokens, fix logging |
| `38f277b` | Fix Start button stuck on "Starting..." |
| `9f7c6cd` | Parallel NFS reads (`cudf.read_parquet` in ThreadPoolExecutor), fix fraud rate reset, fix latency chart |
| `07b2516` | Fix: use `cudf.read_parquet` directly (pyarrow+from_arrow doubled conversion time) |
| `3eb898f` | **Scoring GPU**: switch from python:3.11-slim to RAPIDS base, batch 8 feature files with cuDF |
| `5c7b8c5` | TX/s KPI: use feature computation time only (~7M rows/sec vs ~2M full-batch) |

**GPU utilization (final):**
- **data-prep (node .44, GPU 0):** 99% peaks during 40M-row feature engineering, 60-70% average (sawtooth from NFS I/O gaps)
- **triton (node .44, GPU 1):** ~30% during inference bursts
- **model-train (node .40, GPU 0):** ~50% during GNN training epochs
- **scoring (node .40, GPU 1):** ~40% with GPU-accelerated batch reads

---

## GPU Utilization Journey Summary

| Phase | Date | Avg GPU % | What Changed |
|-------|------|-----------|-------------|
| v3.1 baseline | Mar 3 | ~0% | CSV processing, no real GPU work |
| K8s stable | Mar 4 | ~5% | 10K-row chunks, GPU idle between tiny kernels |
| GNN added | Mar 5 | ~10% | GraphSAGE model, but small data |
| cuDF stable | Mar 6 | ~15% | Fixed SIGSEGV, GPU worker subprocess |
| Probes + 50K | Mar 7-8 | ~20% | Pods stopped dying, larger chunks |
| v4 dedicated | Mar 10 | ~25% | 1 GPU per pod, no contention |
| 128-file batch | Mar 11 | ~40% | Sort+groupby on 1M+ rows, parallel I/O |
| Time-slicing | Mar 11-12 | ~35% | Worse: memory contention, smaller kernels |
| **Mega-batch** | **Mar 12-13** | **60-70%** | **40M rows in single kernel, pipelined writes** |
| Demo polish | Mar 13-14 | 60-70% | cuDF scoring, parallel reads, TX/s fix |

The key insight: GPU utilization scales with **kernel launch size**, not pod count. One 40M-row sort+groupby+merge fills L40S SMs better than twenty 1M-row operations.
