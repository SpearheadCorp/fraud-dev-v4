"""
Pod: data-prep (v4)
Continuous file-queue worker. Atomically claims raw parquet chunks from INPUT_PATH,
engineers 21 features (GPU via cuDF), writes to OUTPUT_PATH.
Multiple replicas race-safely share the queue via POSIX rename atomicity.

GPU worker owns the full file lifecycle: reads raw file, does GPU feature engineering,
writes output to NFS, marks input done. Queue carries only path strings + timing dicts.
"""
import os
import sys
import time
import logging
import signal
import threading
import multiprocessing as mp
import queue as _queue_module
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

_SHUTDOWN = False


def _handle_signal(signum, frame):
    global _SHUTDOWN
    log.info("[INFO] Signal %s received — shutting down", signum)
    _SHUTDOWN = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/data/raw"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/data/features"))

# ---------------------------------------------------------------------------
# Persistent GPU worker subprocess
# The main process NEVER imports cudf/cupy — any CUDA in the parent
# corrupts numba_cuda's context for subsequent use. All GPU work runs in
# a long-lived child (fork mode = fresh CUDA state, no __main__ reimport).
# ---------------------------------------------------------------------------
_gpu_worker_proc: "mp.Process | None" = None
_gpu_req_q: "mp.Queue | None" = None
_gpu_res_q: "mp.Queue | None" = None
GPU_AVAILABLE = False


def _start_gpu_worker() -> bool:
    """Start persistent GPU worker. Returns True when worker signals ready.

    Uses fork context (not spawn): parent never imports cudf/CUDA so fork is
    safe, and fork avoids reimporting __main__ which would cause recursive
    _start_gpu_worker() calls inside the worker process.
    """
    global _gpu_worker_proc, _gpu_req_q, _gpu_res_q
    try:
        import gpu_worker as _gw  # safe: cudf imported inside run_gpu_loop, not at module level
        ctx = mp.get_context("fork")  # fork: no __main__ reimport, clean CUDA state (parent has none)
        _gpu_req_q = ctx.Queue()
        _gpu_res_q = ctx.Queue()
        _gpu_worker_proc = ctx.Process(
            target=_gw.run_gpu_loop,
            args=(_gpu_req_q, _gpu_res_q),
            daemon=True,
        )
        _gpu_worker_proc.start()
        msg = _gpu_res_q.get(timeout=600)  # wait for cudf + libcudf init + warmup (Numba JIT cold start can exceed 2 min)
        return msg == "ready"
    except Exception as exc:
        log.warning("[WARN] GPU worker startup failed: %s", exc)
        return False


# Liveness heartbeat thread — started before GPU worker so the probe never
# kills the pod during Numba JIT cold-start (can exceed 4 min on first run).
# Main loop also touches /tmp/.healthy every iteration for tight steady-state detection.
def _liveness_heartbeat():
    while True:
        try:
            Path("/tmp/.healthy").touch()
        except OSError:
            pass
        time.sleep(10)

threading.Thread(target=_liveness_heartbeat, daemon=True, name="liveness").start()

if _start_gpu_worker():
    GPU_AVAILABLE = True
    log.info("[INFO] GPU worker ready — GPU path enabled")
else:
    log.error("[ERROR] GPU worker failed to start — pod is GPU-only, exiting for K8s restart")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Category / state maps
# ---------------------------------------------------------------------------
ALL_CATEGORIES = [
    "misc_net", "grocery_pos", "entertainment", "gas_transport", "misc_pos",
    "grocery_net", "shopping_net", "shopping_pos", "food_dining", "personal_care",
    "health_fitness", "travel", "kids_pets", "home",
]
CATEGORY_MAP = {cat: idx for idx, cat in enumerate(ALL_CATEGORIES)}

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]
STATE_MAP = {s: idx for idx, s in enumerate(US_STATES)}

FEATURE_COLS = [
    "amt_log", "amt_scaled", "hour_of_day", "day_of_week", "is_weekend",
    "is_night", "distance_km", "category_encoded", "state_encoded",
    "gender_encoded", "city_pop_log", "zip_region", "amt", "lat", "long",
    "city_pop", "unix_time", "merch_lat", "merch_long", "merch_zipcode", "zip",
    "is_fraud",
]

_REQUIRED_COLS = ["amt", "lat", "long", "merch_lat", "merch_long", "unix_time", "is_fraud"]
_PASSTHROUGH_COLS = ["cc_num", "merchant", "trans_num", "category", "chunk_ts"]


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def emit_telemetry(chunk_id: int, rows: int, gpu_time: float) -> None:
    sys.stdout.write(
        f"[TELEMETRY] stage=prep chunk_id={chunk_id} rows={rows} "
        f"gpu_time_s={gpu_time:.3f} gpu_used=1\n"
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main — continuous file-queue loop
# ---------------------------------------------------------------------------

# Pod-unique prefix so multiple replicas don't overwrite each other's output files.
_POD_PREFIX = os.environ.get("HOSTNAME", str(os.getpid()))


def main() -> None:
    INPUT_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    log.info("[INFO] data-prep started: INPUT=%s OUTPUT=%s gpu=%s pod=%s",
             INPUT_PATH, OUTPUT_PATH, GPU_AVAILABLE, _POD_PREFIX)

    chunk_id = 0
    while not _SHUTDOWN:
        Path("/tmp/.healthy").touch()  # liveness heartbeat
        # --- Claim next available chunk via atomic rename ---
        files = sorted(f for f in INPUT_PATH.glob("*.parquet")
                       if not f.name.endswith((".processing", ".done")))
        claimed: Path | None = None
        for f in files:
            proc = Path(str(f) + ".processing")
            try:
                f.rename(proc)
                claimed = proc
                break
            except (FileNotFoundError, OSError):
                continue  # another worker claimed it first

        if claimed is None:
            time.sleep(0.5)
            continue

        # --- Validate chunk (quick read — separate from timed CPU reference path) ---
        try:
            df_check = pd.read_parquet(str(claimed))
        except Exception as exc:
            log.warning("[WARN] Failed to read %s: %s — skipping", claimed.name, exc)
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue

        if len(df_check) == 0:
            claimed.rename(str(claimed).replace(".processing", ".done"))
            continue
        del df_check

        # --- Derive output paths from source filename (preserves chunk identity) ---
        # claimed = "raw_chunk_000042.parquet.processing" → raw_stem = "raw_chunk_000042"
        raw_stem = claimed.name[:-len(".parquet.processing")]
        out_file = OUTPUT_PATH / f"features_{raw_stem}.parquet"
        tmp_file = out_file.with_suffix(".parquet.tmp")

        # --- GPU worker path: send paths, collect result ---
        _gpu_req_q.put((str(claimed), str(out_file), str(tmp_file)))
        try:
            status, n_rows, gpu_timing = _gpu_res_q.get(timeout=600)
        except _queue_module.Empty:
            log.error("[ERROR] GPU worker timeout — exiting for K8s restart")
            sys.exit(1)

        if status != "ok":
            log.error("[ERROR] GPU worker error: %s — exiting for K8s restart", n_rows)
            sys.exit(1)

        # GPU worker handled: atomic write (tmp→rename), claimed.rename(.done).

        gpu_time = gpu_timing.get("total", 0.0)
        emit_telemetry(chunk_id=chunk_id, rows=n_rows, gpu_time=gpu_time)
        log.info("[INFO] chunk %06d: %d rows gpu_time=%.3fs",
                 chunk_id, n_rows, gpu_time)
        chunk_id += 1

    log.info("[INFO] data-prep shutdown complete after %d chunks", chunk_id)


if __name__ == "__main__":
    main()
