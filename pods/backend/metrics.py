"""
Metrics collection (v4): telemetry (K8s pod logs), system (node-exporter), GPU (Prometheus/DCGM),
FlashBlade storage latency (REST API), queue depth, fraud scores.
"""
import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import urllib3
import pandas as pd
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from kubernetes import client, config
from kubernetes.client.rest import ApiException

log = logging.getLogger(__name__)

PROMETHEUS_URL    = os.environ.get("PROMETHEUS_URL",          "http://prometheus:9090")
MODEL_REPO        = Path(os.environ.get("MODEL_REPO_PATH",    "/data/models"))
RAW_PATH          = Path(os.environ.get("OUTPUT_PATH",         "/data/raw"))
FEATURES_PATH     = Path(os.environ.get("FEATURES_PATH",      "/data/features"))
SCORES_PATH       = Path(os.environ.get("SCORES_PATH",        "/data/scores"))
NAMESPACE         = os.environ.get("K8S_NAMESPACE",           "fraud-det-v31")
FLASHBLADE_FS_NAME = os.environ.get("FLASHBLADE_FS_NAME",    "financial-fraud-detection-demo")
FLASHBLADE_MGMT_IP = os.environ.get("FLASHBLADE_MGMT_IP",    "10.23.181.60")


def _load_gpu_window() -> int:
    """Read gpu_util_window_s from demo_config.json (ConfigMap-mounted at runtime).
    Falls back to 30s if the file is missing or the key is absent."""
    cfg_path = Path(__file__).parent / "static" / "demo_config.json"
    try:
        return int(json.loads(cfg_path.read_text()).get("gpu_util_window_s", 30))
    except Exception:
        return 30
FLASHBLADE_API_TOKEN = os.environ.get("FLASHBLADE_API_TOKEN", "")  # Set via K8s Secret or env var
NFS_NODE_INSTANCE  = os.environ.get("NFS_NODE_INSTANCE",      "10.23.181.44:9100")

_fb_session_token: Optional[str] = None


def _fb_login() -> Optional[str]:
    """Authenticate to FlashBlade REST API and return session token."""
    try:
        resp = requests.post(
            f"https://{FLASHBLADE_MGMT_IP}/api/login",
            headers={"api-token": FLASHBLADE_API_TOKEN},
            verify=False, timeout=5,
        )
        if resp.status_code == 200:
            token = resp.headers.get("X-Auth-Token")
            log.info("FlashBlade API login OK")
            return token
    except Exception as exc:
        log.debug("_fb_login: %s", exc)
    return None


def _core_v1() -> client.CoreV1Api:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CoreV1Api()


class PipelineState:
    """Mutable state shared between HTTP endpoints and the metrics collector."""

    def __init__(self) -> None:
        self.is_running: bool = False
        self.start_time: Optional[float] = None
        self.stop_time:  Optional[float] = None
        self.last_telemetry: dict = {}
        self.total_rows_processed: int = 0
        self.total_fraud_exposure_usd: float = 0.0
        self.total_fraud_flagged: int = 0
        self.total_amount_processed_usd: float = 0.0
        self._last_prep_chunk_id: int = -1
        self._processed_score_files: set[str] = set()

    def reset(self) -> None:
        self.is_running = False
        self.start_time = None
        self.stop_time  = None
        self.last_telemetry = {}
        self.total_rows_processed = 0
        self.total_fraud_exposure_usd = 0.0
        self.total_fraud_flagged = 0
        self.total_amount_processed_usd = 0.0
        self._last_prep_chunk_id = -1
        self._processed_score_files.clear()

    @property
    def elapsed_sec(self) -> int:
        if self.start_time is None:
            return 0
        end = self.stop_time if self.stop_time else time.time()
        return int(end - self.start_time)


_TELEMETRY_CACHE = MODEL_REPO / "last_telemetry.json"
_STATE_CACHE = MODEL_REPO / "pipeline_state.json"


class MetricsCollector:
    def __init__(self, state: PipelineState) -> None:
        self.state = state
        self._fraud_cache: dict = {}
        self._fraud_cache_at: float = 0.0
        self._fraud_cache_lock = threading.Lock()
        self._FRAUD_CACHE_TTL = 5.0  # seconds — force recompute even if same files visible
        self._gpu_window: int = _load_gpu_window()
        self._load_telemetry_cache()

    def _load_telemetry_cache(self) -> None:
        try:
            if _TELEMETRY_CACHE.exists():
                self.state.last_telemetry = json.loads(_TELEMETRY_CACHE.read_text())
                log.info("Loaded telemetry cache (%d stages)", len(self.state.last_telemetry))
            
            if _STATE_CACHE.exists():
                state_data = json.loads(_STATE_CACHE.read_text())
                self.state.total_rows_processed = state_data.get("total_rows", 0)
                self.state.total_fraud_exposure_usd = state_data.get("total_fraud_exposure", 0.0)
                self.state.total_fraud_flagged = state_data.get("total_fraud_flagged", 0)
                self.state.total_amount_processed_usd = state_data.get("total_amount_processed", 0.0)
                self.state._last_prep_chunk_id = state_data.get("last_chunk_id", -1)
                # Mark all existing score files as already processed so restart doesn't double-count
                if SCORES_PATH.exists():
                    self.state._processed_score_files = {
                        f.name for f in SCORES_PATH.glob("*.parquet")
                        if not f.name.endswith(".processing")
                    }
                log.info("Loaded state cache (total_rows=%d, fraud_exposure=$%.2f, skipping %d existing score files)",
                         self.state.total_rows_processed, self.state.total_fraud_exposure_usd,
                         len(self.state._processed_score_files))
            
        except Exception as exc:
            log.debug("load_telemetry_cache: %s", exc)

    def _save_telemetry_cache(self) -> None:
        try:
            _TELEMETRY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _TELEMETRY_CACHE.write_text(json.dumps(self.state.last_telemetry))
            
            state_data = {
                "total_rows": self.state.total_rows_processed,
                "total_fraud_exposure": self.state.total_fraud_exposure_usd,
                "total_fraud_flagged": self.state.total_fraud_flagged,
                "total_amount_processed": self.state.total_amount_processed_usd,
                "last_chunk_id": self.state._last_prep_chunk_id,
            }
            _STATE_CACHE.write_text(json.dumps(state_data))
            
        except Exception as exc:
            log.debug("save_telemetry_cache: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self) -> dict:
        # Skip pod-log telemetry when pipeline is stopped — stale log lines
        # would otherwise repopulate KPIs from the last run's data.
        telemetry  = self._parse_telemetry() if self.state.is_running else {}
        if self.state.is_running:
            self._update_fraud_metrics()
        system     = self._collect_system()
        gpu        = self._collect_gpu()        if self.state.is_running else self._gpu_zeros()
        business   = self._compute_kpis(telemetry)
        if not self.state.is_running:
            business['prep_rows_per_sec'] = 0
        storage    = self._collect_storage()
        flashblade = self._collect_flashblade()
        queue      = self._collect_queue_depth()
        fraud      = self._collect_fraud_metrics() if self.state.is_running else {}

        if telemetry:
            self.state.last_telemetry = telemetry
            self._save_telemetry_cache()

        return {
            "is_running":  self.state.is_running,
            "elapsed_sec": self.state.elapsed_sec,
            "timestamp":   time.time(),
            "pipeline": {
                "gather":  telemetry.get("gather",  {}),
                "prep":    telemetry.get("prep",    {}),
                "scoring": telemetry.get("scoring", {}),
                "train":   telemetry.get("train",   {}),
            },
            "queue":     queue,
            "fraud":     fraud,
            "business":  business,
            "storage":   storage,
            "flashblade": flashblade,
            "system":    system,
            "gpu":       gpu,
            "gpu_rt":    self._collect_gpu_rt() if self.state.is_running else self._gpu_zeros(),
            "gpu_window_s": self._gpu_window,
        }

    def collect_fast(self) -> dict:
        """Light-weight collection: GPU, CPU, FlashBlade latency only.
        Called at 200ms intervals between full collect() cycles.
        Also includes fraud metrics via TTL cache so category/geo data
        refreshes every 5s independent of the slow full collect() cycle."""
        system     = self._collect_system()
        gpu        = self._collect_gpu()        if self.state.is_running else self._gpu_zeros()
        flashblade = self._collect_flashblade()
        fraud      = self._collect_fraud_metrics() if self.state.is_running else {}
        return {
            "is_running":  self.state.is_running,
            "elapsed_sec": self.state.elapsed_sec,
            "timestamp":   time.time(),
            "system":    system,
            "gpu":       gpu,
            "gpu_rt":    self._collect_gpu_rt() if self.state.is_running else self._gpu_zeros(),
            "gpu_window_s": self._gpu_window,
            "flashblade": flashblade,
            "fraud":     fraud,
        }

    # ------------------------------------------------------------------
    # Fraud metrics from scored Parquet files
    # ------------------------------------------------------------------

    def _update_fraud_metrics(self) -> None:
        """Scan SCORES_PATH for new parquet files and accumulate fraud totals."""
        try:
            if not SCORES_PATH.exists():
                return

            # Find all .parquet files that aren't currently being written (.processing)
            all_files = sorted(f.name for f in SCORES_PATH.glob("*.parquet")
                              if not f.name.endswith(".processing"))
            
            # Identify files we haven't processed yet in this session
            new_files = [f for f in all_files if f not in self.state._processed_score_files]
            if not new_files:
                return

            log.info("Processing %d new score files for KPIs", len(new_files))
            total_new_fraud_amt = 0.0
            total_new_flagged = 0
            total_new_txn_amt = 0.0

            for fname in new_files:
                fpath = SCORES_PATH / fname
                try:
                    df = pd.read_parquet(str(fpath), columns=["amt", "fraud_score"])
                    fraud_mask = df["fraud_score"] > 0.5
                    total_new_fraud_amt += df.loc[fraud_mask, "amt"].sum()
                    total_new_flagged += int(fraud_mask.sum())
                    total_new_txn_amt += df["amt"].sum()
                    self.state._processed_score_files.add(fname)
                except Exception as e:
                    log.debug("Error reading %s: %s", fname, e)

            self.state.total_fraud_exposure_usd += total_new_fraud_amt
            self.state.total_fraud_flagged += total_new_flagged
            self.state.total_amount_processed_usd += total_new_txn_amt
            
        except Exception as exc:
            log.debug("_update_fraud_metrics: %s", exc)

    # ------------------------------------------------------------------
    # Telemetry from K8s pod logs
    # ------------------------------------------------------------------

    def _get_deployment_pod_logs(self, dep_name: str, tail: int = 200) -> str:
        """Get logs from the most-recent running pod of a Deployment."""
        try:
            core_v1 = _core_v1()
            pods = core_v1.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=f"app={dep_name}",
            )
            # Prefer Running pods; fall back to any latest pod
            running = [p for p in pods.items
                       if p.status and p.status.phase == "Running"]
            pod_list = running if running else pods.items
            if not pod_list:
                return ""
            pod_name = pod_list[-1].metadata.name
            return core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=NAMESPACE, tail_lines=tail,
            )
        except ApiException:
            return ""
        except Exception as exc:
            log.debug("_get_deployment_pod_logs %s: %s", dep_name, exc)
            return ""

    def _get_job_pod_logs(self, job_name: str, tail: int = 200) -> str:
        """Get logs from the pod created by a Job (e.g. model-build)."""
        try:
            core_v1 = _core_v1()
            pods = core_v1.list_namespaced_pod(
                namespace=NAMESPACE,
                label_selector=f"job-name={job_name}",
            )
            if not pods.items:
                return ""
            pod_name = pods.items[-1].metadata.name
            return core_v1.read_namespaced_pod_log(
                name=pod_name, namespace=NAMESPACE, tail_lines=tail,
            )
        except ApiException:
            return ""
        except Exception as exc:
            log.debug("_get_job_pod_logs %s: %s", job_name, exc)
            return ""

    @staticmethod
    def _parse_lines(logs: str) -> dict:
        """Parse last TELEMETRY line per stage from log text."""
        result: dict = {}
        for line in logs.splitlines():
            if "[TELEMETRY]" not in line:
                continue
            try:
                parts = line.split("[TELEMETRY]")[1].strip().split()
                kv: dict = {}
                for part in parts:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        v_clean = v.rstrip("x")
                        try:
                            kv[k] = float(v_clean)
                        except ValueError:
                            kv[k] = v
                stage = kv.pop("stage", "unknown")
                result[stage] = kv
            except Exception as exc:
                log.debug("telemetry parse: %s", exc)
        return result

    def _parse_telemetry(self) -> dict:
        result: dict = {}

        # Continuous Deployment pods
        for dep in ("data-gather", "data-prep", "scoring", "model-train"):
            logs = self._get_deployment_pod_logs(dep)
            result.update(self._parse_lines(logs))

        # Fill missing stages from cache so pod restarts don't zero out live gauges
        for stage, cached in self.state.last_telemetry.items():
            if stage not in result:
                result[stage] = cached

        return result

    # ------------------------------------------------------------------
    # Queue depth (file-queue on NFS)
    # ------------------------------------------------------------------

    def _collect_queue_depth(self) -> dict:
        depths: dict = {}
        for name, path in [
            ("raw",      RAW_PATH),
            ("features", FEATURES_PATH),
        ]:
            try:
                depths[name] = {
                    "pending":    len(list(path.glob("*.parquet")))      if path.exists() else 0,
                    "processing": len(list(path.glob("*.parquet.processing"))) if path.exists() else 0,
                    "done":       len(list(path.glob("*.parquet.done"))) if path.exists() else 0,
                }
            except Exception as exc:
                log.debug("queue_depth %s: %s", name, exc)
                depths[name] = {"pending": 0, "processing": 0, "done": 0}
        return depths

    # ------------------------------------------------------------------
    # Fraud scores from GPU scoring output
    # ------------------------------------------------------------------

    _FRAUD_COLS = ["fraud_score", "amt", "category", "state", "scored_at", "merchant", "trans_num"]

    def _collect_fraud_metrics(self) -> dict:
        try:
            score_files = sorted(SCORES_PATH.glob("*.parquet"), key=lambda p: p.stat().st_mtime)[-10:] \
                          if SCORES_PATH.exists() else []
            if not score_files:
                return {}
            with self._fraud_cache_lock:
                if self._fraud_cache and (time.time() - self._fraud_cache_at) < self._FRAUD_CACHE_TTL:
                    return self._fraud_cache
            df = pd.concat(
                [pd.read_parquet(str(f), columns=self._FRAUD_COLS) for f in score_files],
                ignore_index=True,
            )
            if "fraud_score" not in df.columns:
                return {}
            alerts = (df[df["fraud_score"] > 0.8]
                      .sort_values("scored_at", ascending=False)
                      .drop_duplicates(subset=["merchant"])
                      .head(20)
                      .sort_values("fraud_score", ascending=False))
            alert_cols = [c for c in ("trans_num", "merchant", "amt", "category", "fraud_score")
                          if c in alerts.columns]

            # Real fraud-by-category: sum of amt for rows flagged as high-risk (> 0.5)
            fraud_by_category = {}
            if "category" in df.columns and "amt" in df.columns:
                flagged = df[df["fraud_score"] > 0.5]
                fraud_by_category = (
                    flagged.groupby("category")["amt"].sum()
                    .round(2)
                    .to_dict()
                )

            # Fraud-by-geography: sum of fraud amt per state
            fraud_by_geography = {}
            if "state" in df.columns and "amt" in df.columns:
                flagged_geo = df[df["fraud_score"] > 0.5]
                fraud_by_geography = (
                    flagged_geo.groupby("state")["amt"].sum()
                    .round(2)
                    .sort_values(ascending=False)
                    .to_dict()
                )

            result = {
                "fraud_rate_pct":     float((df["fraud_score"] > 0.5).mean() * 100),
                "total_scored":       len(df),
                "recent_alerts":      alerts[alert_cols].to_dict("records"),
                "fraud_by_category":  fraud_by_category,
                "fraud_by_geography": fraud_by_geography,
            }
            with self._fraud_cache_lock:
                self._fraud_cache = result
                self._fraud_cache_at = time.time()
            return result
        except Exception as exc:
            log.debug("_collect_fraud_metrics: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # System metrics (psutil)
    # ------------------------------------------------------------------

    def _collect_system(self) -> dict:
        N29_INSTANCE = NFS_NODE_INSTANCE           # 10.23.181.44:9100 (worker .44 / n29)
        N25_INSTANCE = os.environ.get("N25_NODE_INSTANCE", "10.23.181.40:9100")  # worker .40 / n25
        try:
            # CPU % for both worker nodes via node-exporter
            cpu_n29 = 0.0
            cpu_n25 = 0.0
            for inst, label in [(N29_INSTANCE, "n29"), (N25_INSTANCE, "n25")]:
                query = ('100 - (avg(rate(node_cpu_seconds_total'
                         f'{{instance="{inst}",mode="idle"}}[1m]))*100)')
                resp = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query}, timeout=3,
                )
                resp.raise_for_status()
                result = resp.json().get("data", {}).get("result", [])
                val = round(float(result[0]["value"][1]), 1) if result else 0.0
                if label == "n29":
                    cpu_n29 = val
                else:
                    cpu_n25 = val

            # RAM from node-exporter on .44
            mem_total_resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": f'node_memory_MemTotal_bytes{{instance="{N29_INSTANCE}"}}'},
                timeout=3,
            )
            mem_avail_resp = requests.get(
                f"{PROMETHEUS_URL}/api/v1/query",
                params={"query": f'node_memory_MemAvailable_bytes{{instance="{N29_INSTANCE}"}}'},
                timeout=3,
            )
            mem_total_res = mem_total_resp.json().get("data", {}).get("result", [])
            mem_avail_res = mem_avail_resp.json().get("data", {}).get("result", [])
            mem_total = float(mem_total_res[0]["value"][1]) if mem_total_res else 0.0
            mem_avail = float(mem_avail_res[0]["value"][1]) if mem_avail_res else 0.0
            mem_used  = mem_total - mem_avail
            ram_pct   = round(mem_used / mem_total * 100, 1) if mem_total else 0.0

            return {
                "cpu_percent":      cpu_n29,
                "cpu_percent_n29":  cpu_n29,
                "cpu_percent_n25":  cpu_n25,
                "ram_percent":  ram_pct,
                "ram_used_gb":  round(mem_used / 1e9, 2),
                "ram_total_gb": round(mem_total / 1e9, 2),
            }
        except Exception as exc:
            log.debug("_collect_system: %s", exc)
            return {"cpu_percent": 0.0, "cpu_percent_n29": 0.0, "cpu_percent_n25": 0.0,
                    "ram_percent": 0.0, "ram_used_gb": 0.0, "ram_total_gb": 0.0}

    # ------------------------------------------------------------------
    # GPU metrics via Prometheus / DCGM
    # ------------------------------------------------------------------

    def _collect_gpu(self) -> dict:
        """Query DCGM GPU metrics for all GPU nodes (both .44 and .40).

        Uses max_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE[Xs]) so tiles show
        the peak GPU activity over the last X seconds (configurable via
        gpu_util_window_s in demo_config.json, default 30s) rather than an
        instant snapshot that reads 0% between mega-batch kernel bursts.

        Falls back to DCGM_FI_DEV_GPU_UTIL if the profiling metric is not
        exported by this cluster's DCGM configuration.
        """
        try:
            metrics: dict = {}
            window = f"{self._gpu_window}s"
            for metric_name, key_prefix, scale in [
                (f"max_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE[{window}])", "gpu_{host}_{gpu}_util_pct", 100.0),
                (f"max_over_time(DCGM_FI_DEV_GPU_UTIL[{window}])",          "gpu_{host}_{gpu}_util_pct", 1.0),
                (f"max_over_time(DCGM_FI_DEV_MEM_COPY_UTIL[{window}])",     "gpu_{host}_{gpu}_mem_pct",  1.0),
            ]:
                # Skip DCGM_FI_DEV_GPU_UTIL fallback if profiling metric already populated util keys
                if "DCGM_FI_DEV_GPU_UTIL" in metric_name:
                    if any(k.endswith("_util_pct") for k in metrics):
                        continue
                    log.warning("_collect_gpu: profiling metric returned no results, falling back to DCGM_FI_DEV_GPU_UTIL (window=%s)", window)

                resp = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": metric_name}, timeout=3,
                )
                resp.raise_for_status()
                for result in resp.json().get("data", {}).get("result", []):
                    gpu_id = result.get("metric", {}).get("gpu", "0")
                    hostname = result.get("metric", {}).get("Hostname", "unknown")
                    # Shorten hostname: slc6-lg-n3-b30-29 -> n29, slc6-lg-n3-b30-25 -> n25
                    short = "n" + hostname.split("-")[-1] if "-" in hostname else hostname
                    key = key_prefix.format(host=short, gpu=gpu_id)
                    metrics[key] = float(result["value"][1]) * scale
            if not metrics:
                log.warning("_collect_gpu: all queries returned empty (window=%s), returning zeros", window)
                return self._gpu_zeros()
            return metrics
        except Exception as exc:
            log.warning("_collect_gpu failed: %s", exc)
            return self._gpu_zeros()

    def _collect_gpu_rt(self) -> dict:
        """1s max_over_time GPU util for chart — catches brief GPU bursts that
        instant queries miss, while staying near real-time."""
        try:
            metrics: dict = {}
            for metric_name, key_prefix, scale in [
                ("max_over_time(DCGM_FI_PROF_GR_ENGINE_ACTIVE[1s])", "gpu_{host}_{gpu}_util_pct", 100.0),
                ("max_over_time(DCGM_FI_DEV_GPU_UTIL[1s])",          "gpu_{host}_{gpu}_util_pct", 1.0),
                ("max_over_time(DCGM_FI_DEV_MEM_COPY_UTIL[1s])",     "gpu_{host}_{gpu}_mem_pct",  1.0),
            ]:
                if "DCGM_FI_DEV_GPU_UTIL" in metric_name:
                    if any(k.endswith("_util_pct") for k in metrics):
                        continue
                resp = requests.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": metric_name}, timeout=3,
                )
                resp.raise_for_status()
                for result in resp.json().get("data", {}).get("result", []):
                    gpu_id = result.get("metric", {}).get("gpu", "0")
                    hostname = result.get("metric", {}).get("Hostname", "unknown")
                    short = "n" + hostname.split("-")[-1] if "-" in hostname else hostname
                    key = key_prefix.format(host=short, gpu=gpu_id)
                    metrics[key] = float(result["value"][1]) * scale
            return metrics if metrics else self._gpu_zeros()
        except Exception as exc:
            log.warning("_collect_gpu_rt failed: %s", exc)
            return self._gpu_zeros()

    @staticmethod
    def _gpu_zeros() -> dict:
        return {
            "gpu_n29_0_util_pct": 0.0, "gpu_n29_0_mem_pct": 0.0,
            "gpu_n29_1_util_pct": 0.0, "gpu_n29_1_mem_pct": 0.0,
            "gpu_n25_0_util_pct": 0.0, "gpu_n25_0_mem_pct": 0.0,
            "gpu_n25_1_util_pct": 0.0, "gpu_n25_1_mem_pct": 0.0,
        }

    # ------------------------------------------------------------------
    # FlashBlade latency via REST API (file-systems/performance)
    # ------------------------------------------------------------------

    def _collect_flashblade(self) -> dict:
        global _fb_session_token
        try:
            if not _fb_session_token:
                _fb_session_token = _fb_login()
            if not _fb_session_token:
                return {"read_latency_ms": 0.0, "write_latency_ms": 0.0, "avg_latency_ms": 0.0}

            resp = requests.get(
                f"https://{FLASHBLADE_MGMT_IP}/api/2.24/file-systems/performance",
                headers={"X-Auth-Token": _fb_session_token},
                params={"names": FLASHBLADE_FS_NAME},
                verify=False, timeout=5,
            )
            if resp.status_code == 401 or resp.status_code == 403:
                # Token expired — re-login once
                _fb_session_token = _fb_login()
                if not _fb_session_token:
                    return {"read_latency_ms": 0.0, "write_latency_ms": 0.0, "avg_latency_ms": 0.0}
                resp = requests.get(
                    f"https://{FLASHBLADE_MGMT_IP}/api/2.24/file-systems/performance",
                    headers={"X-Auth-Token": _fb_session_token},
                    params={"names": FLASHBLADE_FS_NAME},
                    verify=False, timeout=5,
                )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                return {"read_latency_ms": 0.0, "write_latency_ms": 0.0, "avg_latency_ms": 0.0}
            item = items[0]
            read_ms  = item.get("usec_per_read_op",  0.0) / 1000
            write_ms = item.get("usec_per_write_op", 0.0) / 1000
            return {
                "read_latency_ms":  round(read_ms,  3),
                "write_latency_ms": round(write_ms, 3),
                "avg_latency_ms":   round((read_ms + write_ms) / 2, 3),
            }
        except Exception as exc:
            log.debug("_collect_flashblade: %s", exc)
            return {"read_latency_ms": 0.0, "write_latency_ms": 0.0, "avg_latency_ms": 0.0}

    # ------------------------------------------------------------------
    # Business KPIs
    # ------------------------------------------------------------------

    def _compute_kpis(self, telemetry: dict) -> dict:
        # After reset (no rows processed yet), return zeros
        prep       = telemetry.get("prep",    self.state.last_telemetry.get("prep", {}))
        scoring    = telemetry.get("scoring", self.state.last_telemetry.get("scoring", {}))
        # Accumulate rows processed across batches (each telemetry has a chunk_id)
        chunk_id = int(prep.get("chunk_id", -1))
        batch_rows = int(prep.get("rows", 0))
        if chunk_id >= 0 and chunk_id != self.state._last_prep_chunk_id:
            self.state.total_rows_processed += batch_rows
            self.state._last_prep_chunk_id = chunk_id
        total_txns = self.state.total_rows_processed
        if total_txns == 0:
            return {
                "total_transactions": 0, "prep_rows_per_sec": 0,
                "fraud_flagged": 0, "fraud_rate_pct": 0.0, "fraud_exposure_usd": 0.0,
            }
        
        # TPS = rows scored in last batch / time that batch took (stable, no decay between batches)
        scoring_rows = int(scoring.get("rows", 0))
        scoring_latency_ms = float(scoring.get("latency_ms", 0))
        if scoring_rows > 0 and scoring_latency_ms > 0:
            prep_rps = int(scoring_rows / (scoring_latency_ms / 1000))
        else:
            prep_rps = 0
        
        # Use real cumulative metrics from scoring files
        fraud_flagged = self.state.total_fraud_flagged
        fraud_exposure = self.state.total_fraud_exposure_usd
        
        # Calculate real-time fraud rate from last chunk or overall average
        last_fraud_rate = float(scoring.get("fraud_rate", 0.0))
        if last_fraud_rate > 0:
            display_rate = last_fraud_rate
        else:
            # Fallback to cumulative average if no recent telemetry
            display_rate = (fraud_flagged / total_txns) if total_txns > 0 else 0.0

        decision_latency_ms = (
            scoring_latency_ms / scoring_rows
            if scoring_rows > 0 and scoring_latency_ms > 0 else 0.0
        )

        return {
            "total_transactions": total_txns,
            "prep_rows_per_sec": prep_rps,
            "fraud_flagged": fraud_flagged,
            "fraud_rate_pct": round(display_rate * 100, 2),
            "fraud_exposure_usd": round(fraud_exposure, 0),
            "total_amount_processed_usd": round(self.state.total_amount_processed_usd, 0),
            "decision_latency_ms": round(decision_latency_ms, 3),
        }

    # ------------------------------------------------------------------
    # Storage stats
    # ------------------------------------------------------------------

    def _collect_storage(self) -> dict:
        try:
            def _dir_stats(p: Path):
                files = list(p.glob("*.parquet")) if p.exists() else []
                size = sum(f.stat().st_size for f in files) / 1e9
                return len(files), round(size, 2)

            raw_n, raw_gb       = _dir_stats(RAW_PATH)
            feat_n, feat_gb     = _dir_stats(FEATURES_PATH)
            score_n, score_gb   = _dir_stats(SCORES_PATH)
            models_ready = (MODEL_REPO / "fraud_gnn_gpu" / "1" / "state_dict_gnn.pth").exists()
            total_gb = round(raw_gb + feat_gb + score_gb, 2)
            return {
                "raw_files":      raw_n,
                "raw_size_gb":    raw_gb,
                "features_files": feat_n,
                "feat_size_gb":   feat_gb,
                "score_files":    score_n,
                "score_size_gb":  score_gb,
                "total_size_gb":  total_gb,
                "models_ready":   models_ready,
            }
        except Exception as exc:
            log.debug("_collect_storage: %s", exc)
            return {"raw_files": 0, "raw_size_gb": 0.0, "features_files": 0, "feat_size_gb": 0.0,
                    "score_files": 0, "score_size_gb": 0.0, "total_size_gb": 0.0, "models_ready": False}


def load_shap_summary() -> Optional[dict]:
    shap_path = MODEL_REPO / "shap_summary.json"
    if not shap_path.exists():
        return None
    try:
        return json.loads(shap_path.read_text())
    except Exception as exc:
        log.warning("load_shap_summary: %s", exc)
        return None


def load_training_metrics() -> Optional[dict]:
    metrics_path = MODEL_REPO / "training_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except Exception as exc:
        log.warning("load_training_metrics: %s", exc)
        return None
