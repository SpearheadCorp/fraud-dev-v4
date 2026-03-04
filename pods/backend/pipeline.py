"""
Pipeline control: docker compose start/stop/reset/stress via subprocess.
"""
import os
import subprocess
import logging
from pathlib import Path

log = logging.getLogger(__name__)

COMPOSE_FILE = os.environ.get("COMPOSE_FILE", "/app/docker-compose.yaml")
PROJECT_NAME = os.environ.get("COMPOSE_PROJECT", "fraud-det-v31")
STRESS_CONFIG_PATH = Path(os.environ.get("STRESS_CONFIG_PATH", "/data/stress/stress.conf"))


def compose_cmd(*args: str) -> str:
    """Run a docker compose command, return stdout. Raises RuntimeError on failure."""
    cmd = ["docker", "compose", "-f", COMPOSE_FILE, "-p", PROJECT_NAME, *args]
    log.info("[INFO] compose: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"docker compose failed: {result.stderr.strip()}")
    return result.stdout


def compose_cmd_bg(*args: str) -> None:
    """Run a docker compose command in background (fire-and-forget). Does not raise."""
    cmd = ["docker", "compose", "-f", COMPOSE_FILE, "-p", PROJECT_NAME, *args]
    log.info("[INFO] compose (bg): %s", " ".join(cmd))
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_pipeline() -> dict:
    """Start the full ML pipeline (data-gather → data-prep → model-build → inference)."""
    try:
        compose_cmd_bg("--profile", "pipeline", "up", "--build", "--remove-orphans")
        return {"status": "started", "message": "Pipeline starting"}
    except Exception as exc:
        log.error("[ERROR] start_pipeline: %s", exc)
        return {"status": "error", "message": str(exc)}


def stop_pipeline() -> dict:
    """Stop pipeline services (not backend/prometheus)."""
    try:
        compose_cmd("stop", "data-gather", "data-prep", "model-build", "inference")
        return {"status": "stopped"}
    except Exception as exc:
        log.warning("[WARN] stop_pipeline: %s", exc)
        return {"status": "error", "message": str(exc)}


def reset_pipeline(raw_path: Path, features_path: Path) -> dict:
    """Stop pipeline and clear raw + features data (preserve models)."""
    try:
        compose_cmd("stop", "data-gather", "data-prep", "model-build", "inference")
    except Exception as exc:
        log.warning("[WARN] reset_pipeline stop: %s", exc)

    deleted: list = []
    for p in (raw_path, features_path):
        if p.exists():
            import shutil
            shutil.rmtree(str(p), ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)
            deleted.append(str(p))
            log.info("[INFO] Cleared %s", p)
    return {"status": "reset", "cleared": deleted}


def get_service_states() -> dict:
    """Get running status of each pipeline service."""
    try:
        output = compose_cmd("ps", "--format", "json")
        import json
        states: dict = {}
        # docker compose ps --format json outputs one JSON object per line
        for line in output.strip().splitlines():
            if not line.strip():
                continue
            try:
                svc = json.loads(line)
                name = svc.get("Service", svc.get("Name", "unknown"))
                status = svc.get("Status", svc.get("State", "unknown"))
                states[name] = status
            except Exception:
                pass
        return states
    except Exception as exc:
        log.warning("[WARN] get_service_states: %s", exc)
        return {}


def write_stress_config(stress_on: bool, num_workers: int = 32, chunk_size: int = 200000) -> None:
    """Write stress.conf for hot-reload by data-gather."""
    STRESS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"STRESS_MODE={'true' if stress_on else 'false'}\n"
        f"NUM_WORKERS={num_workers if stress_on else 8}\n"
        f"CHUNK_SIZE={chunk_size if stress_on else 100000}\n"
    )
    STRESS_CONFIG_PATH.write_text(content)
    log.info("[INFO] Wrote stress config: stress=%s", stress_on)
