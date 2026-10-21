"""
ml_worker.py — SkyGate ML Detection Worker (Decoupled Process)

Runs as an independent subprocess spawned by server.py at startup.
Communicates with the API server exclusively through two files:

  data/worker_results.json  — written by this worker after each cycle
  data/worker_cmd.json      — written by server.py to send commands

This decoupling means:
  • A TensorFlow crash or OOM in the ML pipeline cannot kill the web API.
  • The worker can be restarted independently without dropping any HTTP
    connections or SSE streams.
  • In future, the worker can be moved to a separate machine or container
    by replacing the file-based IPC with Redis pub/sub or a message queue.

Worker lifecycle
────────────────
  server.py  →  subprocess.Popen(['python', 'ml_worker.py'])
  ml_worker  →  every POLL_INTERVAL seconds:
                  1. Fetches live ADS-B data
                  2. Runs the full 3-layer pipeline
                  3. Writes results → data/worker_results.json
  server.py  →  reads worker_results.json on its own 30-second tick
                  and merges new anomalies into STATE

Commands (worker_cmd.json)
───────────────────────────
  {"cmd": "retrain"}   — triggers an immediate retraining run
  {"cmd": "stop"}      — graceful shutdown
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap path so imports resolve whether run directly or as subprocess ──
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE          = _SRC.parent
RESULTS_FILE   = _BASE / "data" / "worker_results.json"
CMD_FILE       = _BASE / "data" / "worker_cmd.json"
POLL_INTERVAL  = int(os.getenv("PIPELINE_INTERVAL_SECONDS", "30"))

# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown flag
# ─────────────────────────────────────────────────────────────────────────────
_running = True

def _handle_sigterm(signum, frame):
    global _running
    log.info("SIGTERM received — shutting down ML worker")
    _running = False

signal.signal(signal.SIGTERM, _handle_sigterm)
if hasattr(signal, "SIGINT"):
    signal.signal(signal.SIGINT, _handle_sigterm)


# ─────────────────────────────────────────────────────────────────────────────
# File-based IPC helpers
# ─────────────────────────────────────────────────────────────────────────────
def _write_results(anomalies: list, batch_count: int, latency_ms: int) -> None:
    """Atomically write detection results for the API server to consume."""
    payload = {
        "ts":           datetime.now(timezone.utc).isoformat(),
        "batch_count":  batch_count,
        "latency_ms":   latency_ms,
        "anomalies":    anomalies,
    }
    tmp = str(RESULTS_FILE) + ".tmp"
    try:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, RESULTS_FILE)   # atomic on POSIX; near-atomic on Windows
    except OSError as e:
        log.error("Failed to write results file: %s", e)


def _read_cmd() -> str | None:
    """Read and consume a pending command, returning the cmd string or None."""
    if not CMD_FILE.exists():
        return None
    try:
        with open(CMD_FILE) as f:
            data = json.load(f)
        CMD_FILE.unlink(missing_ok=True)   # consume the command
        return data.get("cmd")
    except (OSError, json.JSONDecodeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main detection loop
# ─────────────────────────────────────────────────────────────────────────────
def run() -> None:
    log.info("ML worker started — poll interval %ds", POLL_INTERVAL)

    # Lazy imports: keep TF/sklearn out of the server process memory space
    from data_fetcher import fetch_live_data
    from pipeline    import run_pipeline

    while _running:
        t0 = time.time()

        # ── Check for commands from the API server ────────────────────────
        cmd = _read_cmd()
        if cmd == "stop":
            log.info("Stop command received")
            break
        if cmd == "retrain":
            log.info("Retrain command received — running retraining cycle")
            try:
                from trainer import run_retraining
                run_retraining()
                log.info("Retraining complete")
            except Exception as exc:
                log.error("Retraining failed: %s", exc)

        # ── Fetch + detect ────────────────────────────────────────────────
        try:
            flights     = fetch_live_data()
            batch_count = len(flights) if flights is not None and not flights.empty else 0
            log.info("Fetched ADS-B batch", extra={"val": f"flights={batch_count}"})

            anomalies = []
            if flights is not None and not flights.empty:
                anomalies = run_pipeline(flights)

            latency_ms = int((time.time() - t0) * 1000)

            for a in anomalies:
                a.setdefault("timestamp", datetime.now(timezone.utc).strftime("%H:%M:%S"))
                log.warning(
                    "ANOMALY · %s [%s]",
                    a.get("callsign", "?"),
                    a.get("icao24",   "?"),
                    extra={"val": (
                        f"severity={a.get('severity','?')} "
                        f"layer={a.get('layer','?')} "
                        f"score={a.get('score',0):.3f}"
                    )},
                )

            if not anomalies:
                log.info("All layers nominal", extra={"val": "anomalies=0"})

            _write_results(anomalies, batch_count, latency_ms)

        except Exception as exc:
            log.error("Pipeline error: %s", exc, exc_info=True)
            _write_results([], 0, int((time.time() - t0) * 1000))

        # ── Sleep remainder of interval ───────────────────────────────────
        elapsed = time.time() - t0
        sleep_s = max(0.0, POLL_INTERVAL - elapsed)
        log.debug("Cycle done in %.1fs — sleeping %.1fs", elapsed, sleep_s)
        time.sleep(sleep_s)

    log.info("ML worker exited cleanly")


if __name__ == "__main__":
    run()
