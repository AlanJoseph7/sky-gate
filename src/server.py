"""
server.py — SkyGate FastAPI Web Server
 
Serves the frontend dashboard and exposes REST + SSE endpoints.
 
Run:
    uvicorn server:app --host 0.0.0.0 --port 5000 --reload
 
Or directly:
    python server.py
 
Dependencies (add to requirements.txt):
    fastapi
    uvicorn[standard]
    sse-starlette
    httpx
"""
 
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import asyncio
import json
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure src directory is in sys.path
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Windows asyncio WinError 10054 patch ─────────────────────────────────────
# Python 3.8-3.11 ProactorEventLoop on Windows throws ConnectionResetError when
# calling shutdown() on a socket that was already reset by the client (e.g. browser
# refresh or closed SSE connection). We safely catch this to keep logs clean.
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_conn_lost = _ProactorBasePipeTransport._call_connection_lost

        def _safe_call_connection_lost(self, exc):
            try:
                _orig_call_conn_lost(self, exc)
            except (ConnectionResetError, OSError):
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                self._sock = None
                self._called_connection_lost = True

        _ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost
    except Exception:
        pass

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── SkyGate modules ───────────────────────────────────────────────────────────
from config import cfg
from flight_lookup import lookup_flight
from auth import verify_password, get_password_hash, create_access_token, decode_access_token
from models import User, init_db, get_db
from logger import get_logger

import re
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

log = get_logger(__name__)
 
# ─────────────────────────────────────────────────────────────────────────────
# Shared in-memory state
# ─────────────────────────────────────────────────────────────────────────────
_STATE_FILE = Path(__file__).parent.parent / "data" / "server_state.json"


def _load_persisted_state() -> dict:
    """Load counters from the last run so they survive a server restart."""
    if _STATE_FILE.exists():
        try:
            with open(_STATE_FILE) as f:
                saved = json.load(f)
            return {
                "flights_today":     int(saved.get("flights_today",     0)),
                "anomalies_flagged": int(saved.get("anomalies_flagged", 0)),
            }
        except Exception:
            pass
    return {"flights_today": 0, "anomalies_flagged": 0}


def _save_state_to_disk():
    """Persist counters to disk (called after every detection cycle)."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATE_FILE, "w") as f:
            json.dump(
                {
                    "flights_today":     STATE["flights_today"],
                    "anomalies_flagged": STATE["anomalies_flagged"],
                },
                f,
            )
    except Exception:
        pass


def _load_model_accuracy() -> float:
    """
    Read the F1 score that was computed when the ensemble was last trained.
    Falls back to None if no training has been done yet.
    """
    weights_path = Path(__file__).parent.parent / "models" / "ensemble_weights.json"
    # Also check the src/models/ sibling directory
    alt_path = Path(__file__).parent / "models" / "ensemble_weights.json"
    for p in (weights_path, alt_path):
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                # ensemble_weights.json may carry an f1 key written by detect.py
                f1 = data.get("f1_ensemble")
                if f1 is not None:
                    return round(float(f1) * 100, 1)  # return as percentage
            except Exception:
                pass
    return None  # Not yet trained — UI should show "N/A"


_persisted = _load_persisted_state()
STATE: dict = {
    "flights_today":     _persisted["flights_today"],
    "active_flights":    0,
    "anomalies_flagged": _persisted["anomalies_flagged"],
    "accuracy":          _load_model_accuracy(),   # None until first training run
    "avg_latency_ms":    0,
    "anomaly_log":       [],     # most recent anomalies — frontend table
    "live_log":          [],     # terminal log lines — frontend feed
}
MAX_LOG     = cfg.MAX_LOG_ENTRIES
_state_lock = threading.Lock()

# ── ML worker subprocess ──────────────────────────────────────────────────────
_WORKER_RESULTS = Path(__file__).parent.parent / "data" / "worker_results.json"
_WORKER_CMD     = Path(__file__).parent.parent / "data" / "worker_cmd.json"
_worker_proc: subprocess.Popen | None = None
_last_results_mtime: float = 0.0


def _start_ml_worker() -> subprocess.Popen:
    """
    Launch ml_worker.py as a child subprocess.
    The ML pipeline (TensorFlow/sklearn) runs in a separate OS process —
    a crash there cannot kill the API server or drop SSE connections.
    """
    python = sys.executable
    worker = Path(__file__).parent / "ml_worker.py"
    proc = subprocess.Popen(
        [python, str(worker)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.info("ML worker started", extra={"val": f"pid={proc.pid}"})
    return proc


def send_worker_cmd(cmd: str) -> None:
    """Write a command to the worker IPC file (e.g. 'retrain', 'stop')."""
    try:
        _WORKER_CMD.parent.mkdir(parents=True, exist_ok=True)
        with open(_WORKER_CMD, "w") as f:
            json.dump({"cmd": cmd}, f)
    except OSError as e:
        log.warning("Could not write worker command: %s", e)


def _poll_worker_results() -> None:
    """
    Daemon thread: reads worker_results.json whenever it changes and
    merges the new detection batch into STATE.
    """
    global _last_results_mtime, _worker_proc
    while True:
        time.sleep(5)   # cheap polling — actual cycle is 30 s

        # Auto-restart dead worker
        if _worker_proc is not None and _worker_proc.poll() is not None:
            log.warning(
                "ML worker exited (code=%s) — restarting",
                _worker_proc.returncode,
            )
            _worker_proc = _start_ml_worker()

        if not _WORKER_RESULTS.exists():
            continue
        try:
            mtime = _WORKER_RESULTS.stat().st_mtime
        except OSError:
            continue
        if mtime <= _last_results_mtime:
            continue   # file hasn't changed

        _last_results_mtime = mtime
        try:
            with open(_WORKER_RESULTS) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.error("Failed to read worker results: %s", e)
            continue

        anomalies   = data.get("anomalies", [])
        batch_count = data.get("batch_count", 0)
        latency_ms  = data.get("latency_ms",  0)
        ts_now      = datetime.now(timezone.utc).strftime("%H:%M:%S")

        with _state_lock:
            STATE["flights_today"]     += batch_count
            STATE["active_flights"]     = batch_count
            STATE["anomalies_flagged"] += len(anomalies)
            STATE["avg_latency_ms"]     = latency_ms
            fresh_acc = _load_model_accuracy()
            if fresh_acc is not None:
                STATE["accuracy"] = fresh_acc
            for a in anomalies:
                a.setdefault("timestamp", ts_now)
            STATE["anomaly_log"] = (anomalies + STATE["anomaly_log"])[:MAX_LOG]

        _save_state_to_disk()

        for a in anomalies:
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
        if not anomalies and batch_count:
            log.info("All layers nominal", extra={"val": "anomalies=0"})

        _push_sse("stats",    _stats_payload())
        _push_sse("anomalies", STATE["anomaly_log"][:50])

# SSE subscriber queues — one asyncio.Queue per connected browser tab
_sse_queues: list[asyncio.Queue] = []
_loop: asyncio.AbstractEventLoop | None = None   # captured at startup

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Pydantic models for API
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=6, max_length=72)
    email: EmailStr

class AttackSimulationRequest(BaseModel):
    attack_type: str = Field("teleport", pattern=r"^(teleport|altitude|speed|drift)$", description="teleport | altitude | speed | drift")
    callsign: str = Field("GHOST99", max_length=12, pattern=r"^[A-Za-z0-9_ -]+$")
    icao24: str = Field("DEAD99", max_length=10, pattern=r"^[A-Za-z0-9]+$")
 
# ─────────────────────────────────────────────────────────────────────────────
# OpenSky credentials  (optional — metadata works anonymously at lower rate)
#   export OPENSKY_USER=your_username
#   export OPENSKY_PASS=your_password
# ─────────────────────────────────────────────────────────────────────────────
OPENSKY_USER  = os.getenv("OPENSKY_USER", "")
OPENSKY_PASS  = os.getenv("OPENSKY_PASS", "")
_opensky_auth = (OPENSKY_USER, OPENSKY_PASS) if OPENSKY_USER else None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — starts background worker on app startup
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop, _worker_proc
    _loop = asyncio.get_running_loop()

    # Suppress harmless Windows client disconnect stack traces
    def _client_disconnect_handler(loop, context):
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        loop.default_exception_handler(context)

    _loop.set_exception_handler(_client_disconnect_handler)

    # Background retraining scheduler (daemon thread)
    try:
        from trainer import start_training_scheduler
        start_training_scheduler()
        log.info("Trainer scheduler started", extra={"val": "interval=3600s"})
    except Exception as e:
        log.warning("Trainer not started: %s", e)

    # Initialize users database
    try:
        init_db()
        log.info("Authentication database ready", extra={"val": "users.db"})
    except Exception as e:
        log.error("Database initialization failed: %s", e)

    # Launch ML detection as a separate subprocess (crash-isolated)
    _worker_proc = _start_ml_worker()

    # Daemon thread polls worker_results.json and merges into STATE
    poller = threading.Thread(
        target=_poll_worker_results, daemon=True, name="SkyGate-ResultsPoller"
    )
    poller.start()

    log.info("SkyGate server ready", extra={"val": "http://localhost:5000"})

    yield   # ← app runs here

    # Graceful shutdown: tell the worker to stop
    send_worker_cmd("stop")
    if _worker_proc and _worker_proc.poll() is None:
        _worker_proc.terminate()
        try:
            _worker_proc.wait(timeout=5)
        except Exception:
            _worker_proc.kill()
    log.info("SkyGate shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SkyGate", lifespan=lifespan)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(OSError)
async def os_error_handler(request: Request, exc: OSError):
    """Catch Windows illegal path syntax errors (e.g. colons in StaticFiles URLs) and return 404."""
    return JSONResponse(status_code=404, content={"detail": "Not found"})
 
# CORS: restrict to specific origins in production via SKYGATE_CORS_ORIGINS env var.
# Defaults to localhost for development. Set to comma-separated origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=True,
)

# ── Security Headers Middleware ──────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response

FRONTEND_DIR = Path(__file__).parent / "frontend"
 
 



# ─────────────────────────────────────────────────────────────────────────────
# Logging + SSE push helpers
# ─────────────────────────────────────────────────────────────────────────────
def _log(level: str, msg: str, val: str = ""):
    entry = {"time": _now(), "level": level.lower(), "msg": msg, "val": val}
    with _state_lock:
        STATE["live_log"] = ([entry] + STATE["live_log"])[:MAX_LOG]
    _push_sse("log", entry)
 
 
def _push_sse(event: str, data):
    """Thread-safe push to all connected SSE clients."""
    if _loop is None:
        return
    payload = json.dumps(data)
 
    async def _enqueue():
        dead = []
        for q in list(_sse_queues):
            try:
                q.put_nowait({"event": event, "data": payload})
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            if q in _sse_queues:
                _sse_queues.remove(q)
 
    try:
        asyncio.run_coroutine_threadsafe(_enqueue(), _loop)
    except RuntimeError:
        pass
 
 
def _stats_payload() -> dict:
    accuracy = STATE["accuracy"]
    return {
        "flights_today":     STATE["flights_today"],
        "active_flights":    STATE["active_flights"],
        "anomalies_flagged": STATE["anomalies_flagged"],
        "accuracy":          accuracy if accuracy is not None else "N/A",
        "avg_latency_ms":    STATE["avg_latency_ms"],
    }
 
 
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Auth Routes
# ─────────────────────────────────────────────────────────────────────────────
 
@app.get("/api/health")
async def health_check():
    """Unauthenticated health probe for load balancers and uptime monitors."""
    with _state_lock:
        return JSONResponse({
            "status": "healthy",
            "flights_today": STATE["flights_today"],
            "anomalies_flagged": STATE["anomalies_flagged"],
        })


@app.post("/api/auth/signup")
@limiter.limit(cfg.RATE_LIMIT_SIGNUP)
async def signup(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username, 
        full_name=user.full_name, 
        email=user.email, 
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {
        "message": "User created successfully",
        "full_name": new_user.full_name
    }
 
@app.post("/api/auth/login")
@limiter.limit(cfg.RATE_LIMIT_LOGIN)
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "full_name": user.full_name or user.username
    }
 
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username: str = payload.get("sub")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user
 
# ─────────────────────────────────────────────────────────────────────────────
# API Routes  — must be declared BEFORE the static file mount below
# ─────────────────────────────────────────────────────────────────────────────
 
@app.get("/api/stats")
async def api_stats(current_user: User = Depends(get_current_user)):
    with _state_lock:
        return JSONResponse(_stats_payload())
 
 
@app.get("/api/anomalies")
async def api_anomalies(current_user: User = Depends(get_current_user)):
    with _state_lock:
        return JSONResponse(STATE["anomaly_log"][:50])


@app.post("/api/simulate-attack")
async def api_simulate_attack(
    req: AttackSimulationRequest = AttackSimulationRequest(),
    current_user: User = Depends(get_current_user)
):
    """
    Simulate an ADS-B anomaly/spoofing attack in real time.
    Generates synthetic flight telemetry with the chosen anomaly,
    runs it through the detection pipeline and SIEM classifier,
    and pushes the resulting anomaly to connected SSE streams.
    """
    import pandas as pd
    from pipeline import run_pipeline

    now = datetime.now(timezone.utc)
    prev = now - timedelta(seconds=10)

    callsign = req.callsign.upper().strip() or "GHOST99"
    icao = req.icao24.upper().strip() or "DEAD99"
    attack = req.attack_type.lower()

    # Normal baseline flights to provide context for feature deltas
    base_flights = [
        {'icao': '800001', 'callsign': 'IGO101', 'latitude': 18.5, 'longitude': 73.8, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 90.0, 'timestamp': prev, 'label': 0},
        {'icao': '800001', 'callsign': 'IGO101', 'latitude': 18.52, 'longitude': 73.82, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 90.0, 'timestamp': now, 'label': 0},
        {'icao': '800002', 'callsign': 'AIC202', 'latitude': 19.1, 'longitude': 74.2, 'altitude': 32000.0, 'velocity': 480.0, 'heading': 110.0, 'timestamp': prev, 'label': 0},
        {'icao': '800002', 'callsign': 'AIC202', 'latitude': 19.12, 'longitude': 74.22, 'altitude': 32000.0, 'velocity': 480.0, 'heading': 110.0, 'timestamp': now, 'label': 0},
        {'icao': '800003', 'callsign': 'AXB303', 'latitude': 17.8, 'longitude': 75.0, 'altitude': 28000.0, 'velocity': 420.0, 'heading': 85.0, 'timestamp': prev, 'label': 0},
        {'icao': '800003', 'callsign': 'AXB303', 'latitude': 17.82, 'longitude': 75.02, 'altitude': 28000.0, 'velocity': 420.0, 'heading': 85.0, 'timestamp': now, 'label': 0},
        {'icao': '800004', 'callsign': 'SEJ404', 'latitude': 20.0, 'longitude': 76.5, 'altitude': 36000.0, 'velocity': 460.0, 'heading': 95.0, 'timestamp': prev, 'label': 0},
        {'icao': '800004', 'callsign': 'SEJ404', 'latitude': 20.02, 'longitude': 76.52, 'altitude': 36000.0, 'velocity': 460.0, 'heading': 95.0, 'timestamp': now, 'label': 0},
    ]

    # Injected target based on attack profile
    if attack == "altitude":
        p1 = {'icao': icao, 'callsign': callsign, 'latitude': 19.5, 'longitude': 78.0, 'altitude': 33000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': prev, 'label': 0}
        p2 = {'icao': icao, 'callsign': callsign, 'latitude': 19.52, 'longitude': 78.02, 'altitude': 54000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': now, 'label': 0}
    elif attack == "speed":
        p1 = {'icao': icao, 'callsign': callsign, 'latitude': 19.5, 'longitude': 78.0, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': prev, 'label': 0}
        p2 = {'icao': icao, 'callsign': callsign, 'latitude': 19.65, 'longitude': 78.15, 'altitude': 35000.0, 'velocity': 1150.0, 'heading': 45.0, 'timestamp': now, 'label': 0}
    elif attack == "drift":
        p1 = {'icao': icao, 'callsign': callsign, 'latitude': 19.5, 'longitude': 78.0, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': prev, 'label': 0}
        p2 = {'icao': icao, 'callsign': callsign, 'latitude': 19.58, 'longitude': 78.08, 'altitude': 35000.0, 'velocity': 680.0, 'heading': 175.0, 'timestamp': now, 'label': 0}
    else:  # teleport
        p1 = {'icao': icao, 'callsign': callsign, 'latitude': 19.5, 'longitude': 78.0, 'altitude': 35000.0, 'velocity': 450.0, 'heading': 45.0, 'timestamp': prev, 'label': 0}
        p2 = {'icao': icao, 'callsign': callsign, 'latitude': 23.5, 'longitude': 82.0, 'altitude': 48000.0, 'velocity': 980.0, 'heading': 45.0, 'timestamp': now, 'label': 0}

    df = pd.DataFrame(base_flights + [p1, p2])
    anomalies = await asyncio.to_thread(run_pipeline, df)
    sim_anomalies = [a for a in anomalies if a.get("icao24") == icao] or anomalies

    ts_now = now.strftime("%H:%M:%S")
    with _state_lock:
        STATE["anomalies_flagged"] += len(sim_anomalies)
        for a in sim_anomalies:
            a["timestamp"] = ts_now
        STATE["anomaly_log"] = (sim_anomalies + STATE["anomaly_log"])[:MAX_LOG]

    _save_state_to_disk()

    # Log & Broadcast
    for a in sim_anomalies:
        _log(
            "CRITICAL" if a.get("quarantine") else "WARNING",
            f"SIMULATED ATTACK DETECTED: {a.get('callsign')} [{a.get('icao24')}] - {a.get('threat_class')}",
            f"score={a.get('score')} quarantine={a.get('quarantine')}"
        )

    _push_sse("stats", _stats_payload())
    _push_sse("anomalies", STATE["anomaly_log"][:50])

    return JSONResponse({
        "status": "attack_injected",
        "attack_type": attack,
        "anomalies_detected": len(sim_anomalies),
        "results": sim_anomalies
    })
 
 
@app.get("/api/logs")
async def api_logs(current_user: User = Depends(get_current_user)):
    with _state_lock:
        return JSONResponse(list(reversed(STATE["live_log"][:50])))


# ─────────────────────────────────────────────────────────────────────────────
# SIEM Dashboard API
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/siem/summary")
async def siem_summary(current_user: User = Depends(get_current_user)):
    """KPI cards: totals, worst severity, quarantine count, mean score."""
    with _state_lock:
        log_copy = list(STATE["anomaly_log"])
        stats    = _stats_payload()

    if not log_copy:
        return JSONResponse({
            "total_anomalies":  0,
            "critical_count":   0,
            "quarantine_count": 0,
            "mean_score":       0.0,
            "top_threat_class": "N/A",
            "accuracy":         stats["accuracy"],
            "flights_today":    stats["flights_today"],
            "avg_latency_ms":   stats["avg_latency_ms"],
        })

    sev_order = {"critical": 0, "warn": 1, "info": 2}
    critical  = sum(1 for a in log_copy if a.get("severity") == "critical")
    quarantine = sum(1 for a in log_copy if a.get("quarantine"))
    scores    = [float(a.get("score", 0)) for a in log_copy]
    mean_sc   = round(sum(scores) / len(scores), 3) if scores else 0.0

    threat_counts: dict[str, int] = {}
    for a in log_copy:
        tc = a.get("threat_class", "Unknown")
        threat_counts[tc] = threat_counts.get(tc, 0) + 1
    top_threat = max(threat_counts, key=threat_counts.get) if threat_counts else "N/A"

    return JSONResponse({
        "total_anomalies":  len(log_copy),
        "critical_count":   critical,
        "quarantine_count": quarantine,
        "mean_score":       mean_sc,
        "top_threat_class": top_threat,
        "accuracy":         stats["accuracy"],
        "flights_today":    stats["flights_today"],
        "avg_latency_ms":   stats["avg_latency_ms"],
    })


@app.get("/api/siem/threat-breakdown")
async def siem_threat_breakdown(current_user: User = Depends(get_current_user)):
    """Donut chart data: anomaly count per threat class."""
    with _state_lock:
        log_copy = list(STATE["anomaly_log"])

    counts: dict[str, int] = {}
    for a in log_copy:
        tc = a.get("threat_class", "Unknown")
        counts[tc] = counts.get(tc, 0) + 1

    # Always include the full taxonomy so the chart doesn't jump when empty
    KNOWN = [
        "Malicious Spoofing", "SDR Ghost Track",
        "GPS Jamming / Glitch", "Altimeter Tampering",
        "Broadcast Integrity Failure", "Kinematic Violation (Verified)",
        "Nominal",
    ]
    result = [{"label": k, "count": counts.get(k, 0)} for k in KNOWN]
    # Append any unlabelled entries
    for k, v in counts.items():
        if k not in KNOWN:
            result.append({"label": k, "count": v})

    return JSONResponse(result)


@app.get("/api/siem/timeline")
async def siem_timeline(current_user: User = Depends(get_current_user)):
    """
    Hourly anomaly buckets for the past 12 hours.
    Returns list of {"hour": "HH:00", "count": N, "critical": N}.
    """
    from datetime import datetime, timezone, timedelta

    now    = datetime.now(timezone.utc)
    hours  = [(now - timedelta(hours=i)).strftime("%H:00") for i in range(11, -1, -1)]
    buckets: dict[str, dict] = {h: {"count": 0, "critical": 0} for h in hours}

    with _state_lock:
        log_copy = list(STATE["anomaly_log"])

    for a in log_copy:
        ts_str = a.get("timestamp", "")
        if not ts_str:
            continue
        try:
            # Timestamps stored as "HH:MM:SS" — bucket by hour
            hour = ts_str[:2] + ":00"
            if hour in buckets:
                buckets[hour]["count"] += 1
                if a.get("severity") == "critical":
                    buckets[hour]["critical"] += 1
        except Exception:
            pass

    return JSONResponse([{"hour": h, **v} for h, v in buckets.items()])


@app.get("/api/siem/top-threats")
async def siem_top_threats(current_user: User = Depends(get_current_user)):
    """Top 10 most recent high-severity anomalies for the threat table."""
    with _state_lock:
        log_copy = list(STATE["anomaly_log"])

    sev_rank = {"critical": 0, "warn": 1, "info": 2}
    sorted_a = sorted(
        log_copy,
        key=lambda a: (sev_rank.get(a.get("severity", "info"), 9), -float(a.get("score", 0))),
    )[:10]

    return JSONResponse([{
        "callsign":        a.get("callsign",    "—"),
        "icao24":          a.get("icao24",      "—"),
        "severity":        a.get("severity",    "info"),
        "threat_class":    a.get("threat_class","Unknown"),
        "score":           round(float(a.get("score", 0)), 3),
        "mlat_verified":   a.get("mlat_verified",   True),
        "valid_signature": a.get("valid_signature",  True),
        "quarantine":      a.get("quarantine",       False),
        "rssi_dbm":        a.get("rssi_dbm",        None),
        "anomaly":         a.get("anomaly",         "—"),
        "layer":           a.get("layer",           "—"),
        "timestamp":       a.get("timestamp",       "—"),
    } for a in sorted_a])


# ─────────────────────────────────────────────────────────────────────────
# Airspace Management Endpoints
# ─────────────────────────────────────────────────────────────────────────

# South Asian airspace regions — comprehensive coverage
AIRSPACE_REGIONS = []
# ─────────────────────────────────────────────────────────────────────────
# Airspace Management Endpoints — Global Coverage
# ─────────────────────────────────────────────────────────────────────────

# Comprehensive global airspace regions organized by continent/region
AIRSPACE_REGIONS = [
    # ══════════════════════════════════════════════════════════════════════
    # SOUTH ASIA (11 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "india", "name": "India", "country": "IND", "region": "South Asia", "lat_min": 6.0, "lat_max": 35.0, "lon_min": 68.0, "lon_max": 97.0, "description": "Indian airspace"},
    {"id": "pakistan", "name": "Pakistan", "country": "PAK", "region": "South Asia", "lat_min": 23.5, "lat_max": 37.1, "lon_min": 61.0, "lon_max": 77.0, "description": "Pakistani airspace"},
    {"id": "bangladesh", "name": "Bangladesh", "country": "BGD", "region": "South Asia", "lat_min": 21.6, "lat_max": 26.6, "lon_min": 88.0, "lon_max": 92.6, "description": "Bangladeshi airspace"},
    {"id": "nepal", "name": "Nepal", "country": "NPL", "region": "South Asia", "lat_min": 26.4, "lat_max": 30.2, "lon_min": 80.0, "lon_max": 88.2, "description": "Nepalese airspace"},
    {"id": "bhutan", "name": "Bhutan", "country": "BTN", "region": "South Asia", "lat_min": 26.8, "lat_max": 28.3, "lon_min": 88.7, "lon_max": 92.1, "description": "Bhutanese airspace"},
    {"id": "sri_lanka", "name": "Sri Lanka", "country": "LKA", "region": "South Asia", "lat_min": 5.9, "lat_max": 7.7, "lon_min": 80.6, "lon_max": 81.9, "description": "Sri Lankan airspace"},
    {"id": "maldives", "name": "Maldives", "country": "MDV", "region": "South Asia", "lat_min": -0.6, "lat_max": 5.4, "lon_min": 72.7, "lon_max": 73.7, "description": "Maldivian airspace"},
    {"id": "myanmar", "name": "Myanmar", "country": "MMR", "region": "South Asia", "lat_min": 9.0, "lat_max": 28.5, "lon_min": 92.1, "lon_max": 101.2, "description": "Myanmar airspace"},
    {"id": "thailand", "name": "Thailand", "country": "THA", "region": "South Asia", "lat_min": 5.6, "lat_max": 20.5, "lon_min": 97.3, "lon_max": 105.6, "description": "Thai airspace"},
    {"id": "arabian_sea", "name": "Arabian Sea", "country": "INT", "region": "South Asia", "lat_min": 6.0, "lat_max": 25.0, "lon_min": 52.0, "lon_max": 68.0, "description": "International airspace - Arabian Sea"},
    {"id": "bay_of_bengal", "name": "Bay of Bengal", "country": "INT", "region": "South Asia", "lat_min": 8.0, "lat_max": 22.0, "lon_min": 85.0, "lon_max": 95.0, "description": "International airspace - Bay of Bengal"},

    # ══════════════════════════════════════════════════════════════════════
    # SOUTHEAST ASIA (8 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "vietnam", "name": "Vietnam", "country": "VNM", "region": "Southeast Asia", "lat_min": 8.5, "lat_max": 23.4, "lon_min": 102.1, "lon_max": 109.5, "description": "Vietnamese airspace"},
    {"id": "cambodia", "name": "Cambodia", "country": "KHM", "region": "Southeast Asia", "lat_min": 10.0, "lat_max": 14.7, "lon_min": 102.3, "lon_max": 107.6, "description": "Cambodian airspace"},
    {"id": "laos", "name": "Laos", "country": "LAO", "region": "Southeast Asia", "lat_min": 13.9, "lat_max": 22.5, "lon_min": 100.0, "lon_max": 107.7, "description": "Laotian airspace"},
    {"id": "indonesia", "name": "Indonesia", "country": "IDN", "region": "Southeast Asia", "lat_min": -10.4, "lat_max": 6.0, "lon_min": 95.0, "lon_max": 141.0, "description": "Indonesian airspace"},
    {"id": "malaysia", "name": "Malaysia", "country": "MYS", "region": "Southeast Asia", "lat_min": 1.3, "lat_max": 7.3, "lon_min": 99.6, "lon_max": 119.3, "description": "Malaysian airspace"},
    {"id": "singapore", "name": "Singapore", "country": "SGP", "region": "Southeast Asia", "lat_min": 1.2, "lat_max": 1.5, "lon_min": 103.6, "lon_max": 104.0, "description": "Singaporean airspace"},
    {"id": "philippines", "name": "Philippines", "country": "PHL", "region": "Southeast Asia", "lat_min": 5.0, "lat_max": 19.5, "lon_min": 116.9, "lon_max": 126.6, "description": "Philippine airspace"},
    {"id": "brunei", "name": "Brunei", "country": "BRN", "region": "Southeast Asia", "lat_min": 4.0, "lat_max": 5.1, "lon_min": 114.0, "lon_max": 115.4, "description": "Bruneian airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # EAST ASIA (7 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "china", "name": "China", "country": "CHN", "region": "East Asia", "lat_min": 18.2, "lat_max": 53.6, "lon_min": 73.5, "lon_max": 135.0, "description": "Chinese airspace"},
    {"id": "japan", "name": "Japan", "country": "JPN", "region": "East Asia", "lat_min": 24.0, "lat_max": 45.6, "lon_min": 123.0, "lon_max": 145.8, "description": "Japanese airspace"},
    {"id": "south_korea", "name": "South Korea", "country": "KOR", "region": "East Asia", "lat_min": 33.1, "lat_max": 43.0, "lon_min": 124.5, "lon_max": 131.9, "description": "South Korean airspace"},
    {"id": "north_korea", "name": "North Korea", "country": "PRK", "region": "East Asia", "lat_min": 37.6, "lat_max": 43.0, "lon_min": 124.1, "lon_max": 130.8, "description": "North Korean airspace"},
    {"id": "mongolia", "name": "Mongolia", "country": "MNG", "region": "East Asia", "lat_min": 41.6, "lat_max": 52.0, "lon_min": 87.7, "lon_max": 119.8, "description": "Mongolian airspace"},
    {"id": "taiwan", "name": "Taiwan", "country": "TWN", "region": "East Asia", "lat_min": 21.9, "lat_max": 25.3, "lon_min": 120.0, "lon_max": 121.9, "description": "Taiwanese airspace"},
    {"id": "hongkong_macau", "name": "Hong Kong & Macau", "country": "HKG", "region": "East Asia", "lat_min": 21.9, "lat_max": 22.6, "lon_min": 113.8, "lon_max": 114.4, "description": "HK/Macau airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # MIDDLE EAST (12 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "saudi_arabia", "name": "Saudi Arabia", "country": "SAU", "region": "Middle East", "lat_min": 16.3, "lat_max": 32.1, "lon_min": 34.4, "lon_max": 55.9, "description": "Saudi Arabian airspace"},
    {"id": "uae", "name": "UAE", "country": "ARE", "region": "Middle East", "lat_min": 22.4, "lat_max": 26.1, "lon_min": 51.6, "lon_max": 56.4, "description": "UAE airspace"},
    {"id": "qatar", "name": "Qatar", "country": "QAT", "region": "Middle East", "lat_min": 24.5, "lat_max": 26.2, "lon_min": 50.7, "lon_max": 51.6, "description": "Qatari airspace"},
    {"id": "bahrain", "name": "Bahrain", "country": "BHR", "region": "Middle East", "lat_min": 26.1, "lat_max": 26.2, "lon_min": 50.2, "lon_max": 50.6, "description": "Bahraini airspace"},
    {"id": "kuwait", "name": "Kuwait", "country": "KWT", "region": "Middle East", "lat_min": 28.5, "lat_max": 30.1, "lon_min": 46.5, "lon_max": 48.5, "description": "Kuwaiti airspace"},
    {"id": "iraq", "name": "Iraq", "country": "IRQ", "region": "Middle East", "lat_min": 29.1, "lat_max": 37.4, "lon_min": 38.8, "lon_max": 48.8, "description": "Iraqi airspace"},
    {"id": "iran", "name": "Iran", "country": "IRN", "region": "Middle East", "lat_min": 25.1, "lat_max": 39.8, "lon_min": 44.0, "lon_max": 60.5, "description": "Iranian airspace"},
    {"id": "oman", "name": "Oman", "country": "OMN", "region": "Middle East", "lat_min": 16.6, "lat_max": 26.4, "lon_min": 51.9, "lon_max": 59.9, "description": "Omani airspace"},
    {"id": "yemen", "name": "Yemen", "country": "YEM", "region": "Middle East", "lat_min": 12.4, "lat_max": 19.0, "lon_min": 42.5, "lon_max": 53.2, "description": "Yemeni airspace"},
    {"id": "israel", "name": "Israel", "country": "ISR", "region": "Middle East", "lat_min": 29.5, "lat_max": 33.3, "lon_min": 34.2, "lon_max": 35.8, "description": "Israeli airspace"},
    {"id": "lebanon", "name": "Lebanon", "country": "LBN", "region": "Middle East", "lat_min": 32.0, "lat_max": 34.6, "lon_min": 35.1, "lon_max": 36.6, "description": "Lebanese airspace"},
    {"id": "syria", "name": "Syria", "country": "SYR", "region": "Middle East", "lat_min": 32.3, "lat_max": 37.3, "lon_min": 35.7, "lon_max": 42.5, "description": "Syrian airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # NORTH AFRICA (6 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "egypt", "name": "Egypt", "country": "EGY", "region": "North Africa", "lat_min": 22.0, "lat_max": 31.6, "lon_min": 24.7, "lon_max": 36.9, "description": "Egyptian airspace"},
    {"id": "libya", "name": "Libya", "country": "LBY", "region": "North Africa", "lat_min": 19.5, "lat_max": 33.1, "lon_min": 9.6, "lon_max": 25.3, "description": "Libyan airspace"},
    {"id": "tunisia", "name": "Tunisia", "country": "TUN", "region": "North Africa", "lat_min": 30.2, "lat_max": 37.5, "lon_min": 7.5, "lon_max": 11.6, "description": "Tunisian airspace"},
    {"id": "algeria", "name": "Algeria", "country": "DZA", "region": "North Africa", "lat_min": 18.9, "lat_max": 37.1, "lon_min": -8.6, "lon_max": 8.7, "description": "Algerian airspace"},
    {"id": "morocco", "name": "Morocco", "country": "MAR", "region": "North Africa", "lat_min": 27.1, "lat_max": 35.9, "lon_min": -13.2, "lon_max": -1.0, "description": "Moroccan airspace"},
    {"id": "western_sahara", "name": "Western Sahara", "country": "SAH", "region": "North Africa", "lat_min": 20.8, "lat_max": 27.7, "lon_min": -17.1, "lon_max": -8.7, "description": "Western Sahara airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # SUB-SAHARAN AFRICA (12 regions - Major)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "nigeria", "name": "Nigeria", "country": "NGA", "region": "Africa", "lat_min": 4.3, "lat_max": 13.9, "lon_min": 2.7, "lon_max": 14.7, "description": "Nigerian airspace"},
    {"id": "ghana", "name": "Ghana", "country": "GHA", "region": "Africa", "lat_min": 4.7, "lat_max": 11.2, "lon_min": -3.2, "lon_max": 1.2, "description": "Ghanaian airspace"},
    {"id": "kenya", "name": "Kenya", "country": "KEN", "region": "Africa", "lat_min": -4.7, "lat_max": 4.9, "lon_min": 33.9, "lon_max": 41.9, "description": "Kenyan airspace"},
    {"id": "south_africa", "name": "South Africa", "country": "ZAF", "region": "Africa", "lat_min": -34.8, "lat_max": -22.1, "lon_min": 16.4, "lon_max": 32.9, "description": "South African airspace"},
    {"id": "uganda", "name": "Uganda", "country": "UGA", "region": "Africa", "lat_min": -1.6, "lat_max": 4.2, "lon_min": 29.3, "lon_max": 35.0, "description": "Ugandan airspace"},
    {"id": "tanzania", "name": "Tanzania", "country": "TZA", "region": "Africa", "lat_min": -11.7, "lat_max": -0.9, "lon_min": 29.3, "lon_max": 40.3, "description": "Tanzanian airspace"},
    {"id": "ethiopia", "name": "Ethiopia", "country": "ETH", "region": "Africa", "lat_min": 3.4, "lat_max": 14.8, "lon_min": 32.9, "lon_max": 47.8, "description": "Ethiopian airspace"},
    {"id": "cameroon", "name": "Cameroon", "country": "CMR", "region": "Africa", "lat_min": 1.6, "lat_max": 13.0, "lon_min": 8.5, "lon_max": 16.0, "description": "Cameroonian airspace"},
    {"id": "congo", "name": "Congo/DRC", "country": "COD", "region": "Africa", "lat_min": -13.3, "lat_max": 5.3, "lon_min": 12.2, "lon_max": 31.3, "description": "Congo/DRC airspace"},
    {"id": "angola", "name": "Angola", "country": "AGO", "region": "Africa", "lat_min": -18.0, "lat_max": -4.4, "lon_min": 11.6, "lon_max": 24.1, "description": "Angolan airspace"},
    {"id": "mozambique", "name": "Mozambique", "country": "MOZ", "region": "Africa", "lat_min": -23.8, "lat_max": -10.3, "lon_min": 30.1, "lon_max": 40.8, "description": "Mozambican airspace"},
    {"id": "zimbabwe", "name": "Zimbabwe", "country": "ZWE", "region": "Africa", "lat_min": -22.4, "lat_max": -16.0, "lon_min": 25.2, "lon_max": 33.0, "description": "Zimbabwean airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # EUROPE (18 regions - Major)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "uk", "name": "United Kingdom", "country": "GBR", "region": "Europe", "lat_min": 50.0, "lat_max": 59.0, "lon_min": -8.0, "lon_max": 2.0, "description": "UK airspace"},
    {"id": "france", "name": "France", "country": "FRA", "region": "Europe", "lat_min": 42.3, "lat_max": 51.1, "lon_min": -8.2, "lon_max": 8.2, "description": "French airspace"},
    {"id": "germany", "name": "Germany", "country": "DEU", "region": "Europe", "lat_min": 47.2, "lat_max": 55.9, "lon_min": 5.9, "lon_max": 15.0, "description": "German airspace"},
    {"id": "italy", "name": "Italy", "country": "ITA", "region": "Europe", "lat_min": 36.6, "lat_max": 47.1, "lon_min": 6.6, "lon_max": 18.8, "description": "Italian airspace"},
    {"id": "spain", "name": "Spain", "country": "ESP", "region": "Europe", "lat_min": 36.0, "lat_max": 43.9, "lon_min": -9.3, "lon_max": 3.3, "description": "Spanish airspace"},
    {"id": "netherlands", "name": "Netherlands", "country": "NLD", "region": "Europe", "lat_min": 50.7, "lat_max": 53.5, "lon_min": 3.4, "lon_max": 7.2, "description": "Dutch airspace"},
    {"id": "belgium", "name": "Belgium", "country": "BEL", "region": "Europe", "lat_min": 49.5, "lat_max": 51.5, "lon_min": 2.4, "lon_max": 6.4, "description": "Belgian airspace"},
    {"id": "switzerland", "name": "Switzerland", "country": "CHE", "region": "Europe", "lat_min": 45.8, "lat_max": 47.8, "lon_min": 5.9, "lon_max": 10.5, "description": "Swiss airspace"},
    {"id": "austria", "name": "Austria", "country": "AUT", "region": "Europe", "lat_min": 46.4, "lat_max": 49.0, "lon_min": 9.5, "lon_max": 17.2, "description": "Austrian airspace"},
    {"id": "poland", "name": "Poland", "country": "POL", "region": "Europe", "lat_min": 49.0, "lat_max": 54.8, "lon_min": 14.1, "lon_max": 24.1, "description": "Polish airspace"},
    {"id": "czech", "name": "Czechia", "country": "CZE", "region": "Europe", "lat_min": 48.6, "lat_max": 51.1, "lon_min": 12.1, "lon_max": 18.9, "description": "Czech airspace"},
    {"id": "hungary", "name": "Hungary", "country": "HUN", "region": "Europe", "lat_min": 45.7, "lat_max": 48.6, "lon_min": 16.1, "lon_max": 22.9, "description": "Hungarian airspace"},
    {"id": "romania", "name": "Romania", "country": "ROU", "region": "Europe", "lat_min": 43.6, "lat_max": 48.3, "lon_min": 20.3, "lon_max": 29.7, "description": "Romanian airspace"},
    {"id": "greece", "name": "Greece", "country": "GRC", "region": "Europe", "lat_min": 34.8, "lat_max": 41.7, "lon_min": 19.3, "lon_max": 28.3, "description": "Greek airspace"},
    {"id": "portugal", "name": "Portugal", "country": "PRT", "region": "Europe", "lat_min": 36.9, "lat_max": 42.2, "lon_min": -9.5, "lon_max": -6.2, "description": "Portuguese airspace"},
    {"id": "russia_europe", "name": "Russia (European)", "country": "RUS", "region": "Europe", "lat_min": 41.2, "lat_max": 66.4, "lon_min": 19.6, "lon_max": 60.0, "description": "European Russia airspace"},
    {"id": "ukraine", "name": "Ukraine", "country": "UKR", "region": "Europe", "lat_min": 41.2, "lat_max": 52.4, "lon_min": 22.1, "lon_max": 40.2, "description": "Ukrainian airspace"},
    {"id": "turkey", "name": "Turkey", "country": "TUR", "region": "Europe", "lat_min": 35.8, "lat_max": 42.6, "lon_min": 25.6, "lon_max": 44.8, "description": "Turkish airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # NORTH AMERICA (5 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "usa", "name": "United States", "country": "USA", "region": "North America", "lat_min": 24.5, "lat_max": 49.4, "lon_min": -125.0, "lon_max": -66.9, "description": "US continental airspace"},
    {"id": "canada", "name": "Canada", "country": "CAN", "region": "North America", "lat_min": 41.7, "lat_max": 83.1, "lon_min": -141.0, "lon_max": -52.6, "description": "Canadian airspace"},
    {"id": "mexico", "name": "Mexico", "country": "MEX", "region": "North America", "lat_min": 14.5, "lat_max": 32.7, "lon_min": -117.1, "lon_max": -86.7, "description": "Mexican airspace"},
    {"id": "caribbean", "name": "Caribbean", "country": "INT", "region": "North America", "lat_min": 10.0, "lat_max": 27.0, "lon_min": -85.0, "lon_max": -60.0, "description": "Caribbean airspace"},
    {"id": "atlantic_north", "name": "North Atlantic", "country": "INT", "region": "North America", "lat_min": 20.0, "lat_max": 60.0, "lon_min": -80.0, "lon_max": -20.0, "description": "North Atlantic airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # CENTRAL & SOUTH AMERICA (8 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "central_america", "name": "Central America", "country": "INT", "region": "Central America", "lat_min": 7.0, "lat_max": 17.8, "lon_min": -92.2, "lon_max": -77.0, "description": "Central American airspace"},
    {"id": "colombia", "name": "Colombia", "country": "COL", "region": "South America", "lat_min": -4.2, "lat_max": 12.5, "lon_min": -81.8, "lon_max": -66.9, "description": "Colombian airspace"},
    {"id": "venezuela", "name": "Venezuela", "country": "VEN", "region": "South America", "lat_min": 0.6, "lat_max": 12.2, "lon_min": -73.6, "lon_max": -59.8, "description": "Venezuelan airspace"},
    {"id": "brazil", "name": "Brazil", "country": "BRA", "region": "South America", "lat_min": -33.8, "lat_max": 5.2, "lon_min": -74.0, "lon_max": -34.8, "description": "Brazilian airspace"},
    {"id": "peru", "name": "Peru", "country": "PER", "region": "South America", "lat_min": -17.5, "lat_max": -0.0, "lon_min": -81.3, "lon_max": -68.6, "description": "Peruvian airspace"},
    {"id": "chile", "name": "Chile", "country": "CHL", "region": "South America", "lat_min": -56.5, "lat_max": -17.5, "lon_min": -81.2, "lon_max": -66.4, "description": "Chilean airspace"},
    {"id": "argentina", "name": "Argentina", "country": "ARG", "region": "South America", "lat_min": -55.1, "lat_max": -21.8, "lon_min": -73.6, "lon_max": -53.6, "description": "Argentine airspace"},
    {"id": "atlantic_south", "name": "South Atlantic", "country": "INT", "region": "South America", "lat_min": -56.0, "lat_max": -10.0, "lon_min": -50.0, "lon_max": 0.0, "description": "South Atlantic airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # OCEANIA (5 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "australia", "name": "Australia", "country": "AUS", "region": "Oceania", "lat_min": -43.6, "lat_max": -10.7, "lon_min": 113.3, "lon_max": 153.6, "description": "Australian airspace"},
    {"id": "new_zealand", "name": "New Zealand", "country": "NZL", "region": "Oceania", "lat_min": -47.3, "lat_max": -34.4, "lon_min": 166.4, "lon_max": 178.5, "description": "New Zealand airspace"},
    {"id": "fiji", "name": "Fiji", "country": "FJI", "region": "Oceania", "lat_min": -18.3, "lat_max": -12.5, "lon_min": 177.0, "lon_max": -177.5, "description": "Fijian airspace"},
    {"id": "png", "name": "Papua New Guinea", "country": "PNG", "region": "Oceania", "lat_min": -12.2, "lat_max": -1.3, "lon_min": 140.8, "lon_max": 154.7, "description": "PNG airspace"},
    {"id": "pacific", "name": "Pacific Ocean", "country": "INT", "region": "Oceania", "lat_min": -40.0, "lat_max": 20.0, "lon_min": 130.0, "lon_max": 170.0, "description": "Pacific Ocean airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # CENTRAL ASIA & CAUCASUS (6 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "kazakhstan", "name": "Kazakhstan", "country": "KAZ", "region": "Central Asia", "lat_min": 40.6, "lat_max": 55.4, "lon_min": 51.9, "lon_max": 87.3, "description": "Kazakhstani airspace"},
    {"id": "uzbekistan", "name": "Uzbekistan", "country": "UZB", "region": "Central Asia", "lat_min": 37.2, "lat_max": 45.6, "lon_min": 55.4, "lon_max": 73.2, "description": "Uzbek airspace"},
    {"id": "tajikistan", "name": "Tajikistan", "country": "TJK", "region": "Central Asia", "lat_min": 36.7, "lat_max": 37.5, "lon_min": 70.8, "lon_max": 75.1, "description": "Tajikistani airspace"},
    {"id": "turkmenistan", "name": "Turkmenistan", "country": "TKM", "region": "Central Asia", "lat_min": 35.1, "lat_max": 43.6, "lon_min": 52.5, "lon_max": 66.6, "description": "Turkmen airspace"},
    {"id": "kyrgyzstan", "name": "Kyrgyzstan", "country": "KGZ", "region": "Central Asia", "lat_min": 39.2, "lat_max": 43.3, "lon_min": 69.3, "lon_max": 80.3, "description": "Kyrgyz airspace"},
    {"id": "afghanistan", "name": "Afghanistan", "country": "AFG", "region": "Central Asia", "lat_min": 29.3, "lat_max": 37.5, "lon_min": 60.5, "lon_max": 74.9, "description": "Afghan airspace"},

    # ══════════════════════════════════════════════════════════════════════
    # MAJOR INTERNATIONAL ZONES (3 regions)
    # ══════════════════════════════════════════════════════════════════════
    {"id": "atlantic", "name": "Atlantic Ocean", "country": "INT", "region": "International", "lat_min": -60.0, "lat_max": 80.0, "lon_min": -100.0, "lon_max": 20.0, "description": "Atlantic Ocean international airspace"},
    {"id": "indian_ocean", "name": "Indian Ocean", "country": "INT", "region": "International", "lat_min": -60.0, "lat_max": 20.0, "lon_min": 30.0, "lon_max": 130.0, "description": "Indian Ocean international airspace"},
    {"id": "polar", "name": "Polar Regions", "country": "INT", "region": "International", "lat_min": -90.0, "lat_max": -60.0, "lon_min": -180.0, "lon_max": 180.0, "description": "Polar and Antarctic airspace"},
]

AIRSPACE_STATE = {
    "active_regions": set(),  # Initially no regions active
    "last_updated": datetime.now(timezone.utc).isoformat(),
}


@app.get("/api/airspaces")
async def api_airspaces(current_user: User = Depends(get_current_user)):
    """Get all available airspace regions and current monitoring status."""
    return JSONResponse({
        "regions": AIRSPACE_REGIONS,
        "active": list(AIRSPACE_STATE["active_regions"]),
        "total": len(AIRSPACE_REGIONS),
    })


@app.get("/api/locate")
async def api_locate(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude between -90 and 90"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude between -180 and 180"),
    current_user: User = Depends(get_current_user)
):
    """Find which airspace region(s) contain the given lat/lon."""
    matching = []
    for region in AIRSPACE_REGIONS:
        if (region["lat_min"] <= lat <= region["lat_max"] and
            region["lon_min"] <= lon <= region["lon_max"]):
            matching.append({
                "id": region["id"],
                "name": region["name"],
                "active": region["id"] in AIRSPACE_STATE["active_regions"],
            })
    
    return JSONResponse({
        "latitude": lat,
        "longitude": lon,
        "regions": matching,
        "in_indian_airspace": len(matching) > 0,
    })


@app.patch("/api/airspace")
async def api_update_airspace(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Update active airspace regions for monitoring."""
    try:
        payload = await request.json()
        active = payload.get("active", [])
        
        # Validate that all requested regions exist
        valid_ids = {r["id"] for r in AIRSPACE_REGIONS}
        invalid = set(active) - valid_ids
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid region IDs: {invalid}"
            )
        
        AIRSPACE_STATE["active_regions"] = set(active)
        AIRSPACE_STATE["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        _log("INFO", "Airspace monitoring updated", f"active_regions={len(active)}/{len(AIRSPACE_REGIONS)}")
        
        return JSONResponse({
            "status": "updated",
            "active": list(AIRSPACE_STATE["active_regions"]),
            "timestamp": AIRSPACE_STATE["last_updated"],
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────
# Flight Detail  — combines adsb.lol (live state) + OpenSky (metadata/history)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/flight-detail/{icao24:path}")
async def flight_detail(icao24: str, current_user: User = Depends(get_current_user)):
    icao24 = icao24.lower().strip()
    if not re.match(r"^[0-9a-f]{1,8}$", icao24):
        raise HTTPException(status_code=400, detail="Invalid ICAO24 format: must be 1-8 hex characters")
    result: dict = {"icao24": icao24}
 
    async with httpx.AsyncClient(timeout=7.0) as client:
 
        async def _get(url: str, **kwargs):
            try:
                r = await client.get(url, **kwargs)
                return r.json() if r.status_code == 200 else None
            except Exception:
                return None
 
        adsblol_task = _get(f"https://api.adsb.lol/v2/icao/{icao24}")
 
        meta_kwargs  = {"auth": _opensky_auth} if _opensky_auth else {}
        meta_task    = _get(
            f"https://opensky-network.org/api/metadata/aircraft/icao/{icao24}",
            **meta_kwargs,
        )
 
        now = int(time.time())
        if _opensky_auth:
            # Authenticated: up to 24 hours back
            flt_task = _get(
                f"https://opensky-network.org/api/flights/aircraft"
                f"?icao24={icao24}&begin={now - 86400}&end={now}",
                auth=_opensky_auth,
            )
        else:
            # Anonymous: OpenSky usually limited to last 2 hours
            flt_task = _get(
                f"https://opensky-network.org/api/flights/aircraft"
                f"?icao24={icao24}&begin={now - 7200}&end={now}"
            )
 
        adsblol_data, meta_data, flt_data = await asyncio.gather(
            adsblol_task, meta_task, flt_task
        )
 
        if adsblol_data:
            ac_list = adsblol_data.get("ac") or adsblol_data.get("aircraft") or []
            if ac_list:
                ac = ac_list[0]
                alt_raw   = ac.get("alt_baro")
                on_ground = alt_raw == "ground" or bool(ac.get("on_ground", False))
                alt_ft    = None if (on_ground or alt_raw is None) else (
                    int(alt_raw) if isinstance(alt_raw, (int, float)) else None
                )
                result.update({
                    "callsign":      (ac.get("flight") or ac.get("callsign") or "").strip(),
                    "registration":   ac.get("r") or ac.get("reg") or "—",
                    "operator":       ac.get("ownOp") or ac.get("operator"),
                    "origin_country": ac.get("r")[:2] if ac.get("r") else "—",
                    "baro_altitude":  alt_ft,
                    "alt_unit":       "ft",
                    "velocity":       ac.get("gs"),
                    "vel_unit":       "kts",
                    "true_track":     ac.get("track"),
                    "vertical_rate":  ac.get("baro_rate") or ac.get("geom_rate"),
                    "on_ground":      on_ground,
                    "squawk":         str(ac.get("squawk") or ""),
                    "aircraft_type":  ac.get("t") or ac.get("type"),
                    "wake_category":  ac.get("category"),
                })
 
        if meta_data:
            result.update({
                "manufacturer":  meta_data.get("manufacturerName") or meta_data.get("manufacturer"),
                "aircraft_type": meta_data.get("typecode") or result.get("aircraft_type"),
                "wake_category": meta_data.get("wakeTurbulenceCategory") or result.get("wake_category"),
                "operator":      meta_data.get("operatorCallsign") or meta_data.get("operatorIcao"),
                "owner":         meta_data.get("owner"),
                "registration":  meta_data.get("registration") or result.get("registration"),
            })
 
        if flt_data and isinstance(flt_data, list) and len(flt_data) > 0:
            last = flt_data[-1]
            result["departure_airport"] = last.get("estDepartureAirport") or result.get("departure_airport")
            result["arrival_airport"]   = last.get("estArrivalAirport")   or result.get("arrival_airport")

        # ── ADSBDB Enrichment (Primary source for Route/Airline) ──────────────
        callsign = result.get("callsign")
        if callsign:
            try:
                # Run sync lookup in a thread to keep the server responsive
                route = await asyncio.to_thread(lookup_flight, callsign)
                if route:
                    if route.get("origin"):
                        result["departure_airport"] = route["origin"]
                        result["departure_city"]    = route["origin_city"]
                    if route.get("destination"):
                        result["arrival_airport"]   = route["destination"]
                        result["arrival_city"]      = route["destination_city"]
                    if route.get("airline") and not result.get("operator"):
                        result["operator"]          = route["airline"]
            except Exception as e:
                print(f"  [Server] Enrichment error for {callsign}: {e}")
 
    return JSONResponse(result)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SSE
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/stream")
async def api_stream(request: Request, token: str = None):
    # SSE usually doesn't send headers easily via EventSource,
    # so we allow token via query param for the stream.
    if not token:
        # Check Authorization header as fallback
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            token = auth.split(" ")[1]
    
    if not token or decode_access_token(token) is None:
        raise HTTPException(status_code=401, detail="Unauthorized stream access")

    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_queues.append(q)
 
    with _state_lock:
        seed_logs  = list(reversed(STATE["live_log"][:20]))
        seed_stats = _stats_payload()
 
    async def _generator():
        for entry in seed_logs:
            yield {"event": "log", "data": json.dumps(entry)}
        yield {"event": "stats", "data": json.dumps(seed_stats)}
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield msg
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": "{}"}
        except asyncio.CancelledError:
            pass
        finally:
            if q in _sse_queues:
                _sse_queues.remove(q)
 
    return EventSourceResponse(_generator())
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Static frontend — mounted LAST so API routes above always take priority.
#
# StaticFiles with html=True automatically serves:
#   /                  → frontend/index.html
#   /pipeline.html     → frontend/pipeline.html
#   /monitor.html      → frontend/monitor.html
#   /anomalies.html    → frontend/anomalies.html
#   /features.html     → frontend/features.html
#   /styles.css        → frontend/styles.css
#   /shared.js         → frontend/shared.js
#
# Place all HTML/CSS/JS files directly inside ./frontend/
# ─────────────────────────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/")
    async def index_fallback():
        return JSONResponse({
            "status": "SkyGate API running",
            "note": "Create a ./frontend/ folder and place index.html + other pages inside it."
        })
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("SkyGate dashboard -> http://localhost:%d", cfg.PORT)
    uvicorn.run(
        "server:app",
        host=cfg.HOST,
        port=cfg.PORT,
        reload=False,
    )