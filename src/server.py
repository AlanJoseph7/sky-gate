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
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
 
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
 
# ── SkyGate modules ───────────────────────────────────────────────────────────
from data_fetcher import fetch_live_data
from pipeline     import run_pipeline
 
# ─────────────────────────────────────────────────────────────────────────────
# Shared in-memory state
# ─────────────────────────────────────────────────────────────────────────────
STATE: dict = {
    "flights_today":     0,
    "anomalies_flagged": 0,
    "accuracy":          97.2,
    "avg_latency_ms":    0,
    "anomaly_log":       [],     # most recent anomalies — frontend table
    "live_log":          [],     # terminal log lines — frontend feed
}
MAX_LOG     = 200
_state_lock = threading.Lock()
 
# SSE subscriber queues — one asyncio.Queue per connected browser tab
_sse_queues: list[asyncio.Queue] = []
_loop: asyncio.AbstractEventLoop | None = None   # captured at startup
 
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
    global _loop
    _loop = asyncio.get_running_loop()
 
    # Background retraining scheduler (daemon thread)
    try:
        from trainer import start_training_scheduler
        start_training_scheduler()
        _log("INFO", "Trainer scheduler started", "interval=3600s")
    except Exception as e:
        _log("WARN", "Trainer not started", str(e))
 
    # Background fetch/detect worker
    worker = threading.Thread(
        target=_background_worker, daemon=True, name="SkyGate-Worker"
    )
    worker.start()
    _log("INFO", "SkyGate server ready", "http://localhost:5000")
 
    yield   # ← app runs here
 
 
# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SkyGate", lifespan=lifespan)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Serve static frontend from ./frontend/
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Background detection worker  (runs in its own daemon thread)
# ─────────────────────────────────────────────────────────────────────────────
def _background_worker():
    """Fetches live ADS-B data every 30 s and runs the full pipeline."""
    while True:
        t0 = time.time()
        try:
            flights     = fetch_live_data()
            batch_count = len(flights) if flights is not None and not flights.empty else 0
 
            _log("INFO", "Fetching ADS-B batch", f"flights={batch_count}")
 
            if flights is not None and not flights.empty:
                anomalies  = run_pipeline(flights)
                latency_ms = int((time.time() - t0) * 1000)
                ts_now     = _now()
 
                with _state_lock:
                    STATE["flights_today"]     += batch_count
                    STATE["anomalies_flagged"] += len(anomalies)
                    STATE["avg_latency_ms"]     = latency_ms
 
                    for a in anomalies:
                        a.setdefault("timestamp", ts_now)
 
                    STATE["anomaly_log"] = (
                        anomalies + STATE["anomaly_log"]
                    )[:MAX_LOG]
 
                if anomalies:
                    for a in anomalies:
                        _log(
                            "ALERT",
                            f"ANOMALY · {a.get('callsign','?')} [{a.get('icao24','?')}]",
                            f"severity={a.get('severity','?')} · layer={a.get('layer','?')} "
                            f"· score={a.get('score',0):.3f}",
                        )
                else:
                    _log("OK", "All layers nominal", "anomalies=0")
 
                _push_sse("stats", _stats_payload())
                _push_sse("anomalies", STATE["anomaly_log"][:50])
 
        except Exception as e:
            _log("ERROR", "Pipeline error", str(e))
 
        time.sleep(30)
 
 
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
    return {k: STATE[k] for k in
            ["flights_today", "anomalies_flagged", "accuracy", "avg_latency_ms"]}
 
 
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
 
@app.get("/")
async def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), headers={"Cache-Control": "no-cache"})
    return JSONResponse({"status": "SkyGate API running — place index.html in ./frontend/"})
 
 
@app.get("/api/stats")
async def api_stats():
    with _state_lock:
        return JSONResponse(_stats_payload())
 
 
@app.get("/api/anomalies")
async def api_anomalies():
    with _state_lock:
        return JSONResponse(STATE["anomaly_log"][:50])
 
 
@app.get("/api/logs")
async def api_logs():
    with _state_lock:
        return JSONResponse(list(reversed(STATE["live_log"][:50])))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Flight Detail  — combines adsb.lol (live state) + OpenSky (metadata/history)
#
#   adsb.lol  → altitude (ft), speed (kts), heading, squawk, vertical rate
#   OpenSky   → manufacturer, full type name, registration, owner/operator
#   OpenSky   → last known departure / arrival airports  (needs credentials)
#
#   All three calls are made concurrently with asyncio.gather so the response
#   is as fast as the slowest single API rather than their sum.
#
#   Failures are silently swallowed — the frontend falls back to mock data
#   for any field that is missing.
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/flight-detail/{icao24}")
async def flight_detail(icao24: str):
    icao24 = icao24.lower().strip()
    result: dict = {"icao24": icao24}
 
    async with httpx.AsyncClient(timeout=7.0) as client:
 
        # ── helper: silent fetch ──────────────────────────────────────────────
        async def _get(url: str, **kwargs):
            try:
                r = await client.get(url, **kwargs)
                return r.json() if r.status_code == 200 else None
            except Exception:
                return None
 
        # ── fire all three requests concurrently ──────────────────────────────
        adsblol_task = _get(f"https://api.adsb.lol/v2/icao/{icao24}")
 
        meta_kwargs  = {"auth": _opensky_auth} if _opensky_auth else {}
        meta_task    = _get(
            f"https://opensky-network.org/api/metadata/aircraft/icao/{icao24}",
            **meta_kwargs,
        )
 
        if _opensky_auth:
            now  = int(time.time())
            flt_task = _get(
                f"https://opensky-network.org/api/flights/aircraft"
                f"?icao24={icao24}&begin={now - 86400}&end={now}",
                auth=_opensky_auth,
            )
        else:
            flt_task = asyncio.sleep(0, result=None)   # no-op coroutine
 
        adsblol_data, meta_data, flt_data = await asyncio.gather(
            adsblol_task, meta_task, flt_task
        )
 
        # ── 1. adsb.lol live state ────────────────────────────────────────────
        if adsblol_data:
            ac_list = adsblol_data.get("ac") or adsblol_data.get("aircraft") or []
            if ac_list:
                ac = ac_list[0]
 
                # alt_baro is either a number (feet) or the string "ground"
                alt_raw   = ac.get("alt_baro")
                on_ground = alt_raw == "ground" or bool(ac.get("on_ground", False))
                alt_ft    = None if (on_ground or alt_raw is None) else (
                    int(alt_raw) if isinstance(alt_raw, (int, float)) else None
                )
 
                result.update({
                    "callsign":      (ac.get("flight") or ac.get("callsign") or "").strip(),
                    "registration":   ac.get("r") or "—",
                    "origin_country": ac.get("r")[:2] if ac.get("r") else "—",
                    "baro_altitude":  alt_ft,
                    "alt_unit":       "ft",      # tells frontend: skip m→ft conversion
                    "velocity":       ac.get("gs"),          # already in knots
                    "vel_unit":       "kts",     # tells frontend: skip m/s→kts conversion
                    "true_track":     ac.get("track"),
                    "vertical_rate":  ac.get("baro_rate") or ac.get("geom_rate"),
                    "on_ground":      on_ground,
                    "squawk":         str(ac.get("squawk") or ""),
                    # ICAO type code e.g. "A320" — may be overridden by OpenSky below
                    "aircraft_type":  ac.get("t") or ac.get("type"),
                    "wake_category":  ac.get("category"),
                })
 
        # ── 2. OpenSky metadata ───────────────────────────────────────────────
        if meta_data:
            result.update({
                # richer manufacturer name e.g. "Airbus" not just "AIR"
                "manufacturer":  meta_data.get("manufacturerName") or meta_data.get("manufacturer"),
                # OpenSky typecode is more reliable than adsb.lol "t" field
                "aircraft_type": meta_data.get("typecode") or result.get("aircraft_type"),
                "wake_category": meta_data.get("wakeTurbulenceCategory") or result.get("wake_category"),
                "operator":      meta_data.get("operatorCallsign") or meta_data.get("operatorIcao"),
                "owner":         meta_data.get("owner"),
                # OpenSky registration is canonical
                "registration":  meta_data.get("registration") or result.get("registration"),
            })
 
        # ── 3. OpenSky flights (dep / arr airports) ───────────────────────────
        if flt_data and isinstance(flt_data, list) and len(flt_data) > 0:
            last = flt_data[-1]   # most recently completed flight segment
            result["departure_airport"] = last.get("estDepartureAirport") or "???"
            result["arrival_airport"]   = last.get("estArrivalAirport")   or "???"
 
    return JSONResponse(result)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SSE  (keep this AFTER flight-detail so the route order is correct)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/stream")
async def api_stream(request: Request):
    """
    Server-Sent Events endpoint.
    Browser connects once; server pushes log lines, stats, and anomaly
    updates in real time without polling.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_queues.append(q)
 
    # Snapshot recent state for this new connection
    with _state_lock:
        seed_logs  = list(reversed(STATE["live_log"][:20]))
        seed_stats = _stats_payload()
 
    async def _generator():
        # Seed existing log lines so terminal isn't empty on load
        for entry in seed_logs:
            yield {"event": "log", "data": json.dumps(entry)}
 
        # Seed current stats immediately
        yield {"event": "stats", "data": json.dumps(seed_stats)}
 
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield msg
                except asyncio.TimeoutError:
                    # Heartbeat keeps load-balancers / proxies alive
                    yield {"event": "heartbeat", "data": "{}"}
        except asyncio.CancelledError:
            pass
        finally:
            if q in _sse_queues:
                _sse_queues.remove(q)
 
    return EventSourceResponse(_generator())
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("SkyGate dashboard -> http://localhost:5000")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5000,
        reload=False,""
        log_level="info",
    )