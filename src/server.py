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
from flight_lookup import lookup_flight
from auth         import verify_password, get_password_hash, create_access_token, decode_access_token
from models       import User, init_db, get_db
from sqlalchemy.orm import Session
from fastapi      import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic     import BaseModel, Field
 
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

# OAuth2 setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# Pydantic models for API
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=6, max_length=72)
    email: str
 
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

    # Initialize users database
    try:
        init_db()
        _log("INFO", "Authentication database ready", "users.db")
    except Exception as e:
        _log("ERROR", "Database initialization failed", str(e))
 
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
 
FRONTEND_DIR = Path(__file__).parent / "frontend"
 
 
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
# Auth Routes
# ─────────────────────────────────────────────────────────────────────────────
 
@app.post("/api/auth/signup")
async def signup(user: UserCreate, db: Session = Depends(get_db)):
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
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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
 
 
@app.get("/api/logs")
async def api_logs(current_user: User = Depends(get_current_user)):
    with _state_lock:
        return JSONResponse(list(reversed(STATE["live_log"][:50])))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Flight Detail  — combines adsb.lol (live state) + OpenSky (metadata/history)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/flight-detail/{icao24}")
async def flight_detail(icao24: str, current_user: User = Depends(get_current_user)):
    icao24 = icao24.lower().strip()
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
    print("SkyGate dashboard -> http://localhost:5000")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
    )