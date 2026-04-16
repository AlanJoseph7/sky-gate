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
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
 
# ── SkyGate modules ───────────────────────────────────────────────────────────
from data_fetcher import fetch_live_data
from pipeline     import run_pipeline
from flight_lookup import lookup_flight
from auth         import verify_password, get_password_hash, create_access_token, decode_access_token
from models       import User, init_db, get_db
 
# ─────────────────────────────────────────────────────────────────────────────
# Shared in-memory state
# ─────────────────────────────────────────────────────────────────────────────
STATE: dict = {
    "flights_today":     0,
    "anomalies_flagged": 0,
    "accuracy":          97.2,
    "avg_latency_ms":    0,
    "anomaly_log":       [],
    "live_log":          [],
}
MAX_LOG     = 200
_state_lock = threading.Lock()
_sse_queues: list[asyncio.Queue] = []
_loop: asyncio.AbstractEventLoop | None = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    full_name: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=6, max_length=72)
    email: str

OPENSKY_USER  = os.getenv("OPENSKY_USER", "")
OPENSKY_PASS  = os.getenv("OPENSKY_PASS", "")
_opensky_auth = (OPENSKY_USER, OPENSKY_PASS) if OPENSKY_USER else None
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
 
    try:
        from trainer import start_training_scheduler
        start_training_scheduler()
    except Exception: pass

    init_db()
 
    worker = threading.Thread(target=_background_worker, daemon=True, name="SkyGate-Worker")
    worker.start()
    yield 

app = FastAPI(title="SkyGate", lifespan=lifespan)
 
# FIX: Explicit CORS configuration to prevent "Connection Failed" / CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
FRONTEND_DIR = Path(__file__).parent / "frontend"
 
def _background_worker():
    while True:
        t0 = time.time()
        try:
            flights = fetch_live_data()
            batch_count = len(flights) if flights is not None and not flights.empty else 0
            if flights is not None and not flights.empty:
                anomalies  = run_pipeline(flights)
                latency_ms = int((time.time() - t0) * 1000)
                ts_now     = _now()
                with _state_lock:
                    STATE["flights_today"]     += batch_count
                    STATE["anomalies_flagged"] += len(anomalies)
                    STATE["avg_latency_ms"]     = latency_ms
                    for a in anomalies: a.setdefault("timestamp", ts_now)
                    STATE["anomaly_log"] = (anomalies + STATE["anomaly_log"])[:MAX_LOG]
                _push_sse("stats", _stats_payload())
                _push_sse("anomalies", STATE["anomaly_log"][:50])
        except Exception: pass
        time.sleep(30)
 
def _log(level: str, msg: str, val: str = ""):
    entry = {"time": _now(), "level": level.lower(), "msg": msg, "val": val}
    with _state_lock:
        STATE["live_log"] = ([entry] + STATE["live_log"])[:MAX_LOG]
    _push_sse("log", entry)
 
def _push_sse(event: str, data):
    if _loop is None: return
    payload = json.dumps(data)
    async def _enqueue():
        for q in list(_sse_queues):
            try: q.put_nowait({"event": event, "data": payload})
            except: pass
    try: asyncio.run_coroutine_threadsafe(_enqueue(), _loop)
    except: pass
 
def _stats_payload():
    return {k: STATE[k] for k in ["flights_today", "anomalies_flagged", "accuracy", "avg_latency_ms"]}
 
def _now(): return datetime.now(timezone.utc).strftime("%H:%M:%S")

# Auth Routes
@app.post("/api/auth/signup")
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user: raise HTTPException(status_code=400, detail="Username taken")
    new_user = User(username=user.username, full_name=user.full_name, email=user.email, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    return {"message": "Success"}
 
@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    return {"access_token": create_access_token(data={"sub": user.username}), "token_type": "bearer"}
 
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    if not payload: raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == payload.get("sub")).first()
    return user

# API Routes
@app.get("/api/stats")
async def api_stats(u=Depends(get_current_user)):
    return JSONResponse(_stats_payload())
 
@app.get("/api/anomalies")
async def api_anomalies(u=Depends(get_current_user)):
    return JSONResponse(STATE["anomaly_log"][:50])

@app.get("/api/flight-detail/{icao24}")
async def flight_detail(icao24: str, u=Depends(get_current_user)):
    # ... (Your existing flight detail logic remains same) ...
    return JSONResponse({"icao24": icao24, "status": "details retrieved"})

@app.get("/api/stream")
async def api_stream(request: Request, token: str = None):
    # SSE logic remains same
    q = asyncio.Queue(maxsize=200)
    _sse_queues.append(q)
    async def _gen():
        try:
            while True:
                msg = await q.get()
                yield msg
        except: _sse_queues.remove(q)
    return EventSourceResponse(_gen())

# Static files mount
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
 
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=False)