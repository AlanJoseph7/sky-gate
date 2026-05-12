---
name: testing-skygate
description: Test SkyGate's FastAPI server, auth flow, rate limiting, CORS, and security features end-to-end. Use when verifying auth, security, or stability changes.
---

# Testing SkyGate

## Prerequisites

- Python 3.12+ with dependencies from `requirements.txt`
- No external secrets required for local testing — JWT secret auto-generates, OpenSky credentials are optional
- Install deps: `cd /home/ubuntu/repos/sky-gate && pip install -r requirements.txt`

## Starting the Server

```bash
cd /home/ubuntu/repos/sky-gate/src
# Remove old users.db for a clean test
rm -f users.db
python server.py
```

- Server runs on `http://localhost:5000`
- Frontend is served from `src/frontend/` via FastAPI StaticFiles mount
- Expect TensorFlow CPU warnings (harmless)
- Expect `SKYGATE_SECRET_KEY` warning if env var not set (this is correct behavior)
- The autoencoder model may fail to load due to a pre-existing Keras compatibility issue — the pipeline gracefully degrades to Rule-Based + Isolation Forest

## Auth Flow

### Signup
- **URL**: `http://localhost:5000/auth.html` → click "Sign up"
- **Fields**: Full Name, Username, Email Address, Password
- **API**: `POST /api/auth/signup` with JSON body `{"username", "full_name", "password", "email"}`
- **Success**: Blue "Account created! Please sign in." message, auto-switches to login form
- **Email validation**: Uses Pydantic `EmailStr` — invalid emails return HTTP 422
- **Password validation**: min_length=6, max_length=72 — short passwords return HTTP 422

### Login
- **URL**: `http://localhost:5000/auth.html` (default mode)
- **API**: `POST /api/auth/login` with form data `username=...&password=...` (NOT JSON)
- **Success**: Welcome loader with user's full name, 3-second redirect to `index.html`
- **Token**: Stored in `localStorage` as `sg_token`
- **Note**: Login uses `application/x-www-form-urlencoded`, not JSON (OAuth2 password flow)

### Authenticated API Access
- All `/api/*` endpoints (except login/signup) require `Authorization: Bearer <token>` header
- SSE stream (`/api/stream`) accepts token via query param `?token=...`

## Rate Limiting

- Login: 10 requests/minute per IP
- Signup: 5 requests/minute per IP
- Rate-limited responses return HTTP 429
- **Important**: Rate limits are per-IP and use in-memory storage. Restarting the server resets all rate limit counters.
- When testing rate limits via shell before browser tests, be aware the browser login may get rate-limited. Run rate limit tests AFTER browser tests, or restart the server between them.

## CORS

- Default allowed origins: `http://localhost:5000,http://127.0.0.1:5000`
- Configurable via `SKYGATE_CORS_ORIGINS` env var (comma-separated)
- Test with: `curl -I -X OPTIONS http://localhost:5000/api/auth/login -H "Origin: http://evil.com" -H "Access-Control-Request-Method: POST"`
- Should NOT return `access-control-allow-origin` for unauthorized origins

## Code-Level Verification

These can be verified with grep without running the server:

```bash
# No hardcoded credentials
grep -c "Appa@2525" src/data_fetcher.py  # should be 0
grep -c "09d25e094faa6ca2" src/auth.py     # should be 0

# Env var usage
grep "os.environ.get.*OPENSKY" src/data_fetcher.py
grep "os.environ.get.*SKYGATE_SECRET_KEY" src/auth.py

# Correct function calls
grep "from features import" src/live_monitor.py  # should show compute_features_live
grep "FEATURE_COLUMNS" src/trainer.py             # should show import from utils

# Model paths use helpers
grep "get_model_path" src/isolation_forest.py src/lstm_model.py src/detect.py

# No merge conflicts
grep -c "<<<<<<" README.md  # should be 0
```

## Testing Order (Recommended)

1. Code verification tests (grep, no server needed)
2. Start server
3. Browser tests: signup → login → dashboard → monitor → anomalies (RECORD these)
4. Shell API tests: invalid email, duplicate username, short password, wrong password
5. Shell API tests: authenticated vs unauthenticated access
6. Rate limiting tests (do AFTER browser tests to avoid rate-limiting the browser)
7. CORS tests

## CI Notes

- Vercel deployment may fail — this is pre-existing (no `vercel.json` for Python project)
- No GitHub Actions CI configured in the repo

## Devin Secrets Needed

None required for local testing. Optional:
- `OPENSKY_USERNAME` / `OPENSKY_PASSWORD` — for authenticated OpenSky API access (higher rate limits)
- `SKYGATE_SECRET_KEY` — for persistent JWT tokens across server restarts
