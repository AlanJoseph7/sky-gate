"""
config.py — SkyGate Configuration Manager

Centralized configuration handling for environment variables and defaults.

Required environment variables
──────────────────────────────
  SKYGATE_SECRET_KEY   JWT signing key (min 32 chars, no default allowed)

Set them in a .env file at the project root or export them in your shell.
Example .env:
  SKYGATE_SECRET_KEY=<output of: python -c "import secrets; print(secrets.token_hex(32))">
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Sentinel — the old placeholder text that must never reach production
# ─────────────────────────────────────────────────────────────────────────────
_PLACEHOLDER_SECRET = "your-secret-key-change-this-in-production-use-secrets-token-hex-32"


class Config:
    """Configuration object with sensible defaults and environment overrides."""

    # ─────────────────────────────────────────────────────────────────────────
    # Authentication & Security
    # ─────────────────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = os.getenv("SKYGATE_SECRET_KEY", "")
    JWT_ALGORITHM  = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))

    def __init__(self):
        # ── Validate required secrets at startup ──────────────────────────────
        # This runs once when `cfg = Config()` is called at the bottom.
        # A missing or placeholder secret kills the process immediately —
        # never allow a known-public fallback to reach a running server.
        secret = self.JWT_SECRET_KEY
        if not secret or secret == _PLACEHOLDER_SECRET:
            print(
                "\n[FATAL] SKYGATE_SECRET_KEY is not set or is still the default placeholder.\n"
                "        Generate a secure key and add it to your .env file:\n"
                "        python -c \"import secrets; print(secrets.token_hex(32))\"\n"
                "        Then add to .env: SKYGATE_SECRET_KEY=<generated_key>\n",
                file=sys.stderr,
            )
            sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # OpenSky Network API (Optional — adsb.lol is free)
    # ─────────────────────────────────────────────────────────────────────────
    OPENSKY_USERNAME = os.getenv("OPENSKY_USERNAME", "")
    OPENSKY_PASSWORD = os.getenv("OPENSKY_PASSWORD", "")

    # ─────────────────────────────────────────────────────────────────────────
    # adsb.lol API Configuration
    # ─────────────────────────────────────────────────────────────────────────
    # Note: adsb.lol has various endpoints. Try /api/0/aircraft for all aircraft
    # or use /api/0/boxes/{lamin}/{lomin}/{lamax}/{lomax} for bounding boxes
    ADSBLOL_API_BASE = os.getenv("ADSBLOL_API_BASE", "https://api.adsb.lol")
    ADSBLOL_API_PATH = os.getenv(
        "ADSBLOL_API_PATH",
        "https://api.adsb.lol/api/0/aircraft"  # Fetch all aircraft, filter locally
    )
    ADSBLOL_API_KEY = os.getenv("ADSBLOL_API_KEY", "")

    # ─────────────────────────────────────────────────────────────────────────
    # Web Server Configuration
    # ─────────────────────────────────────────────────────────────────────────
    HOST = os.getenv("SKYGATE_HOST", "127.0.0.1")
    PORT = int(os.getenv("SKYGATE_PORT", "5000"))

    # ─────────────────────────────────────────────────────────────────────────
    # CORS Configuration
    # ─────────────────────────────────────────────────────────────────────────
    CORS_ORIGINS = os.getenv(
        "SKYGATE_CORS_ORIGINS",
        "http://localhost:5000,http://127.0.0.1:5000"
    ).split(",")

    # ─────────────────────────────────────────────────────────────────────────
    # Rate Limiting
    # ─────────────────────────────────────────────────────────────────────────
    RATE_LIMIT_SIGNUP = os.getenv("RATE_LIMIT_SIGNUP", "5/minute")
    RATE_LIMIT_LOGIN = os.getenv("RATE_LIMIT_LOGIN", "10/minute")

    # ─────────────────────────────────────────────────────────────────────────
    # Logging & State Management
    # ─────────────────────────────────────────────────────────────────────────
    MAX_LOG_ENTRIES = int(os.getenv("MAX_LOG_ENTRIES", "100"))

    # ─────────────────────────────────────────────────────────────────────────
    # Detection Pipeline Configuration
    # ─────────────────────────────────────────────────────────────────────────
    PIPELINE_INTERVAL_SECONDS = int(os.getenv("PIPELINE_INTERVAL_SECONDS", "30"))
    FETCH_INTERVAL_SECONDS = int(os.getenv("FETCH_INTERVAL_SECONDS", "30"))

    # ─────────────────────────────────────────────────────────────────────────
    # Model Configuration
    # ─────────────────────────────────────────────────────────────────────────
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
    ANOMALY_TYPES = [
        "altitude_jump",
        "teleportation",
        "speed_spike",
        "gradual_speed_drift",
        "altitude_oscillation",
        "heading_oscillation",
    ]


# Global config instance
cfg = Config()
